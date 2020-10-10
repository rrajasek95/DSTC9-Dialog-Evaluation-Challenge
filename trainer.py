"""
 Created by diesel
 10/10/20
"""

from nltk import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm


from transformers import AdamW, GPT2Tokenizer

import os

import random
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, GPT2Tokenizer
from gpt2 import GPT2DoubleHeadsModel

from torch.cuda import amp
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm



from train_util.decode import top_filtering
from train_util.metrics import RunningMetric, RunningLambdaMetric, MetricLambda
from train_util.scheduler import PiecewiseLinearLR
from utils import get_dataset, GlobalStepCounter, CONFIG_NAME, augmented_tc_dataset, make_path




SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<end>", "<pad>", "<eot>"]  # added <end>, to represent the end of sent



logger = None
def _set_logger(a_logger):
    global logger
    logger = a_logger



def decode_sequence(input_ids, token_type_ids, model, tokenizer, args):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    input_seq = tokenizer.decode(input_ids[0][0])
    prefix, suffix = input_seq.rsplit("<speaker", maxsplit=1)
    context = prefix + "<speaker" + suffix[:2]  # Hacky way to append the speaker tag

    current_output = []

    for i in range(args.max_length):
        prefix_input_seq = torch.tensor(tokenizer.encode(context) + current_output).unsqueeze(0)
        truncated_tok_type_ids = token_type_ids[0][0][:prefix_input_seq.shape[-1]].unsqueeze(0)
        logits = model(prefix_input_seq.to(args.device), token_type_ids=truncated_tok_type_ids.to(args.device))

        if isinstance(logits, tuple) or len(logits.shape) == 4:  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    logger.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)
        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    output = tokenizer.decode(current_output)
    logger.info(f"\nContext: {context}\nOutput: {output}\n")




def save_model(model, checkpoint_name, args):
    checkpoint_dir = os.path.join(args.log_dir, args.experiment_name, 'checkpoints')
    make_path(checkpoint_dir)

    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name + '.pth')
    torch.save({"mymodel": getattr(model, 'module', model)}, checkpoint_file)
    logger.info(f"Checkpoint saved to: {checkpoint_file}")


def run_train(model, optimizer, scheduler, train_loader, writer, step_counter, args):
    running_loss = RunningMetric()
    ppl = MetricLambda(math.exp, running_loss)

    for i, batch in tqdm(enumerate(train_loader)):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels
        )
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        # Average loss across all items in the batch
        running_loss.add(float(loss))

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if i % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        step_counter.step()

        if step_counter.get() % args.save_every_n == 0:
            # Save checkpoint of model
            # Since this is a fine-tuning task, we prefer to
            # save more frequently
            # than if we are simply pretraining the model
            save_model(model, f'checkpoint_{step_counter.get()}', args)

        writer.add_scalar('Train/loss', float(running_loss.get()), step_counter.get())
        writer.add_scalar('Train/ppl', math.exp(float(running_loss.get())), step_counter.get())


    logger.info(f"Epoch loss: {running_loss.get()}")
    logger.info(f"Epoch PPL: {ppl.get()}")


def run_evaluation(model, val_loader, tokenizer, writer, args):
    model.eval()

    running_nll = RunningLambdaMetric(CrossEntropyLoss(ignore_index=-100))
    ppl = MetricLambda(math.exp, running_nll)
    # Pick a random output from a random batch
    random_batch = random.randint(0, len(val_loader))
    # random_batch = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)

            # [Batch size, num_ands, seq_len]
            # [Batch size, num cands],
            # [batch size, num_cands, seq_len]
            # [batch_size]
            # [batch_size, num_cands, seq_len]
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            # if we dont send labels to model, it doesnt return losses
            lm_logits, mc_logits, *_ = model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )

            if i == random_batch:
                # Review outputs of random batch
                decode_sequence(input_ids, token_type_ids, model, tokenizer, args)

            # Compute loss metrics using this
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            running_nll.add(lm_logits_flat_shifted, lm_labels_flat_shifted)

    logger.info(f"NLL Loss: {running_nll.get()}")
    logger.info(f"Perlexity: {ppl.get()}")

def run_training(model, optimizer, scheduler, loaders, tokenizer, writer, args):
    print("run_training() ...")
    #train_loader, val_loader, train_sampler, valid_sampler = loaders

    train_loader, val_loader = loaders["train"], loaders["valid"]
    step_counter = GlobalStepCounter()

    if args.eval_before_start:
        run_evaluation(model, val_loader, tokenizer, writer, args)

    for epoch in range(args.n_epochs):
        #if args.distributed:
        #    # Ensures that the sampler splits the data properly
        #    train_sampler.set_epoch(epoch)
        #    valid_sampler.set_epoch(epoch)

        # Run training step
        run_train(model, optimizer, scheduler, train_loader, writer, step_counter, args)

        # Training step done, now evaluate
        run_evaluation(model, val_loader, tokenizer, writer, args)

    if args.n_epochs < 1:
        run_evaluation(model, val_loader, tokenizer, writer, args)




def main():
    pass


if __name__ == "__main__":
    main()
