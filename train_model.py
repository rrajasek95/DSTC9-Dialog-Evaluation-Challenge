
"""
A significant part of the code is adapted from the
HuggingFace Transformers library repository and the TransferTransfo setup.

Link: https://github.com/huggingface/transformers/tree/master/examples/text-generation
Link2: https://github.com/huggingface/transfer-learning-conv-ai
This code sets up a base knowledge-conditioned neural generative model.

I am looking to investigate two variants for pretraining:
1. Transfer learning from baseline GPT2 to Topical Chats corpus
2. Transfer learning from DialoGPT to Topical Chats corpus

"""
import argparse
import logging
import math
import os
import random

from collections import defaultdict
from itertools import chain
from pprint import pformat

from torch.nn import NLLLoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, GPT2DoubleHeadsModel, GPT2Tokenizer, OpenAIGPTTokenizer, OpenAIGPTDoubleHeadsModel

from tc_dataset import TopicalChatsDataset
from metrics import RunningMetric, RunningLambdaMetric, MetricLambda
from scheduler import PiecewiseLinearLR
from utils import get_dataset

logger = logging.getLogger(__file__)

# The _nofact token needs to be added
ADDITIONAL_TOKENS = ["_nofact"]
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]

ATTR_TO_SPECIAL_TOKEN = {
    'bos_token': '<bos>',
    'eos_token': '<eos>',
    'pad_token': '<pad>',
    'additional_special_tokens': ["<speaker1>", "<speaker2>"]
}

MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


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


def add_special_tokens_(tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    num_added_norm_tokens = tokenizer.add_tokens(ADDITIONAL_TOKENS)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there

    return num_added_tokens + num_added_norm_tokens


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def collate_batch_elements(batch, tokenizer, args):
    """
    Topical chats is a ridiculously large dataset (2GB+ including facts/reading set).
    Maintaining an entire tensor dataset in memory is a terrible idea
    since *every* input is padded to the size of the largest element.
    The training dataset has 179k instances, so imagine padding all
    of those to max_length (RIP RAM!)

    Another point to note is that the actual number of instances per batch in this
    implementation is num_candidates*batch_size. I haven't thought about how to
    optimize this but I guess it'll take a bit more effort
    - Rishi

    """
    batch_inputs = defaultdict(list)
    chained_batch = chain(*batch)

    for instance in chained_batch:
        for field, data in instance.items():
            batch_inputs[field].append(data)

    padded_dataset = pad_dataset(batch_inputs, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))

    tensorized_input = []
    # Verify input sent the same way:
    #
    # "input_ids": [Batch size, num_cands, seq_len]
    # "mc_token_ids": [Batch size, num cands],
    # "lm_labels": [batch size, num_cands, seq_len]
    # "mc_labels": [batch_size]
    # "token_type_ids": [batch_size, num_cands, seq_len]

    batch_size = tuple([len(batch_inputs[MODEL_INPUTS[0]])//args.num_candidates])
    for input_name in MODEL_INPUTS:
        tensor = torch.tensor(padded_dataset[input_name])

        if input_name != "mc_labels":
            tensor = tensor.view((-1, args.num_candidates) + tensor.shape[1:])
        else:
            tensor = torch.ones(size=batch_size, dtype=torch.long) * (args.num_candidates - 1)
        tensorized_input.append(tensor)
    return tensorized_input

def build_input_from_segments(history, response, fact, tokenizer, lm_labels=False):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids((SPECIAL_TOKENS[:-1]))

    """
    Input construction (may change):
    <bos> FACT <speaker1/2> UTT1 <speaker1/2> ... <speaker2> RESPONSE
    Considerations for design:
    1. Topical chat examples are created by adding a response every turn
    2. Last turn is always speaker2
    
    To my knowledge, the position of the fact in input is mostly immaterial due to
    the self-attention mechanism (since all tokens are equidistant). The positional
    embeddings affect only the contextual representation
      - Rishi
    """
    sequence = [[bos] + fact] + history + [response + [eos]]

    sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1

    # I have no idea what this part refers to, Cargo Culting for now
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance


def sample_candidates(dataset, current_conversation_index, args):
    # Lets just hope that the number of cases where the true responses gets included in the
    # candidates is vanishingly small
    candidates = [response for (_, response, _) in random.sample(dataset, args.num_candidates - 1)]

    return candidates


def get_data_loaders(args, tokenizer):
    topicalchat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Building TC inputs and labels")
    datasets = {"train": defaultdict(list), "valid_freq": defaultdict(list)}

    for dataset_name, dataset in topicalchat.items():

        for i, dialog in enumerate(dataset):
            (history, response, fact) = dialog

            # Truncate history turns to reduce memory requirement
            if len(history) > (2*args.max_history+1):
                history = history[-(2*args.max_history+1):]

            candidates = sample_candidates(dataset, i, args)

            candidates.append(response)

            for j, candidate in enumerate(candidates):
                lm_labels = bool(j == args.num_candidates - 1)
                instance = build_input_from_segments(history, candidate, fact, tokenizer, lm_labels)

                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)
            datasets[dataset_name]["mc_labels"].append(args.num_candidates - 1)
            datasets[dataset_name]["n_candidates"] = args.num_candidates
            # The original TransferTransfo model permutes personae
            # we have only one fact segment to condition on, so it's wholly unnecessary

    del topicalchat # Out of memory issue, so have to free memory

    logger.info("Pad inputs and convert to Tensor")

    tensor_datasets = {"train": [], "valid_freq": []}

    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))

        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])

            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid_freq"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))

    return train_loader, valid_loader, train_sampler, valid_sampler

def get_data_loaders_optimized(args, tokenizer):

    topical_chat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    train_dataset, valid_dataset = TopicalChatsDataset(topical_chat["train"], tokenizer, SPECIAL_TOKENS, args), TopicalChatsDataset(topical_chat["valid_freq"], tokenizer, SPECIAL_TOKENS, args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                              collate_fn=lambda x: collate_batch_elements(x, tokenizer, args),
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size,
                              collate_fn=lambda x: collate_batch_elements(x, tokenizer, args),
                              shuffle=False)

    # logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset[0][0].shape))
    # logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset[0][0].shape))

    return train_loader, valid_loader, train_sampler, valid_sampler

def run_train(model, optimizer, scheduler, train_loader, writer, args):
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
        running_loss.add(loss.mean())

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

        writer.add_scalar('Train/loss', loss, i)
        writer.add_scalar('Train/ppl', math.exp(loss), i)

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
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
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
    train_loader, val_loader, train_sampler, valid_sampler = loaders

    if args.eval_before_start:
        run_evaluation(model, val_loader, tokenizer, writer, args)

    for epoch in range(args.n_epochs):
        if args.distributed:
            # Ensures that the sampler splits the data properly
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

        # Run training step
        run_train(model, optimizer, scheduler, train_loader, writer, args)

        # Training step done, now evaluate
        run_evaluation(model, val_loader, tokenizer, writer, args)

    if args.n_epochs < 1:
        run_evaluation(model, val_loader, tokenizer, writer, args)


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="processed_output",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2-medium",
                        help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    # parser.add_argument("--personality_permutations", type=int, default=1,
    #                     help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument('--log_dir', type=str, default="runs/",
                        help="Output log directory for summary")
    parser.add_argument('--experiment_name', type=str, default="topical_chats_gpt2",
                        help="Name of experiment for logging purposes")
    # Decoding arguments
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d",
                   args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer  # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)


    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = add_special_tokens_(tokenizer)

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare datasets")
    loaders = get_data_loaders_optimized(args, tokenizer)
    train_loader, _, _, _ = loaders

    # Load the model after the tokenizer. We hit an OOM error if we try to pre-load the model
    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)

    # Add special tokens if they are not already added
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    scheduler = PiecewiseLinearLR(optimizer, [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])

    writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, args.experiment_name))

    run_training(model, optimizer, scheduler, loaders, tokenizer, writer, args)

if __name__ == '__main__':
    train()