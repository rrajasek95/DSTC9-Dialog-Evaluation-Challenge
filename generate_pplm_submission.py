"""
This script tries to adapt the PPLM framework to be used with a Dialog Act tagger. In this case, I try to use
the SWBD DA tagger.

The model construction is very unusual because of the following:
1. We have a conditional language model p(x | c, k) conditioned on dialog history c and knowledge k
2. We have an unconditional discriminator p(DA | x) that predicts a dialog act on x

Goal is to construct a model p(x | DA, c, k)

p(x | DA, c , k) = p(DA | x, c, k) * p (x | c, k) / p(DA | c, k)

The model is constructed by making some **very** strong simplifying assumptions:
1. p(DA | x, c, k) = p (DA | x)
2. p(DA | c, k) = p(DA)

p(x | DA, c, k) = p(DA | x ) * p (x | c, k) / P(DA)
or, more simply,
p(x | DA, c, k) ∝ p(DA | x ) * p (x | c, k)

Namely, that the disciminator is unconditional on c, k. I don't know if this is a valid assumption, but I am
operating on a hunch that it is. This allows me to use an existing dialog act classifier
that is not trained on k, c information. This is a verification of that assumption.

Our goal is to sample from p(x | DA, c, k) to produce outputs of a specific DA. This creates a very flexible
sentence planning model.
"""
import argparse
import logging
import os
import sys
from collections import defaultdict
from itertools import chain
from operator import add

import dill
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import GPT2Tokenizer

from DA_Classifier import train
from gpt2 import GPT2DoubleHeadsModel, GPT2LMHeadModel
from tc_dataset import TopicalChatsDataset
from utils import get_dataset

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)

DISCRIMINATOR_MODELS_PARAMS = {
    "description": """
    The discriminator is a Dialog Act Tagger (BiLSTM-CRF)     
    """,
    "path": os.path.join('DA_Classifier', 'cached_models', 'm3_acc79.84_loss0.58_e4.pt')
}

MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

SMALL_CONST = 1e-15
BIG_CONST = 1e10

def load_discriminator(args):

    logger.info("Loading the Discriminative BiLSTM-CRF model")
    checkpoint = torch.load(DISCRIMINATOR_MODELS_PARAMS["path"], pickle_module=dill, map_location=args.device)
    fields = checkpoint["fields"]
    model_opt = checkpoint["opt"]

    embedding_size = model_opt.word_vec_size
    embeddings = train.build_embeddings(model_opt, fields["conversation"], embedding_size)

    model = train.build_model(model_opt, fields, embeddings)
    model.load_state_dict(checkpoint["model"])

    model.eval()
    logger.info(model)
    logger.info(fields)
    return model, fields

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

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)

def get_loader(args, tokenizer):
    topical_chat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    splits = list(topical_chat.keys())
    for split in splits:
        if split != args.split:
            del topical_chat[split]
        # Free up memory from unneeded splits
    dataset = TopicalChatsDataset(topical_chat[args.split], tokenizer, SPECIAL_TOKENS, args)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.valid_batch_size,
                              collate_fn=lambda x: collate_batch_elements(x, tokenizer, args),
                              shuffle=False)

    return loader, sampler


def perturb_past(past, model, last, unpert_past, unpert_logits, accumulated_hidden, grad_norms, stepsize, classifier,
                 classifier_fields, class_label, num_iterations, horizon_length, window_length, decay, gamma, kl_scale,
                 device):
    grad_accumulator = [(np.zeros(p.shape).astype("float32")) for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(0.0, 1.0 + SMALL_CONST, 1.0 / (window_length))[1:]
    else:
        decay_mask = 1.0

    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple([window_length]) + tuple(past[0].shape[-1:])

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2]) + tuple([curr_length - window_length]) + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)


    loss_per_iter = []
    new_accumulated_hidden = None

    for i in range(num_iterations):
        print("Iteration", i + 1)
        curr_perturbation = [torch.from_numpy(p_).requires_grad_(True).to(device=device) for p_ in grad_accumulator]
        # make sure p_.grad is not None
        for p_ in curr_perturbation:
            p_.retain_grad()

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0

        loss_list = []

        # TODO:Compute loss for the current decoded sequence so far

        #

        kl_loss = 0.0

        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = unpert_probs + SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(device).detach()
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
            corrected_probs = probs + correction.detach()

            kl_loss = kl_scale *((corrected_probs * (corrected_probs/unpert_probs).log()).sum())
            print(" kl_loss ", kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        print(" PPLM loss", (loss - kl_loss).data.cpu().numpy())

        loss.backward()

        # TODO: Calculate gradient norms

        # Normalize gradients
        grad = [
            -stepsize * (p_.grad * window_mask / grad_norms[index]**gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        grad_accumulator = list(map(add, grad, grad_accumulator))

        # Reset gradients
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []

        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

        grad_accumulator = [torch.from_numpy(p_).requires_grad_(True).to(device=device) for p_ in grad_accumulator]
        pert_past = list(map(add, past, grad_accumulator))

        return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter

def generate_text_pplm(model, tokenizer, context,
                       device, perturb, classifier, classifier_fields, class_label, length,
                       stepsize, temperature, top_k, sample, num_iterations, grad_length, horizon_length, window_length,
                       decay, gamma, gm_scale, kl_scale, repetition_penalty):
    past = None
    output_so_far = None

    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)

        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    for i in trange(length, ascii=True):

        # Get past/probs for current output, except for last word
        # GPT takes 2 inputs: past + current token

        # Run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last=output_so_far[:, -1, :]

            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :, -1])

            unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
            unpert_last_hidden = unpert_all_hidden[-1]

            # Check if we're above grad max length
            if i >= grad_length:
                current_stepsize = stepsize * 0
            else:
                current_stepsize = stepsize

            # Modify the past if necessary
            if not perturb or num_iterations == 0:
                pert_past = past
            else:
                accumulated_hidden = unpert_last_hidden[:, :-1, :]
                accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

                if past is not None:
                    pert_past, _, grad_norms, loss_this_iter = perturb_past(
                        past,
                        model,
                        last,
                        unpert_past=unpert_past,
                        unpert_logits=unpert_logits,
                        accumulated_hidden=accumulated_hidden,
                        grad_norms=grad_norms,
                        stepsize=current_stepsize,
                        classifier=classifier,
                        classifier_fields=classifier_fields,
                        class_label=class_label,
                        num_iterations=num_iterations,
                        horizon_length=horizon_length,
                        window_length=window_length,
                        decay=decay,
                        gamma=gamma,
                        kl_scale=kl_scale,
                        device=device
                    )

                    loss_in_time.append(loss_this_iter)
                else:
                    pert_past = past

            pert_logits, past, pert_all_hidden = model(last, past=pert_past)
            pert_logits = pert_logits[:, -1, :] / temperature

            for token_idx in set(output_so_far[0].tolist()):
                if pert_logits[0, token_idx] < 0:
                    pert_logits[0, token_idx] *= repetition_penalty
                else:
                    pert_logits[0, token_idx] /= repetition_penalty

            pert_probs = F.softmax(pert_logits, dim=-1)

            # TODO: Fill this in
            if classifier is not None:
                pass
                # Compute loss of predicted dialog act
                # this is done by converting the decoded output into a field
            else:
                unpert_discrim_loss = 0

            if perturb:

                unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=1)

                pert_probs = (pert_probs ** gm_scale) * (unpert_probs ** (1 - gm_scale))
                pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)

                # Rescale
                if torch.sum(pert_probs) <= 1:
                    pert_probs = pert_probs / torch.sum(pert_probs)
            else:
                pert_logits = top_k_filter(pert_logits, k=top_k)
                pert_probs = F.softmax(pert_logits, dim=-1)

            if sample:
                last = torch.multinomial(pert_probs, num_samples=1)

            else:
                _, last = torch.topk(pert_probs, k=1, dim=-1)

            # update context/output_so_far appending the new token
            output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)

            print(tokenizer.decode(output_so_far.tolist()[0]))

    return output_so_far, unpert_discrim_loss, loss_in_time

def full_text_generation(model, tokenizer, context, device, num_samples, discrim, discrim_fields, class_label, length,
                         stepsize, temperature, top_k, sample, num_iterations, grad_length, horizon_length,
                         window_length, decay, gamma, gm_scale, kl_scale, repetition_penalty):

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        repetition_penalty=repetition_penalty
    )

    if device == "cuda":
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            classifier=discrim,
            classifier_fields=discrim_fields,
            class_label=class_label,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            repetition_penalty=repetition_penalty
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if discrim is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(losses_in_time)

    if device == "cuda":
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def run_pplm_da(args):

    # Load discriminator model
    discriminator, fields = load_discriminator(args)

    # Load GPT2 model
    model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)

    # Freeze GPT2 weights
    for param in model.parameters():
        param.requires_grad = False

    loader, sampler = get_loader(args, tokenizer)

    for i, batch in tqdm(enumerate(loader)):
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch

        for j in range(len(input_ids)):
            input_seq = tokenizer.decode(input_ids[j][0])
            prefix, suffix = input_seq.rsplit("<speaker", maxsplit=1)
            context = prefix + "<speaker" + suffix[:2]  # Hacky way to append the speaker tag

            tokenized_cond_text = tokenizer.encode(context)

            logger.info("= Prefix of sentence = ")
            logger.info(f"{context}\n")

            unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
                model=model,
                tokenizer=tokenizer,
                context=tokenized_cond_text,
                device=args.device,
                num_samples=1,
                discrim=discriminator,
                discrim_fields=fields,
                class_label=args.class_label,
                length=args.length,
                stepsize=args.stepsize,
                temperature=args.temperature,
                top_k=args.top_k,
                sample=args.sample,
                num_iterations=args.num_iterations,
                grad_length=args.grad_length,
                horizon_length= args.horizon_length,
                window_length = args.window_length,
                decay=args.decay,
                gamma=args.gamma,
                gm_scale = args.gm_scale,
                kl_scale=args.kl_scale,
                repetition_penalty=args.repetition_penalty
            )

            logger.info("= Unperturbed generated text")
            logger.info(unpert_gen_tok_text)

            for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
                try:
                    logger.info(f"= Perturbed generated text {i + 1} =")
                    logger.info(pert_gen_tok_text)
                except Exception as exc:
                    logger.info(f"Random error {exc}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # LM args
    parser.add_argument("--dataset_path", type=str, default="processed_output",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument('--dataset_cache', type=str, default='./dataset_cache', help='Path or url of the dataset cache')

    parser.add_argument('--model_checkpoint',
                        default="gpt2-medium",
                        help="Pretrained model name or path to model checkpoint")
    parser.add_argument('--max_history', type=int, default=2, help='Number of previous exchanges to keep in history')
    parser.add_argument('--max_fact_length', type=int, default=200,
                        help='Number of fact tokens to include in the input')
    parser.add_argument('--valid_batch_size', type=int, default=4,
                        help='Batch size for generating outputs')
    parser.add_argument('--length', type=int,
                        default=50,
                        help='Max length for the decoded utterance')
    parser.add_argument("--num_candidates", type=int, default=1, help="Number of candidates for training")
    parser.add_argument('--split', type=str, default="valid_freq", help="Split to generate outputs for")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    # Discriminator args
    parser.add_argument('--class_label', type=int,
                        default=1,
                        help='Label of class to condition on')

    # PPLM args
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; 0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length", type=int, default=1, help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true", help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="Penalize repetition. More than 1.0 -> less repetition",
    )

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else "cpu",
                        choices=['cpu', 'cuda'],
                        help='Device to run on. Defaults to CUDA if available')

    args = parser.parse_args()
    args.distributed = (args.local_rank != -1)
    run_pplm_da(args)
