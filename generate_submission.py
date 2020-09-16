import argparse
import logging
from collections import defaultdict
from itertools import chain
from pprint import pformat

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import OpenAIGPTTokenizer, GPT2Tokenizer, OpenAIGPTDoubleHeadsModel

from gpt2 import GPT2DoubleHeadsModel
from tc_dataset import TopicalChatsDataset, TopicalChatsKDDataset, TopicalChatsSWBDDataset, \
    TopicalChatsSentimentDataset, TopicalChatsSentGenerationDataset, TopicalChatsKDSentGenerationDataset
from train_util.decode import top_filtering
from utils import get_dataset, augmented_tc_dataset, get_dataset_sentence_generation
import torch.nn.functional as F
import os

"""
Code to generate the submissions for the DSTC9 Dialog Evaluation Track

This comprises of a decoding script which is executed on the valid_freq split of Topical Chats, since that's the 
official evaluation set for the track.

TODO: 
Create a decoding script that supports PPLM style output generation. 
This part is for my master's project - Rishi 
"""

logger = logging.getLogger(__file__)

ADDITIONAL_TOKENS = ["_nofact"]
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<end>", "<pad>", "<eot>"]  # added <end>, to represent the end of sent

ATTR_TO_SPECIAL_TOKEN = {
    'bos_token': '<bos>',
    'eos_token': '<eos>',
    'pad_token': '<pad>',
    'additional_special_tokens': ["<speaker1>", "<speaker2>"]
}

MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids", "das_to_return"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
OUTPUT_PATIENCE = 5

def decode_sequences(input_ids, token_type_ids, model, tokenizer, args):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    outputs = []
    for i in range(len(input_ids)):
        input_seq = tokenizer.decode(input_ids[i][0])
        prefix, suffix = input_seq.rsplit("<speaker", maxsplit=1)
        context = prefix + "<speaker" + suffix[:2]  # Hacky way to append the speaker tag

        current_output = []

        attempts = 0
        # Keep trying to generate output until a limited number of times
        expanded_tok_type_ids = token_type_ids[i][0].tolist()
        for j in range(args.max_length):  # Add trailing tokens
            expanded_tok_type_ids.append(expanded_tok_type_ids[-1])
        expanded_tok_type_ids = torch.tensor(expanded_tok_type_ids).to(args.device)
        patience = 10
        for j in range(args.max_length):
            prefix_input_seq = torch.tensor(tokenizer.encode(context) + current_output).unsqueeze(0)
            truncated_tok_type_ids = expanded_tok_type_ids[:prefix_input_seq.shape[-1]].unsqueeze(0)
            logits = model(prefix_input_seq.to(args.device), token_type_ids=truncated_tok_type_ids.to(args.device))

            if isinstance(logits, tuple) or len(logits.shape) == 4:  # for gpt2 and maybe others
                logits = logits[0]
            logits = logits[0, -1, :] / args.temperature
            logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
            if prev.item() in special_tokens_ids:
                patience = 10
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1 or patience == 0:
                        # Disabled this rather noisy warning
                        # logger.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)
                    patience -= 1
            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())

        output = tokenizer.decode(current_output)
        outputs.append(output.replace('\n', '') + '\n')
    return outputs

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" or name != "das_to_return" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def collate_sent_batch_elements(batch):

    batch_inputs = defaultdict(list)
    chained_batch = chain(*batch)

    for instance in chained_batch:
        for field, data in instance.items():
            batch_inputs[field].append(data)

    return batch


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

    padded_dataset = pad_dataset(batch_inputs, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]))

    tensorized_input = []
    # Verify input sent the same way:
    #
    # "input_ids": [Batch size, num_cands, seq_len]
    # "mc_token_ids": [Batch size, num cands],
    # "lm_labels": [batch size, num_cands, seq_len]
    # "mc_labels": [batch_size]
    # "token_type_ids": [batch_size, num_cands, seq_len]
    # "das_to_return": [batch_size]

    batch_size = tuple([len(batch_inputs[MODEL_INPUTS[0]])//args.num_candidates])
    for input_name in MODEL_INPUTS:
        if input_name != "das_to_return":
            tensor = torch.tensor(padded_dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, args.num_candidates) + tensor.shape[1:])
            else:
                tensor = torch.ones(size=batch_size, dtype=torch.long) * (args.num_candidates - 1)
        else:
            all_das = padded_dataset[input_name]
            tensor = [all_das[i] for i in range(0, len(all_das))]

        tensorized_input.append(tensor)
    return tensorized_input

def get_sentence_loader(args, tokenizer):
    # if args.dataset_configuration == "dstc9":
    #     topical_chat = get_dataset_sentence_generation(tokenizer, args.dataset_path, args.dataset_cache, args.training_configuration)
    # else:
    #     topical_chat = augmented_tc_dataset(tokenizer, args.dataset_path, args.dataset_cache,
    #                                         args.knowledge_index_path, args.training_configuration, args.knowledge_policy)
    #
    # splits = list(topical_chat.keys())
    # for split in splits:
    #     if split != args.split:
    #         del topical_chat[split]
    topical_chat = torch.load("valid_freq_cache")

    if args.training_configuration == "baseline":
        dataset = TopicalChatsSentGenerationDataset(topical_chat, tokenizer, SPECIAL_TOKENS, args)
    else:
        dataset = TopicalChatsKDSentGenerationDataset(topical_chat[args.split], tokenizer, SPECIAL_TOKENS, args)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.valid_batch_size,
                              collate_fn=lambda x: collate_sent_batch_elements(x),
                              shuffle=False)

    return loader, sampler, dataset

def get_loader(args, tokenizer):
    if args.dataset_configuration == "dstc9":
        topical_chat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, args.training_configuration, args.generation_configuration)
    else:
        topical_chat = augmented_tc_dataset(tokenizer, args.dataset_path, args.dataset_cache,
                                            args.knowledge_index_path, args.training_configuration, args.knowledge_policy)
    splits = list(topical_chat.keys())
    for split in splits:
        if split != args.split:
            del topical_chat[split]
        # Free up memory from unneeded splits
    if args.training_configuration == "baseline":
        dataset = TopicalChatsDataset(topical_chat[args.split], tokenizer, SPECIAL_TOKENS, args)
    elif args.training_configuration == "kd-pd-nrg-swbd":
        dataset = TopicalChatsSWBDDataset(topical_chat[args.split], tokenizer, SPECIAL_TOKENS, args,
                                          inference=args.heuristic_policy)
    elif args.training_configuration == "sentiment":
        dataset = TopicalChatsSentimentDataset(topical_chat[args.split], tokenizer, SPECIAL_TOKENS, args)
    else:
        dataset = TopicalChatsKDDataset(topical_chat[args.split], tokenizer, SPECIAL_TOKENS, args,
                                        inference=args.heuristic_policy)  # Enable heuristic dialog policy

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.valid_batch_size,
                              collate_fn=lambda x: collate_batch_elements(x, tokenizer, args),
                              shuffle=False)

    return loader, sampler, dataset


def generate_sentence_wise_output(model, tokenizer, dataset, example, args):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    output_so_far = []

    for i, plan in enumerate(example["plan"]):
        instance = dataset.prepare_generation_plan_for_sentence(example["history"], plan, tokenizer)

        input_ids = instance["input_ids"]
        token_type_ids = instance["token_type_ids"]
        expanded_tok_type_ids = token_type_ids

        for j in range(args.max_length + len(output_so_far)):  # Add trailing tokens
            expanded_tok_type_ids.append(expanded_tok_type_ids[-1])

        for j in range(args.max_length):
            inp = input_ids + output_so_far
            input_ids_t = torch.tensor(inp)
            token_type_ids_t = torch.tensor(expanded_tok_type_ids)[:input_ids_t.shape[-1]]
            logits = model(input_ids=input_ids_t.to(args.device), token_type_ids=token_type_ids_t.to(args.device))
            if isinstance(logits, tuple) or len(logits.shape) == 4:
                logits = logits[0].unsqueeze(0)

            logits = logits[0, -1, :] / args.temperature
            logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(logits, dim=-1)


            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
            if prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        # Disabled this rather noisy warning
                        # logger.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)
            if prev.item() in special_tokens_ids:
                break
            output_so_far.append(prev.item())
        if len(output_so_far) >= args.max_length:
            break
        # output_so_far.append(special_tokens_ids[1])
        output_so_far.append(special_tokens_ids[4])


    return tokenizer.decode(output_so_far, skip_special_tokens=True)



def generate_submissions_sent(args):
    tokenizer_class = GPT2Tokenizer

    tokenizer = tokenizer_class.from_pretrained(args.model_metadata_path)

    # data = torch.load(args.model_checkpoint + '/pytorch_model.bin', map_location=torch.device('cpu'))
    data = torch.load(args.model_checkpoint + '/pytorch_model.bin')

    model = data["mymodel"]
    model.to(args.device)

    outputs = []
    loader, sampler, dataset = get_sentence_loader(args, tokenizer)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):

            example = batch[0][0]

            output = generate_sentence_wise_output(model, tokenizer, dataset, example, args)
            if i % args.log_every_n == 0:
                logger.info(output)
            outputs.append(output.replace('\n', '') + '\n')

    save_outputs_and_plan([], args, outputs)


def generate_submissions(args):

    tokenizer_class = GPT2Tokenizer

    tokenizer = tokenizer_class.from_pretrained(args.model_metadata_path)

    outputs = []

    cache_file = {}
    completed_index = -1
    if os.path.isfile(args.submission_cache_path):
        logger.info("Load previous submission from cache at %s", args.submission_cache_path)
        cache_file = torch.load(args.submission_cache_path)
        outputs = cache_file["outputs"]
        completed_index = cache_file["i"]
    loader, sampler, dataset = get_loader(args, tokenizer)

    # This is not the proper way to load the model! This is a hack to be able to generate outputs from the
    # model I previously trained. This needs to be fixed in the original training script as well
    data = torch.load(args.model_checkpoint + '/pytorch_model.bin')
    model = data["mymodel"]
    model.to(args.device)

    all_das = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            if completed_index >= 0:
                completed_index -= 1
                continue
            # # Added this to generate missing outputs for KD-PD-NRG
            # if i + 1 not in [1267, 1831, 2475, 2498, 4220,
            #                  4683, 7252, 7504, 9236, 9476,
            #                  9612, 11114]:
            #     continue
            # batch = tuple(input_tensor.to(args.device) for input_tensor in batch)

            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, das_to_return = batch
            outputs += decode_sequences(input_ids, token_type_ids, model, tokenizer, args)
            all_das += das_to_return

            if i % args.log_every_n == 0:
                logger.info("Saving outputs to cache at %s", args.submission_cache_path)
                cache_file["outputs"] = outputs
                cache_file["i"] = i
                torch.save(cache_file, args.submission_cache_path)
                input_seq = tokenizer.decode(input_ids[0][0])
                prefix, suffix = input_seq.rsplit("<speaker", maxsplit=1)
                context = prefix + "<speaker" + suffix[:2]  # Hacky way to append the speaker tag
                logger.info(f"Context: {context}")
                logger.info(f"Sample output: {outputs[i*args.valid_batch_size]}")  # Log first sentence of that batch
    save_outputs_and_plan(all_das, args, outputs)


def save_outputs_and_plan(all_das, args, outputs):
    outputs_tags = []
    print(f"outputs len: {len(outputs)}")
    print(f"all_das len: {len(all_das)}")

    for output, plan in zip(outputs, all_das):
        outputs_tags.append(output.strip() + "".join(plan) + "\n")

    if outputs_tags:
        outputs_tags_file_path = args.output_file_path + "_tagged.txt"
        with open(outputs_tags_file_path, 'w') as output_file:
            output_file.writelines(outputs_tags)
    with open(args.output_file_path, 'w') as output_file:
        output_file.writelines(outputs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="processed_output",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument('--training_configuration', type=str, default="baseline",
                        help="Training configuration to run",
                        choices=["baseline", "kd-pd-nrg", "kd-pd-nrg-swbd", "sentiment"])
    parser.add_argument('--dataset_configuration', type=str, default="dstc9",
                        help="Configuration of dataset to load for training",
                        choices=["dstc9", "topical-chats"])
    parser.add_argument('--generation_configuration', type=str, default='sentence',
                        choices=['turn', 'sentence'])
    parser.add_argument('--heuristic_policy', action='store_true',
                        help="Enable heuristic dialog policy for generation (as opposed to using ground truth)")
    parser.add_argument('--knowledge_index_path', type=str, default="./tc_processed/tc_knowledge_index_bert_all.pkl",
                        help="Path to knowledge index file")
    parser.add_argument('--model_checkpoint', type=str, default="runs/bert_sentence_generation/",
                        help="Path, url or short name of the model")
    parser.add_argument("--split", type=str,
                        choices=['valid_freq', 'test_freq', 'valid_rare', 'test_rare'],
                        default='valid_freq',
                        help='Split of topical chats to generate outputs for')
    parser.add_argument('--model_metadata_path', type=str, default='./runs/bert_sentence_generation',
                        help='Path to the tokenizer and model configuration')
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument('--dataset_cache', type=str, default='./dataset_caches', help='Path or url of the dataset cache')
    parser.add_argument('--max_history', type=int, default=2, help='Number of previous exchanges to keep in history')
    parser.add_argument('--max_fact_length', type=int, default=200,
                        help='Number of fact tokens to include in the input')
    parser.add_argument('--valid_batch_size', type=int, default=1,
                        help='Batch size for generating outputs')
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    #                     help="Device (cuda or cpu)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument('--output_file_path', type=str, default='submissions/submissions.txt')
    parser.add_argument('--submission_cache_path', type=str, default='./submission_cache')
    parser.add_argument('--log_every_n', type=int, default=20,
                        help="Log a sample of outputs after every n iterations")
    # Decoding arguments
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")  # 95% of the reply lengths do not exceed 50
    parser.add_argument("--knowledge_policy", type=str, default="bert", choices=["tf_idf", "embeddings", "infersent", "bert"])

    args = parser.parse_args()
    args.distributed = (args.local_rank != -1)
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d",
                   args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    if args.generation_configuration == "turn":
        generate_submissions(args)
    else:
        generate_submissions_sent(args)
