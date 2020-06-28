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
from tc_dataset import TopicalChatsDataset
from utils import get_dataset
import torch.nn.functional as F

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

def decode_sequences(input_ids, token_type_ids, model, tokenizer, args):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    outputs = []
    for i in range(len(input_ids)):
        input_seq = tokenizer.decode(input_ids[i][0])
        prefix, suffix = input_seq.rsplit("<speaker", maxsplit=1)
        context = prefix + "<speaker" + suffix[:2]  # Hacky way to append the speaker tag

        current_output = []

        expanded_tok_type_ids = token_type_ids[i][0].tolist()
        for j in range(args.max_length): # Add trailing tokens
            expanded_tok_type_ids.append(expanded_tok_type_ids[-1])
        expanded_tok_type_ids = torch.tensor(expanded_tok_type_ids).to(args.device)
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
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        # Disabled this rather noisy warning
                        # logger.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)
            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())

        output = tokenizer.decode(current_output)
        outputs.append(output + '\n')
    return outputs

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

def generate_submissions(args):

    tokenizer_class = GPT2Tokenizer

    tokenizer = tokenizer_class.from_pretrained(args.model_metadata_path)

    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel

    # This is not the proper way to load the model! This is a hack to be able to generate outputs from the
    # model I previously trained. This needs to be fixed in the original training script as well
    data = torch.load(args.model_checkpoint + '/pytorch_model.bin')
    model = data["mymodel"]
    print(model)

    # model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)

    loader, sampler = get_loader(args, tokenizer)

    outputs = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            # batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch

            outputs += decode_sequences(input_ids, token_type_ids, model, tokenizer, args)

            if i % args.log_every_n == 0:
                logger.info(f"Sample output: {outputs[i*args.valid_batch_size]}")  # Log first sentence of that batch

    with open(args.output_file_path, 'w') as output_file:
        output_file.writelines(outputs)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="processed_output",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument('--model_checkpoint', type=str, default="runs/topical_chats_gpt2/",
                        help="Path, url or short name of the model")
    parser.add_argument("--split", type=str,
                        choices=['valid_freq', 'test_freq', 'valid_rare', 'test_rare'],
                        default='valid_freq',
                        help='Split of topical chats to generate outputs for')
    parser.add_argument('--model_metadata_path', type=str, default='runs/topical_chats_gpt2',
                        help='Path to the tokenizer and model configuration')
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument('--dataset_cache', type=str, default='./dataset_cache', help='Path or url of the dataset cache')
    parser.add_argument('--max_history', type=int, default=2, help='Number of previous exchanges to keep in history')
    parser.add_argument('--max_fact_length', type=int, default=200,
                        help='Number of fact tokens to include in the input')
    parser.add_argument('--valid_batch_size', type=int, default=4,
                        help='Batch size for generating outputs')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument('--output_file_path', type=str, default='submissions/submissions.txt')
    parser.add_argument('--log_every_n', type=int, default=20,
                        help="Log a sample of outputs after every n iterations")
    # Decoding arguments
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")  # 95% of the reply lengths do not exceed 50

    args = parser.parse_args()
    args.distributed = (args.local_rank != -1)
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d",
                   args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    generate_submissions(args)