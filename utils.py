import json
import logging
import os

import torch
from transformers import GPT2Tokenizer

CONFIG_NAME = 'config.json'

logger = logging.getLogger(__file__)

def load_data(dataset_path, split):
    path_prefix = os.path.join(dataset_path, split)

    # Splitting history into multiple sentences for ease of further processing
    src = [l.strip().split("_eos")[:-1] for l in open(path_prefix + '.src').readlines()]
    tgt = [l.strip().replace("_go", "").replace("_eos", "") for l in open(path_prefix + '.tgt').readlines()]
    fct = [l.strip() for l in open(path_prefix + '.fct').readlines()]
    return list(zip(src, tgt, fct))



def get_dataset(tokenizer, dataset_path, dataset_cache):
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Loading dataset from %s", dataset_path)

        splits = ['train', 'valid_freq', 'test_freq', 'test_rare', 'valid_rare']

        dataset = dict()

        for split in splits:
            data_items = load_data(dataset_path, split)

            def tokenize(obj):
                if isinstance(obj, str):
                    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
                if isinstance(obj, tuple):
                    return tuple(tokenize(o) for o in obj)
                return list(tokenize(o) for o in obj)

            logger.info("Tokenize the dataset")
            dataset[split] = tokenize(data_items)
        torch.save(dataset, dataset_cache)
    return dataset


def process_split(dataset_path, split, tokenizer):
    path_prefix = os.path.join(dataset_path, split)

    data = []
    with open(path_prefix + '_full_anno.json', 'r') as annotated_split_file:
        annotated_data = json.load(annotated_split_file)
        for conv_id, conv_data in annotated_data.items():
            context = []
            for turn in conv_data["content"]:
                current_turn_data = (tokenizer.encode(turn["message"]), turn["mezza_da"])
                data.append((context, current_turn_data))
                context = context + [current_turn_data]

    return data


def augmented_tc_dataset(tokenizer, dataset_path, dataset_cache):
    dataset_cache = dataset_cache + '_augmented_' + type(tokenizer).__name__

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Loading dataset from %s", dataset_path)

        splits = ['train', 'valid_freq', 'test_freq', 'test_rare', 'valid_rare']

        dataset = {}
        for split in splits:

            dataset[split] = process_split(dataset_path, split, tokenizer)
            logger.info("Processed split %s", split)
        torch.save(dataset, dataset_cache)

    return dataset

class GlobalStepCounter(object):
    def __init__(self):
        self.num_steps = 0

    def get(self):
        return self.num_steps

    def step(self):
        self.num_steps += 1

if __name__ == '__main__':
    pass
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    # dataset_path = 'processed_output'
    # dataset_cache = './dataset_cache'
    #
    #
    # get_dataset(tokenizer, dataset_path, dataset_cache)