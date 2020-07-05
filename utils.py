import itertools
import json
import logging
import os
import pickle

import torch
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
from transformers import GPT2Tokenizer

CONFIG_NAME = 'config.json'

logger = logging.getLogger(__file__)

def load_data(dataset_path, split, training_configuration):
    path_prefix = os.path.join(dataset_path, split)

    # Splitting history into multiple sentences for ease of further processing
    src = [l.strip().split("_eos")[:-1] for l in open(path_prefix + '.src').readlines()]
    tgt = [l.strip().replace("_go", "").replace("_eos", "") for l in open(path_prefix + '.tgt').readlines()]
    fct = [l.strip() for l in open(path_prefix + '.fct').readlines()]
    if training_configuration != "baseline" and split == "train":
        history_da = [l.strip().split("_eos")[:-1] for l in open(path_prefix + ".src.da").readlines()]
        history_knowledge = itertools.repeat(itertools.repeat(""))
        # history_knowledge = [l.strip().split("_eos")[:-1] for l in open(path_prefix + ".src.fct")]
        resp_da = [l.strip() for l in open(path_prefix + '.tgt.da').readlines()]
    else:
        history_da = itertools.repeat(itertools.repeat(None))
        history_knowledge = itertools.repeat(itertools.repeat(None))
        resp_da = itertools.repeat(None)

    context = [zip(s, h, k) for (s, h, k) in zip(src, history_da, history_knowledge)]
    return list(zip(context, zip(tgt, resp_da, fct)))



def get_dataset(tokenizer, dataset_path, dataset_cache, training_configuration):
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Loading dataset from %s", dataset_path)

        splits = ['train', 'valid_freq', 'test_freq', 'test_rare', 'valid_rare']

        dataset = dict()

        for split in splits:
            data_items = load_data(dataset_path, split, training_configuration)

            def tokenize(obj):
                if obj is None:
                    return None
                if isinstance(obj, str):
                    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
                if isinstance(obj, tuple):
                    return tuple(tokenize(o) for o in obj)
                return list(tokenize(o) for o in obj)

            logger.info("Tokenize the dataset")
            dataset[split] = tokenize(data_items)
        torch.save(dataset, dataset_cache)
    return dataset


def process_split(dataset_path, split, tokenizer, index):
    vec, knowledge_sentences, tfidf_vecs, dialog_act = index
    path_prefix = os.path.join(dataset_path, split)

    data = []
    with open(path_prefix + '_full_anno.json', 'r') as annotated_split_file:
        annotated_data = json.load(annotated_split_file)
        for conv_id, conv_data in tqdm(annotated_data.items()):
            context = []
            for turn in conv_data["content"]:
                # This is a highly approximate heuristic.
                # Both Gopalakrishnan et al. 2019 and Hedayatnia et al. 2020
                # acknowledge they don't have anything better for this issue
                response = turn["message"]

                for segment in turn["segments"]:
                    sentence = segment["text"]
                    sentence_vec = vec.transform([sentence])
                    similarities = linear_kernel(tfidf_vecs, sentence_vec).flatten()
                    closest_knowledge_index = similarities.argsort()[-1]

                    if similarities[closest_knowledge_index] > 0.3:
                        knowledge_sentence = knowledge_sentences[closest_knowledge_index]
                        break
                else:
                    response_vec = vec.transform([response])
                    similarities = linear_kernel(tfidf_vecs, response_vec).flatten()
                    closest_knowledge_index = similarities.argsort()[-1]

                    knowledge_sentence = knowledge_sentences[closest_knowledge_index] \
                        if similarities[closest_knowledge_index] > 0.3 else ""

                current_turn_data = (tokenizer.encode(response), turn[dialog_act], tokenizer.encode(knowledge_sentence))
                data.append((context, current_turn_data))
                context = context + [current_turn_data]

    return data


def augmented_tc_dataset(tokenizer, dataset_path, dataset_cache, knowledge_index_path, dialog_act):
    dataset_cache = dataset_cache + '_augmented_' + type(tokenizer).__name__
    with open(knowledge_index_path, 'rb') as knowledge_index_file:
        index_data = pickle.load(knowledge_index_file)
    vec = index_data["tfidf_vec"]
    knowledge_sentences = index_data["knowledge_list"]
    tfidf_vecs = vec.transform(knowledge_sentences)
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Loading dataset from %s", dataset_path)

        splits = ['train', 'valid_freq', 'test_freq', 'test_rare', 'valid_rare']

        dataset = {}
        for split in splits:

            dataset[split] = process_split(dataset_path, split, tokenizer, (vec, knowledge_sentences, tfidf_vecs, dialog_act))
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

def generate_references_for_split(dataset_path, dataset_cache, split, output_path):
    path_prefix = os.path.join(dataset_path, split)
    responses = []
    with open(path_prefix + '_full_anno.json', 'r') as annotated_split_file:
        annotated_data = json.load(annotated_split_file)
        for conv_id, conv_data in tqdm(annotated_data.items()):
            for turn in conv_data["content"]:
                text = turn["message"]
                if text:
                    responses.append(text.strip() + "\n")

    with open(output_path, 'w') as references_file:
        references_file.writelines(responses[:-1])


if __name__ == '__main__':
    pass
    # generate_references_for_split('tc_processed', None, 'valid_freq', 'tc_processed/valid_freq.tgt')
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    # dataset_path = 'processed_output'
    # dataset_cache = './dataset_cache'
    #
    #
    # get_dataset(tokenizer, dataset_path, dataset_cache)