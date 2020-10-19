import argparse
import logging
import json
import utils
from itertools import chain
from pprint import pformat
from copy import deepcopy
import string

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

logger = logging.getLogger(__file__)


def create_topical_chat_dict(args):
    # logger.info("Load tokenized dataset from cache at %s", args.dataset_path)
    # dataset = torch.load(args.dataset_path)

    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_metadata_path)

    with open(args.key_file) as f:
        keys = f.readlines()
    keys = [x.strip() for x in keys]

    unique_keys = []
    for key in keys:
        if key not in unique_keys:
            unique_keys.append(key)

    data_dir = os.path.join(
        args.dataset_path,
        'tc_processed'
    )

    read_dir = os.path.join(
        args.dataset_path,
        'alexa-prize-topical-chat-dataset',
        'reading_sets',
        'post-build'
    )

    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    vec = utils.load_knowledge_vecs(args.knowledge_policy, args.knowledge_index_path)
    ranker = utils.get_ranker_retriever(args.knowledge_policy, vec)
    split_data = {}
    for split in splits:
        with open(os.path.join(data_dir, split + '_anno.json'), 'r') as data_file:
            split_data.update(json.load(data_file))

    reading_set = {}
    for split in splits:
        with open(os.path.join(read_dir, split + '.json'), 'r') as data_file:
            reading_set.update(json.load(data_file))
    data = []
    cur_convo_id = keys[0]
    turn_count = 1
    convo_history = []

    cur_convo = split_data[cur_convo_id]
    turn = cur_convo["content"][0]
    sentences_first = []
    for segment in turn["segments"]:
        sentence = segment["text"]
        sentences_first.append(sentence)
    convo_history.append((sentences_first, None, None))

    if args.create_knowledge:
        create_knowledge_file(1, keys, cur_convo_id, split_data, reading_set, tokenizer, ranker)
    else:
        for conv_id in keys:
            if cur_convo_id != conv_id:
                turn_count = 1
                cur_convo_id = conv_id
                convo_history = []
                cur_convo = split_data[cur_convo_id]
                turn = cur_convo["content"][0]
                sentences_first = []
                for segment in turn["segments"]:
                    sentence = segment["text"]
                    sentences_first.append(sentence)
                convo_history.append((sentences_first, None, None))
            cur_convo = split_data[conv_id]
            agent_knowledge, agent_mapping = utils.prepare_reading_set_for_conversation(conv_id, reading_set)
            turn = cur_convo["content"][turn_count]
            turn_knowledge = []
            sentences = []
            for segment in turn["segments"]:
                sentence = segment["text"]
                sentences.append(sentence)
                segment_knowledge = prepare_sentence_knowledge_data(agent_mapping, conv_id, None, tokenizer,
                                                                       turn, sentence, ranker)
                turn_knowledge.append(segment_knowledge)
            cur_turn_data = sentences, None, turn_knowledge
            history_copy = deepcopy(convo_history)
            data.append((history_copy, cur_turn_data))
            history = sentences, None, None
            convo_history.append(history)
            turn_count += 1
        torch.save(data, "test_freq_cache")

def clean(s):
    return ''.join([c if c not in string.punctuation else ' ' for c in s.lower()])


def prepare_sentence_knowledge_data(agent_mapping, conv_id, dialog_act, tokenizer, turn, sentence, ranker):
    knowledge_sentence = ranker.get_top_fact(clean(sentence), conv_id, threshold=True)
    original_knowledge_sentence = agent_mapping[turn["agent"]].get(knowledge_sentence, "")
    return original_knowledge_sentence


def create_knowledge_file(turn_count, keys, cur_convo_id, split_data, reading_set, tokenizer, ranker):
    total_knowledge = []
    for conv_id in keys:
        if cur_convo_id != conv_id:
            turn_count = 1
            cur_convo_id = conv_id
        cur_convo = split_data[conv_id]
        agent_knowledge, agent_mapping = utils.prepare_reading_set_for_conversation(conv_id, reading_set)
        turn = cur_convo["content"][turn_count]
        turn_knowledge = []
        for segment in turn["segments"]:
            sentence = segment["text"]
            segment_knowledge = prepare_sentence_knowledge_data(agent_mapping, conv_id, None, tokenizer,
                                                                   turn, sentence, ranker)
            turn_knowledge.append(segment_knowledge)
        total_knowledge.append(turn_knowledge)
        turn_count += 1
    with open('valid_freq_facts_bert.txt', 'w') as f:
        for item in total_knowledge:
            fact_string = ""
            for s in item:
                if s == "":
                    fact_string += "no_fact; "
                else:
                    fact_string += s + "; "
            f.write("%s\n" % fact_string.replace("\n", "").strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./",
                        help="Path of the cached Topical Chat Dataset")
    parser.add_argument('--key_file', type=str, default="processed_output/test_freq.keys",
                        help="path to key file of dstc9")
    parser.add_argument("--knowledge_policy", type=str, default="bert_sentence", choices=["tf_idf", "embeddings", "infersent", "bert", "bert_sentence"])
    parser.add_argument('--knowledge_index_path', type=str, default="./tc_processed/tc_knowledge_index_bert_all.pkl",
                        help="Path to knowledge index file")
    parser.add_argument('--model_metadata_path', type=str, default='./runs/bert_sentence_generation',
                        help='Path to the tokenizer and model configuration')
    parser.add_argument('--create_knowledge', type=bool, default=False,
                        help='generate just the knowledge')
    args = parser.parse_args()
    # create_topical_chat_dict(args)

    dataset = torch.load("test_freq_cache")

    dataset