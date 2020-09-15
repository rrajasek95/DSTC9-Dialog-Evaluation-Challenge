import argparse
import logging
import json
import utils
from itertools import chain
from pprint import pformat
from copy import deepcopy


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
    for conv_id in unique_keys:
        convo_history_segments = []
        cur_convo = split_data[conv_id]
        agent_knowledge, agent_mapping = utils.prepare_reading_set_for_conversation(conv_id, reading_set)

        turn_counter = 0
        for turn in cur_convo["content"]:
            turn_history = []
            da_index = 0
            for segment in turn["segments"]:
                sentence = segment["text"]
                current_segment_data = utils.prepare_sentence_knowledge_data(agent_mapping, conv_id, None, tokenizer,
                                                                       turn, sentence, ranker, da_index)

                if len(convo_history_segments) == turn_counter:
                    convo_history_segments.append(turn_history)
                else:
                    convo_history_segments[turn_counter] = turn_history
                convo_history_segments_copy = deepcopy(convo_history_segments)
                context = (convo_history_segments_copy, None, None)

                data.append((context, current_segment_data))
                turn_history.append(current_segment_data[0])
                da_index += 1
            turn_counter += 1
    torch.save(data, "valid_freq_cache")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./",
                        help="Path of the cached Topical Chat Dataset")
    parser.add_argument('--key_file', type=str, default="processed_output/valid_freq.keys",
                        help="path to key file of dstc9")
    parser.add_argument("--knowledge_policy", type=str, default="bert_sentence", choices=["tf_idf", "embeddings", "infersent", "bert"])
    parser.add_argument('--knowledge_index_path', type=str, default="./tc_processed/tc_knowledge_index_bert_all.pkl",
                        help="Path to knowledge index file")
    parser.add_argument('--model_metadata_path', type=str, default='./runs/bert_sentence_generation',
                        help='Path to the tokenizer and model configuration')
    args = parser.parse_args()
    create_topical_chat_dict(args)