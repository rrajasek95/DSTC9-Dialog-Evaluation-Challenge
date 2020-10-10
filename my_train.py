"""
 Created by diesel
 10/8/20

A significant part of the code is adapted from the
HuggingFace Transformers library repository and the TransferTransfo setup.

Link: https://github.com/huggingface/transformers/tree/master/examples/text-generation
Link2: https://github.com/huggingface/transfer-learning-conv-ai
This code sets up a base knowledge-conditioned neural generative model.

I am looking to investigate two variants for pretraining:
1. Transfer learning from baseline GPT2 to Topical Chats corpus
2. Transfer learning from DialoGPT to Topical Chats corpus


"""


import logging
import os


import argparse
import itertools

from collections import defaultdict
from pprint import pformat
from itertools import chain

import pickle
import json




from nltk import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm


from transformers import AdamW, GPT2Tokenizer

import trainer

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


from tc_dataset import TopicalChatsDataset, TopicalChatsKDDataset, TopicalChatsSentimentDataset, \
    TopicalChatsDatasetSent, TopicalChatsKDSentDataset


from dataset_base import DatasetBase
from torch.utils.data import DataLoader


logger = logging.getLogger(__file__)
# logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
logging.basicConfig(level=logging.INFO)


def save_model_config(model, tokenizer, args):
    log_dir = args.log_dir
    torch.save(args, os.path.join(log_dir, args.experiment_name, 'model_training_args.bin'))
    getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir,args.experiment_name, CONFIG_NAME))
    tokenizer.save_pretrained(os.path.join(log_dir, args.experiment_name))


def read_lines(fpath):
    with open(fpath, "r") as fin:
        return fin.read().split("\n")

def make_example(tokenizer, tgt, src, fct):
    def _tokenize(text):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    ex = {
        "src": [_tokenize(sent) for sent in src],
        "tgt": _tokenize(tgt),
        "fct": _tokenize(fct)
    }
    return ex

def get_data_loaders(args, tokenizer, special_tokens,):

    loaders = {}
    for name, data_path, batch_size in [
        ("train", args.train_dataset_path, args.train_batch_size),
        ("valid", args.valid_dataset_path, args.valid_batch_size)]:

        cache_path = f"{data_path}.examples.pt"
        if cache_path and os.path.isfile(cache_path):
            logger.info(f"Load tokenized dataset from cache at {cache_path}")
            examples = torch.load(cache_path)
        else:
            logger.info(f"Loading dataset from {data_path}")
            # Splitting history into multiple sentences for ease of further processing
            # list list string
            src = [l.strip().split("_eos")[:-1] for l in read_lines(f"{data_path}.src")]
            # list string
            tgt = [l.strip().replace("_go", "").replace("_eos", "") for l in read_lines(f"{data_path}.tgt")]
            # list string
            fct = [l.strip() for l in read_lines(f"{data_path}.fct")]

            examples = []
            for s, t, f in zip(src, tgt, fct):
                e = make_example(tokenizer, t, s, f)
                examples.append(e)

            torch.save(examples, cache_path)

        dataset = DatasetBase(examples, tokenizer, special_tokens, args)
        loaders[name] = DataLoader(dataset, sampler=None, batch_size=batch_size,
                                  collate_fn=lambda x: dataset.collate_batch_elements(x, tokenizer, args, special_tokens),
                                  shuffle=True)


    return loaders



def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset_path", type=str, default="processed_output/train",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--valid_dataset_path", type=str, default="processed_output/valid_freq",
                        help="Path or url of the dataset. If empty download from S3.")

    parser.add_argument('--training_configuration', type=str, default="baseline",
                        help="Training configuration to run",
                        choices=["baseline", "kd-pd-nrg", "kd-pd-nrg-swbd", "sentiment"])
    parser.add_argument('--dataset_configuration', type=str, default="topical-chats",
                        help="Configuration of dataset to load for training",
                        choices=["dstc9", "topical-chats"])
    parser.add_argument('--generation_configuration', type=str, default="sentence",
                        help="How the output is generated, sentence by sentence or turn level",
                        choices=["sentence", "turn_level"])
    parser.add_argument('--knowledge_index_path', type=str, default="tc_processed/tc_knowledge_index_bert_all.pkl",
                        help="Path to knowledge index file")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_caches', help="Path or url of the dataset cache")
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

    parser.add_argument('--max_fact_length', type=int, default=200,
                        help='Number of fact tokens to include in the input')

    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument('--log_every_n', type=int, default=500,
                        help='Number of iterations to run before logging')
    # This is a notoriously difficult model to train, so we prefer to save
    # as frequently as possible
    parser.add_argument('--save_every_n', type=int, default=5000,
                        help='Number of iterations to run before saving the model')
    # TODO: implement some mechanism to resume training. Need more sophisticated state management

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument('--parallel', action="store_true",
                        help="Set up data parallelism for training")
    parser.add_argument('--log_dir', type=str, default="runs/",
                        help="Output log directory for summary")
    parser.add_argument('--experiment_name', type=str, default="topical_chats_gpt2",
                        help="Name of experiment for logging and checkpointing purposes")
    # Decoding arguments
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--knowledge_policy", type=str, default="bert_sentence", choices=["tf_idf", "embeddings", "infersent", "bert", "bert_sentence"])
    args = parser.parse_args(["--device", "cuda"])

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d",
                   args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))
    logger.info("torch.cuda.is_available(): " + str(torch.cuda.is_available()))

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    num_added_tokens, orig_num_tokens, tokenizer, special_tokens = setup_tokenizer(args)

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')


    logger.info("Prepare datasets")
    loaders = get_data_loaders(args, tokenizer, special_tokens)


    if args.distributed or args.parallel:
        # Gradient checkpointing significantly slows down distributed training,
        # so we use the original variant of the class for training
        import transformers.modeling_gpt2 as mgpt2
        model_class = mgpt2.GPT2DoubleHeadsModel
    else:
        # Load the model after the tokenizer. We hit an OOM error if we try to pre-load the model
        model_class = GPT2DoubleHeadsModel

    # Hack to evaluate model in the way we saved. TODO: Fix this today
    if os.path.isdir(args.model_checkpoint):
        data = torch.load(args.model_checkpoint + '/pytorch_model.bin')
        model = data["mymodel"]
    else:
        model = model_class.from_pretrained(args.model_checkpoint)

    # Add special tokens if they are not already added
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

    if args.parallel:
        # Setup data parallel version of the model to make
        # use of multi-GPU
        model = torch.nn.DataParallel(model)

    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    scheduler = PiecewiseLinearLR(optimizer, [(0, args.lr), (args.n_epochs * len(loaders["train"]), 0.0)])

    writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, args.experiment_name))


    # Save configuration
    save_model_config(model, tokenizer, args)
    # save_model(model, 'test_checkpoint', args)
    trainer.run_training(model, optimizer, scheduler, loaders, tokenizer, writer, args)


def setup_tokenizer(args):
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)


    # The _nofact token needs to be added
    ADDITIONAL_TOKENS = ["_nofact"]
    SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<end>", "<pad>",
                      "<eot>"]  # added <end>, to represent the end of sent

    ATTR_TO_SPECIAL_TOKEN = {
        'bos_token': '<bos>',
        'eos_token': '<eos>',
        'pad_token': '<pad>',
        'additional_special_tokens': ["<speaker1>", "<speaker2>", "<end>", "<eot>"]
    }

    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_tokens(ADDITIONAL_TOKENS)
    num_added_tokens += tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there


    return num_added_tokens, orig_num_tokens, tokenizer, SPECIAL_TOKENS


if __name__ == '__main__':
    train()
