"""
 Created by diesel
 10/8/20
"""


import argparse
import itertools
import json
import logging
import os
from pprint import pformat

from itertools import chain
import pickle
import random

import torch
from nltk import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

from collections import defaultdict

from torch.utils.data import DataLoader
from transformers import AdamW, GPT2Tokenizer

from dataset_base import DatasetBase


logger = logging.getLogger(__file__)
# logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
logging.basicConfig(level=logging.INFO)

def read_lines(fpath):
    with open(fpath, "r") as fin:
        return fin.read().split("\n")


def make_example(tokenizer, tgt, src, fct):
    """
    # For the baseline implementation, we don't need to consider the DA
    (history, (response, _, fact)) = self.dataset[index]

    """
    #print("src:", src) # list str
    #print("-- -- -- --")
    #print("tgt:", [tgt]) # str
    #print("-- -- -- --")
    #print("fct:", [fct]) #
    #print("-- -- -- --")

    def _tokenize(text):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    ex = {
        "src": [_tokenize(sent) for sent in src],
        "tgt": _tokenize(tgt),
        "fct": _tokenize(fct)
    }

    #for k,v in ex.items():
    #    print("\n", k)
    #    print(v)

    return ex



def tokenize(obj, tokenizer):
    if obj is None:
        done = None
    elif isinstance(obj, str):
        done = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    elif isinstance(obj, tuple):
        done = tuple(tokenize(o, tokenizer) for o in obj)
    else:
        done = list(tokenize(o, tokenizer) for o in obj)
    return done


def truncate_sequences(args, history, fact):
    # Truncate history turns to reduce memory requirement
    if len(history) > (2 * args.max_history + 1):
        history = history[-(2 * args.max_history + 1):]

    # Truncate facts to decrease overall input length
    trunc_facts = fact[:min(len(fact), args.max_fact_length)]
    return history, trunc_facts


def sample_candidates(args, dataset):
    # Lets just hope that the number of cases where the true responses gets included in the
    # candidates is vanishingly small
    candidates = [ex["tgt"] for ex in random.sample(dataset, args.num_candidates - 1)]

    return candidates




def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def collate_batch_elements(batch, tokenizer, args, special_tokens):
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

    padded_dataset = pad_dataset(batch_inputs, padding=tokenizer.convert_tokens_to_ids(special_tokens[-2]))

    tensorized_input = []
    # Verify input sent the same way:
    #
    # "input_ids": [Batch size, num_cands, seq_len]
    # "mc_token_ids": [Batch size, num cands],
    # "lm_labels": [batch size, num_cands, seq_len]
    # "mc_labels": [batch_size]
    # "token_type_ids": [batch_size, num_cands, seq_len]

    MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]

    batch_size = tuple([len(batch_inputs[MODEL_INPUTS[0]])//args.num_candidates])
    for input_name in MODEL_INPUTS:
        tensor = torch.tensor(padded_dataset[input_name])

        if input_name != "mc_labels":
            tensor = tensor.view((-1, args.num_candidates) + tensor.shape[1:])
        else:
            tensor = torch.ones(size=batch_size, dtype=torch.long) * (args.num_candidates - 1)
        tensorized_input.append(tensor)
    return tensorized_input


def main():



    parser = argparse.ArgumentParser()
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument('--max_fact_length', type=int, default=200,
                        help='Number of fact tokens to include in the input')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")


    args = parser.parse_args()
    logger.info("Arguments: %s", pformat(args))

    data_path = "./processed_output/valid_freq"
    working_data = {}

    #for side in ["src", "tgt", "fct"]:
    #    lines = read_lines(f"{data_path}.{side}")
    #    for l in lines:
    #        print("line:", l)
    #        print("processed:", l.strip())

    # Splitting history into multiple sentences for ease of further processing
    # list list string
    src = [l.strip().split("_eos")[:-1] for l in read_lines(f"{data_path}.src")]
    # list string
    tgt = [l.strip().replace("_go", "").replace("_eos", "") for l in read_lines(f"{data_path}.tgt")]
    # list string
    fct = [l.strip() for l in read_lines(f"{data_path}.fct")]



    #history_da = itertools.repeat(itertools.repeat(None))
    #history_knowledge = itertools.repeat(itertools.repeat(None))
    #resp_da = itertools.repeat(None)
    #context = [zip(s, h, k) for (s, h, k) in zip(src, history_da, history_knowledge)]
    #the_data = list(zip(context, zip(tgt, resp_da, fct)))


    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained("gpt2-medium")

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
    num_added_tokens = tokenizer.add_tokens(ADDITIONAL_TOKENS)
    num_added_tokens += tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there


    examples = []
    for s, t, f in zip(src, tgt, fct):
        e = make_example(tokenizer, t, s, f)
        examples.append(e)

    dataset = DatasetBase(examples, tokenizer, SPECIAL_TOKENS, args)


    train_loader = DataLoader(dataset, sampler=None, batch_size=4,
                              collate_fn=lambda x: dataset.collate_batch_elements(x, tokenizer, args, SPECIAL_TOKENS),
                              shuffle=True)


    print("done building dataset ...")

    print("looping through train loader")
    for i, batch in tqdm(enumerate(train_loader)):

        #print("\nbatch:")
        #print(batch)

        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch

    exit(0)

    #the_data = examples
    #the_data = tokenize(the_data, tokenizer)

    print("len the_data:", len(the_data))
    """
    for it in the_data:
        print(" * len data_it:", len(it))
    >>>>
    len the_data: 11275
     * len data_it: 2
     * len data_it: 2
     * ...
    """

    #print("\nthe_data[0][0]:", the_data[0][0])
    # history
    history = [
        (
            [31373, 764, 466, 345, 711, 597, 2008, 1830, 5633, 220],
            None,
            None
        )
    ]

    history = [h[0] for h in history]

    #print("\nthe_data[0][1]:", the_data[0][1])
    x = (
        # response
        [1312, 466, 711, 1830, 764, 1312, 2883, 262, 869, 286, 7077, 983, 2168, 764, 220],
        # _
        None,
        # fact
        [1, 983, 11915, 474, 1531, 36650, 14520, 328, 282, 10488, 326, 355, 257, 5440, 356, 711, 1115, 2997, 2250, 286, 2008, 1830, 257, 1285, 764, 366]
    )


    for jj, item in enumerate(the_data):

        #history, (response, _, fact) = item

        history = item["src"]
        response = item["tgt"]
        fact = item["fct"]

        print("\n\nhistory[0]:", history[0])
        print("response:", response)


        #history = [h[0] for h in history]
        history, fact = truncate_sequences(args, history, fact)

        print("history[0]:", len(history[0]))
        print("\n\nhistory[0]:", history[0])
        print("\n\nhistory:", history)
        print("history[0][0]:", history[0][0])

        candidates = sample_candidates(args, the_data)
        candidates.append(response)

        instances = []
        for j, candidate in enumerate(candidates):
            lm_labels = bool(j == args.num_candidates - 1)

            bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids((SPECIAL_TOKENS[:4]))
            sequence = (
                    # fact
                    [[bos] + fact] +
                    # history
                    history +
                    # response
                    [candidate + [eos]]
            )

            print("\nsequence:")
            print(sequence)

            # add speaker token to beginning of turns
            sequence = [sequence[0]] + [
                [speaker2 if (len(sequence) - i) % 2 else speaker1] + s
                for i, s in enumerate(sequence[1:])
            ]

            print("\nsequence:")
            print(sequence)

            sequence2 = list(chain(*sequence))
            print("\nsequence2:")
            print(sequence2)


            instance = {}
            instance["input_ids"] = list(chain(*sequence))
            instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
            instance["mc_token_ids"] = len(instance["input_ids"]) - 1
            instance["lm_labels"] = [-100] * len(instance["input_ids"])
            if lm_labels:
                instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]


            instances.append(instance)

        if jj > 2:
            exit(0)


        return_item = instances



    """
    examples = []
    for s, t, f in zip(src, tgt, fct):
        e = make_example(tokenizer, t, s, f)
        examples.append(e)

        if len(examples) > 6:
            break
    my_dataset = DatasetBase(examples, tokenizer, SPECIAL_TOKENS, args)
    """




if __name__ == "__main__":
    main()
