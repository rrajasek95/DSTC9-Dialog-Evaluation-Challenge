import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import describe
from transformers import GPT2Tokenizer, GPT2Config
from numpy import percentile

from utils import get_dataset

"""
Most of the analysis being done here is to identify 
potential bottlenecks in running modeling experiments.

One key reason is that the models can't have long inputs. For e.g.
running GPT2-medium on a (4 * 2 * 960) size crashes CoLab. 

Useful things to find out:
1. Distribution of conversation lengths (can use this to find a sensible num. history turns)
2. Average num of tokens per turn (Need this to find out how many tokens are present in an average input)
3. Expected context size: E[num_history_tokens]
4. Average output size
5. Distribution of knowledge length
6. Distribution of input size: E[knowledge_tokens + num_history_tokens + response_tokens]
"""

def analyze_split(split, meta):
    # Average conversation length
    meta_info = defaultdict(list)
    for dialog in split:
        (history, reply, fact) = dialog
        meta_info["history_length"].append(len(history))

        num_history_tokens = sum(len(toks) for toks in history)
        meta_info["num_history_tokens"].append(num_history_tokens)
        reply_len = len(reply)
        meta_info["reply_token_count"].append(reply_len)
        fact_len = len(fact)
        meta_info["fact_token_count"].append(fact_len)
        meta_info["num_total_tokens"].append(num_history_tokens + reply_len + fact_len)

    results_path = os.path.join('analysis_results', f'dataset_analysis_{meta["title"]}.txt')
    with open(results_path, 'w') as data_file:
        data_file.write(f"Split: {meta['title']}\n")
        for metric, values in meta_info.items():
            data_file.write(f"{metric}\n")
            data_file.write(f"{describe(values)}\n")
            data_file.write(f"{percentile(values, [50, 75, 90, 95, 99])}\n\n")

        print(f"Results saved to {results_path}")


def analyze_dataset(dataset):

    dataset_meta = {
        "train": {
            "title": "Train"
        },
        "valid_freq": {
            "title": "Valid Frequent"
        }
    }

    for split, meta in dataset_meta.items():
        data_split = dataset[split]
        analyze_split(data_split, meta)

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    dataset_path = 'processed_output'
    dataset_cache = './dataset_cache'

    dataset = get_dataset(tokenizer, dataset_path, dataset_cache)

    analyze_dataset(dataset)