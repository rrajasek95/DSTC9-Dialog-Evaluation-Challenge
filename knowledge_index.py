"""

"""
import argparse
import json
import os
import pickle
import string

from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def load_split_conversation_data(data_file_path, split):
    split_conversation_path = os.path.join(
        data_file_path,
        'conversations',
        f'{split}.json'
    )

    with open(split_conversation_path, 'r') as split_conversation_file:
        split_conv_data = json.load(split_conversation_file)

    return split_conv_data

def extract_fact_set(factsets):

    sentences = []
    for idx, data in factsets.items():

        fun_facts = data.get("fun_facts")

        if fun_facts:
            for fact in fun_facts:
                sentences.append(clean(fact))

        short_wiki = data.get("shortened_wiki_lead_section")

        if short_wiki:
            sentences.append(clean(short_wiki))

        summarized_wiki = data.get("summarized_wiki_lead_section")

        if summarized_wiki:
            sentences.append(clean(summarized_wiki))
    return sentences


def load_split_reading_set(data_file_path, split):
    split_reading_set_path = os.path.join(
        data_file_path,
        'reading_sets',
        'post-build',
        f'{split}.json'
    )
    with open(split_reading_set_path, 'r') as reading_set_file:
        split_reading_set = json.load(reading_set_file)

    return split_reading_set


def build_knowledge_set(reading_set):
    knowledge_set = set()

    for conv_id, data in reading_set.items():
        knowledge_set.update(extract_fact_set(data["agent_1"]))
        knowledge_set.update(extract_fact_set(data["agent_2"]))

        article_data = data["article"]

        article_indices = ['AS1', 'AS2', 'AS3', 'AS4']

        # Article information
        if "AS1" in article_data:
            for idx in article_indices:
                sentence = article_data[idx]
                if len(word_tokenize(sentence)) < 5:
                    continue
                knowledge_set.add(clean(sentence))

    return knowledge_set


def build_topical_chats_knowledge_index(args):
    data_file_path = os.path.join(
        args.data_dir,
        'alexa-prize-topical-chat-dataset',
    )

    splits = ['train', 'valid_freq', 'valid_rare', 'test_freq', 'test_rare']

    knowledge_set = set()

    for split in splits:
        split_reading_set = load_split_reading_set(data_file_path, split)

        knowledge_set |= build_knowledge_set(split_reading_set)

    corpus = sorted(list(knowledge_set), key=lambda x: len(x))

    vectorizer = TfidfVectorizer()

        with open(os.path.join(data_dir, file_path), 'r') as reading_set_file:
            split_knowledge = json.load(reading_set_file)

    transformed_knowledge_sentences = vectorizer.fit_transform(corpus)


    knowledge_index_path = os.path.join(args.data_dir,
              'tc_processed',
              'tc_knowledge_index.pkl')

    with open(knowledge_index_path, 'wb') as knowledge_index_file:
        pickle.dump({
            "vectorizer": vectorizer,
            "knowledge": corpus,
            "knowledge_vecs": transformed_knowledge_sentences
        }, knowledge_index_file)

"""
Methods derived from the baseline dynamic.py script.

I am using this as an alternative since the fact 
handling seems more robust here than what I did for
topical chats!
"""


def clean(s):
  return ''.join([c if c not in string.punctuation else ' ' for c in s.lower()])


def build_tfidf_from_dstc9(args):
    vectorizer = TfidfVectorizer()
    corpus = []

    for suffix in ["src", "tgt", "fct"]:
        data_file_path = os.path.join(args.data_dir,
                     'processed_output',
                     f"train.{suffix}")

        with open(data_file_path, "r") as file:
            corpus += [e.strip() for e in file.readlines()]
    corpus = [clean(e) for e in corpus]
    vectorizer.fit(corpus)

    potential_facts = [e.strip() for e in open("../processed_output/train.fct").readlines()]
    potential_facts = [e for e in potential_facts if len(e.split()) < 20]

    index_path = os.path.join(args.data_dir,
                           'processed_output',
                           'knowledge_index_dstc9.pkl')
    with open(index_path, 'wb') as index_file:
        index_dict = {
            "tfidf_vec": vectorizer,
            "knowledge_list": potential_facts
        }

        pickle.dump(index_dict, index_file)
        print(f"Index has been built and saved to '{index_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default="dstc9",
                        choices=['topical_chats', 'dstc9'],
                        help="Dataset for knowledge")

    parser.add_argument('--data_dir', type=str, default='./',
                        help='Directory where the topical chats folder is present')
    args = parser.parse_args()
    if args.dataset_type == "dstc9":
        build_tfidf_from_dstc9(args)
    else:
        build_topical_chats_knowledge_index(args)
