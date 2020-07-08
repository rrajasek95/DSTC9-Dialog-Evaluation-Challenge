"""

"""
import argparse
import json
import os
import pickle
import string

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_fact_set(factsets):

    sentences = []
    for idx, data in factsets.items():

        fun_facts = data.get("fun_facts")

        if fun_facts:
            for fact in fun_facts:
                sentences.append(fact)

        short_wiki = data.get("shortened_wiki_lead_section")

        if short_wiki:
            for sent in sent_tokenize(short_wiki):
                sentences.append(sent)

        summarized_wiki = data.get("summarized_wiki_lead_section")

        if summarized_wiki:
            for sent in sent_tokenize(summarized_wiki):
                sentences.append(sent)
    return sentences



def extract_knowledge_sentences(split_knowledge):

    knowledge_set = set()
    for conv_id, data in split_knowledge.items():

        knowledge_set.update(extract_fact_set(data["agent_1"]))
        knowledge_set.update(extract_fact_set(data["agent_2"]))

        article_data = data["article"]

        article_indices = ['AS1', 'AS2', 'AS3', 'AS4']

        # Article information
        # if "AS1" in article_data:
        #     for idx in article_indices:
        #         sentence = article_data[idx]
        #         for sent in sent_tokenize(sentence):
        #             if len(word_tokenize(sent)) < 5:
        #                 continue
        #             knowledge_set.add(sent)
    return knowledge_set


def index_knowledge(args):
    data_dir = os.path.join(
        args.data_dir,
        'alexa-prize-topical-chat-dataset',
        'reading_sets',
        'post-build',
    )
    reading_set_files = os.listdir(data_dir)

    knowledge_list = build_knowledge_set(data_dir, reading_set_files)

    tfidf_vec = TfidfVectorizer(tokenizer=nltk.tokenize.word_tokenize)
    tfidf_vec.fit(knowledge_list)

    knowledge_index = BM25Okapi(knowledge_list, tokenizer=word_tokenize)

    with open(os.path.join(args.data_dir,
                           'tc_processed',
                           'knowledge_index.pkl'), 'wb') as index_file:
        index_dict = {
            "tfidf_vec": tfidf_vec,
            "bm25_index": knowledge_index,
            "knowledge_list": knowledge_list
        }
        pickle.dump(index_dict, index_file)


def build_knowledge_set(data_dir, reading_set_files):
    knowledge_set = set()
    for file_path in reading_set_files:
        if 'hash' in file_path:
            continue

        with open(os.path.join(data_dir, file_path), 'r') as reading_set_file:
            split_knowledge = json.load(reading_set_file)

        knowledge_set.update(extract_knowledge_sentences(split_knowledge))
    knowledge_list = list(knowledge_set)
    print(len(knowledge_set))
    return knowledge_list


"""
Methods derived from the baseline dynamic.py script.

I am using this as an alternative since the fact 
handling seems more robust here than what I did for
topical chats!
"""


def clean(s):
  return ''.join([c for c in s.lower() if c not in string.punctuation])


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

    index_path = os.path.join(args.data_dir,
                           'processed_output',
                           'knowledge_index_dstc9.pkl')
    with open(index_path, 'wb') as index_file:
        index_dict = {
            "tfidf_vec": vectorizer,
            "knowledge_list": corpus
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
        index_knowledge(args)
