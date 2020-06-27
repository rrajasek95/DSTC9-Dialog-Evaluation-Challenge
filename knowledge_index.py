"""

"""
import argparse
import json
import os
import pickle

import nltk
import spacy
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
        #
        #         knowledge_set.update(sent_tokenize(sentence))
    return knowledge_set

def index_knowledge(args):
    nlp = spacy.load('en_core_web_lg')

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./',
                        help='Directory where the topical chats folder is present')
    args = parser.parse_args()
    index_knowledge(args)