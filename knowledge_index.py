"""

"""
import argparse
import json
import os
import string
import glove.glove_utils
from encoder.fb_models import InferSent
from glove.glove_utils import get_sentence_glove_embedding
import pickle

from nltk import word_tokenize
from nltk import sent_tokenize
import torch
import nltk
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


def load_json_data(data_file_path, split):
    split_conversation_path = os.path.join(
        data_file_path,
        f'{split}_anno.json'
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


def build_fb_embs_set(reading_set, infersent, knowledge_convo_embs):

    for conv_id, data in reading_set.items():
        knowledge_sents = extract_fact_set(data["agent_1"])
        knowledge_sents += extract_fact_set(data["agent_2"])

        article_data = data["article"]

        article_indices = ['AS1', 'AS2', 'AS3', 'AS4']

        # Article information
        if "AS1" in article_data:
            for idx in article_indices:
                sentence = article_data[idx]
                if len(word_tokenize(sentence)) < 5:
                    continue
                knowledge_sents.append(clean(sentence))

        knowledge_convo_embs[conv_id] = []
        for sent in knowledge_sents:
            embeddings = infersent.encode([sent], tokenize=True)
            knowledge_convo_embs[conv_id].append([sent, embeddings[0]])

    return knowledge_convo_embs


def build_knowledge_set(reading_set, knowledge_convo_embs=None, embs=False, emb_matrix=None, tokenizer=None):
    knowledge_set = set()

    if embs:
        for conv_id, data in reading_set.items():
            knowledge_sents = extract_fact_set(data["agent_1"])
            knowledge_sents += extract_fact_set(data["agent_2"])

            article_data = data["article"]

            article_indices = ['AS1', 'AS2', 'AS3', 'AS4']

            # Article information
            if "AS1" in article_data:
                for idx in article_indices:
                    sentence = article_data[idx]
                    if len(word_tokenize(sentence)) < 5:
                        continue
                    knowledge_sents.append(clean(sentence))

            knowledge_convo_embs[conv_id] = []
            for sent in knowledge_sents:
                knowledge_convo_embs[conv_id].append([sent, get_sentence_glove_embedding(sent, emb_matrix, tokenizer)])
        return knowledge_convo_embs

    else:
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


def get_messages(data, corpus):
    for k, v in data.items():
        for mes in v["content"]:
            sents = sent_tokenize(mes["message"])
            for sent in sents:
                corpus.append(sent)
    return corpus


def build_topical_chats_knowledge_index_glove(args):
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
    all_data = {}
    for split in splits:
        data = load_json_data('./tc_processed', split)
        all_data.update(data)

    corpus = get_messages(all_data, corpus)
    glove_obj = glove.glove_utils.GloveUtils(corpus)

    knowledge_index_path = os.path.join(args.data_dir,
                                        'tc_processed',
                                        'tc_knowledge_index_glove.pkl')

    with open(knowledge_index_path, 'wb') as knowledge_index_file:
        pickle.dump({
            "tokenizer": glove_obj.tokenizer,
            # "knowledge": corpus,
            "embedding_matrix": glove_obj.embedding_matrix,
            # "vocab": glove_obj.vocab,
            # "embeddings_index": glove_obj.embeddings_index
        }, knowledge_index_file)



def build_topical_chats_knowledge_index_facebook(args):
    data_file_path = os.path.join(
        args.data_dir,
        'alexa-prize-topical-chat-dataset',
    )

    splits = ['train', 'valid_freq', 'valid_rare', 'test_freq', 'test_rare']

    V = 2
    MODEL_PATH = 'encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    W2V_PATH = 'fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)

    infersent.build_vocab_k_words(K=100000)
    knowledge_convo_embs = {}

    for split in splits:
        split_reading_set = load_split_reading_set(data_file_path, split)
        knowledge_convo_embs = build_fb_embs_set(split_reading_set, infersent, knowledge_convo_embs)

    knowledge_index_path = os.path.join(args.data_dir, 'tc_processed', 'tc_knowledge_index_facebook.pkl')

    with open(knowledge_index_path, 'wb') as knowledge_index_file:
        pickle.dump({
            "knowledge_vecs": knowledge_convo_embs
        }, knowledge_index_file)


def build_conversation_knowledge_embeddings(args, glove_embs_path):
    with open(glove_embs_path, 'rb') as knowledge_index_file:
        index_data = pickle.load(knowledge_index_file)

    emb_matrix = index_data["embedding_matrix"]
    tokenizer = index_data["tokenizer"]

    data_file_path = os.path.join(
        args.data_dir,
        'alexa-prize-topical-chat-dataset',
    )

    splits = ['train', 'valid_freq', 'valid_rare', 'test_freq', 'test_rare']

    knowledge_convo_embs = {}

    for split in splits:
        split_reading_set = load_split_reading_set(data_file_path, split)
        knowledge_convo_embs = build_knowledge_set(split_reading_set, knowledge_convo_embs=knowledge_convo_embs,
                                                   embs=True, emb_matrix=emb_matrix, tokenizer=tokenizer)

    knowledge_index_path = os.path.join(args.data_dir,
                                        'tc_processed',
                                        'tc_knowledge_sent_embs.pkl')

    with open(knowledge_index_path, 'wb') as knowledge_index_file:
        pickle.dump({
            "tokenizer": tokenizer,
            "emb_matrix": emb_matrix,
            "knowledge_vecs": knowledge_convo_embs
        }, knowledge_index_file)


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

    parser.add_argument('--knowledge_policy', type=str, default='facebook', choices=['glove', 'tf_idf', 'facebook'])
    args = parser.parse_args()
    if args.knowledge_policy == 'facebook':
        build_topical_chats_knowledge_index_facebook(args)
    elif args.knowledge_policy == "glove":
        build_topical_chats_knowledge_index_glove(args)
        build_conversation_knowledge_embeddings(args, 'tc_processed/tc_knowledge_index_glove.pkl')
    elif args.dataset_type == "dstc9":
        build_tfidf_from_dstc9(args)
    else:
        build_topical_chats_knowledge_index(args)
