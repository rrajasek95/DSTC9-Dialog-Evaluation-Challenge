import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from nltk import word_tokenize, sent_tokenize

from encoder.fb_models import InferSent
from glove.glove_utils import get_cosine_similarity_embs_all
from knowledge_index import extract_fact_set, clean
from sentence_transformers import SentenceTransformer



def generate_knowledge_selections(test_freq_conversations, test_freq_reading_set, vectorizer):

    turn_knowledge = []
    for conv_id, conv_data in test_freq_conversations.items():
        agent_knowledge = get_conversation_knowledge(conv_id, test_freq_reading_set)

        for i, turn in enumerate(conv_data["content"]):
            text = turn["message"]

            available_knowledge = agent_knowledge[turn["agent"]]

            text_tfidf = vectorizer.transform([clean(text)])
            knowledge_tfidf = vectorizer.transform(available_knowledge)
            similarity = np.squeeze(np.asarray(text_tfidf.dot(knowledge_tfidf.transpose()).todense()))

            top_n_indices = similarity.argsort()[-3:][::-1].tolist()

            knowledge_1 = available_knowledge[top_n_indices[0]]
            if i > 0:
                same_as_prev_knowledge = knowledge_1 == turn_knowledge[-1]["knowledge_1"]
            else:
                same_as_prev_knowledge = False

            data = {
                "conversation_id": conv_id,
                "turn": (i + 1),
                "text": text,
                "knowledge_1": knowledge_1,
                "knowledge_1_similarity": similarity[top_n_indices[0]],
                "knowledge_2": available_knowledge[top_n_indices[1]],
                "knowledge_2_similarity": similarity[top_n_indices[1]],
                "knowledge_3": available_knowledge[top_n_indices[2]],
                "knowledge_3_similarity": similarity[top_n_indices[2]],
                "same_as_prev_knowledge": same_as_prev_knowledge
            }

            turn_knowledge.append(data)

    test_freq_knowledge_dataframe = pd.DataFrame(turn_knowledge,
                                                 columns=[
                                                     'conversation_id',
                                                     'turn',
                                                     'text',
                                                     'knowledge_1',
                                                     'knowledge_1_similarity',
                                                     'knowledge_2',
                                                     'knowledge_2_similarity',
                                                     'knowledge_3',
                                                     'knowledge_3_similarity',
                                                     'same_as_prev_knowledge'
                                                 ])

    test_freq_knowledge_dataframe.to_csv(os.path.join(
        'tc_processed',
        'test_freq_tfidf.csv'
    ))


def get_conversation_knowledge(conv_id, test_freq_reading_set):
    conv_reading_set = test_freq_reading_set[conv_id]
    fact_set_1 = set(extract_fact_set(conv_reading_set["agent_1"]))
    fact_set_2 = set(extract_fact_set(conv_reading_set["agent_2"]))
    article_data = conv_reading_set["article"]
    article_indices = ['AS1', 'AS2', 'AS3', 'AS4']
    common_knowledge_set = set()
    if "AS1" in article_data:
        for idx in article_indices:
            sentence = article_data[idx]
            if len(word_tokenize(sentence)) < 5:
                continue
            common_knowledge_set.add(clean(sentence))
    fact_set_1.update(common_knowledge_set)
    fact_set_2.update(common_knowledge_set)
    agent_knowledge = {
        "agent_1": list(fact_set_1),
        "agent_2": list(fact_set_2)
    }
    return agent_knowledge


def generate_knowledge_infersent(test_freq_conversations, test_freq_reading_set, vectorizer):
    infersent = load_infersent_model()
    turn_knowledge = []

    for conv_id, conv_data in test_freq_conversations.items():
        for i, turn in enumerate(conv_data["content"]):
            text = turn["message"]
            available_knowledge = vectorizer[conv_id]


            fact_sims = get_cosine_similarity_embs_all(clean(text), available_knowledge, infersent)
            fact_sims.sort(key=lambda x: x[1], reverse=True)
            if i > 0:
                same_as_prev_knowledge = fact_sims[0][0] == turn_knowledge[-1]["knowledge_1"]
            else:
                same_as_prev_knowledge = False
            data = {
                "conversation_id": conv_id,
                "turn": (i + 1),
                "text": text,
                "knowledge_1": fact_sims[0][0],
                "knowledge_1_similarity": fact_sims[0][1],
                "knowledge_2": fact_sims[1][0],
                "knowledge_2_similarity": fact_sims[1][1],
                "knowledge_3": fact_sims[2][0],
                "knowledge_3_similarity": fact_sims[2][1],
                "same_as_prev_knowledge": same_as_prev_knowledge
            }
            turn_knowledge.append(data)
    test_freq_knowledge_dataframe = pd.DataFrame(turn_knowledge,
                                                 columns=[
                                                     'conversation_id',
                                                     'turn',
                                                     'text',
                                                     'knowledge_1',
                                                     'knowledge_1_similarity',
                                                     'knowledge_2',
                                                     'knowledge_2_similarity',
                                                     'knowledge_3',
                                                     'knowledge_3_similarity',
                                                     'same_as_prev_knowledge'
                                                 ])

    test_freq_knowledge_dataframe.to_csv(os.path.join(
        'tc_processed',
        'test_freq_infersent_group_3_articles.csv'
    ))


def load_infersent_model():
    V = 2
    MODEL_PATH = 'encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    W2V_PATH = 'fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)
    infersent.build_vocab_k_words(K=100000)
    return infersent


def generate_knowledge_bert(test_freq_conversations, test_freq_reading_set, vectorizer):
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    turn_knowledge = []

    for conv_id, conv_data in test_freq_conversations.items():
        for i, turn in enumerate(conv_data["content"]):
            text = turn["message"]
            available_knowledge = vectorizer[conv_id]

            fact_sims = get_cosine_similarity_embs_all(clean(text), available_knowledge, model, knowledge_policy="bert")
            fact_sims.sort(key=lambda x: x[1], reverse=True)
            if i > 0:
                same_as_prev_knowledge = fact_sims[0][0] == turn_knowledge[-1]["knowledge_1"]
            else:
                same_as_prev_knowledge = False
            data = {
                "conversation_id": conv_id,
                "turn": (i + 1),
                "text": text,
                "knowledge_1": fact_sims[0][0],
                "knowledge_1_similarity": fact_sims[0][1],
                "knowledge_2": fact_sims[1][0],
                "knowledge_2_similarity": fact_sims[1][1],
                "knowledge_3": fact_sims[2][0],
                "knowledge_3_similarity": fact_sims[2][1],
                "same_as_prev_knowledge": same_as_prev_knowledge
            }
            turn_knowledge.append(data)
    test_freq_knowledge_dataframe = pd.DataFrame(turn_knowledge,
                                                 columns=[
                                                     'conversation_id',
                                                     'turn',
                                                     'text',
                                                     'knowledge_1',
                                                     'knowledge_1_similarity',
                                                     'knowledge_2',
                                                     'knowledge_2_similarity',
                                                     'knowledge_3',
                                                     'knowledge_3_similarity',
                                                     'same_as_prev_knowledge'
                                                 ])

    test_freq_knowledge_dataframe.to_csv(os.path.join(
        'tc_processed',
        'test_freq_bert_summarize_arts.csv'
    ))


def generate_knowledge_bert_sent(test_freq_conversations, test_freq_reading_set, vectorizer):
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    turn_knowledge = []

    for conv_id, conv_data in test_freq_conversations.items():
        for i, turn in enumerate(conv_data["content"]):
            message = turn["message"]
            sentences = sent_tokenize(message)
            for text in sentences:
                available_knowledge = vectorizer[conv_id]

                fact_sims = get_cosine_similarity_embs_all(clean(text), available_knowledge, model, knowledge_policy="bert")
                fact_sims.sort(key=lambda x: x[1], reverse=True)
                if i > 0:
                    same_as_prev_knowledge = fact_sims[0][0] == turn_knowledge[-1]["knowledge_1"]
                else:
                    same_as_prev_knowledge = False
                data = {
                    "conversation_id": conv_id,
                    "turn": (i + 1),
                    "text": text,
                    "knowledge_1": fact_sims[0][0],
                    "knowledge_1_similarity": fact_sims[0][1],
                    "knowledge_2": fact_sims[1][0],
                    "knowledge_2_similarity": fact_sims[1][1],
                    "knowledge_3": fact_sims[2][0],
                    "knowledge_3_similarity": fact_sims[2][1],
                    "same_as_prev_knowledge": same_as_prev_knowledge
                }
                turn_knowledge.append(data)
    test_freq_knowledge_dataframe = pd.DataFrame(turn_knowledge,
                                                 columns=[
                                                     'conversation_id',
                                                     'turn',
                                                     'text',
                                                     'knowledge_1',
                                                     'knowledge_1_similarity',
                                                     'knowledge_2',
                                                     'knowledge_2_similarity',
                                                     'knowledge_3',
                                                     'knowledge_3_similarity',
                                                     'same_as_prev_knowledge'
                                                 ])

    test_freq_knowledge_dataframe.to_csv(os.path.join(
        'tc_processed',
        'test_freq_bert_sentences.csv'
    ))

def load_test_freq_data(args):
    test_freq_reading_set_path = os.path.join(
        args.reading_set_path,
        'test_freq.json'
    )

    test_freq_conversations_path = os.path.join(
        args.conversations_path,
        'test_freq.json'
    )

    with open(test_freq_conversations_path, 'r') as test_freq_conv_f:
        test_freq_conversations = json.load(test_freq_conv_f)

    with open(test_freq_reading_set_path, 'r') as test_freq_reading_set_file:
        test_freq_reading_set = json.load(test_freq_reading_set_file)

    return test_freq_conversations, test_freq_reading_set


def load_infersent_knowledge_index(args):
    with open(args.knowledge_index_path, 'rb') as knowledge_index_file:
        index_dict = pickle.load(knowledge_index_file)
        vectorizer = index_dict["knowledge_vecs"]

    return vectorizer

def load_bert_knowledge_index(args):
    with open(args.knowledge_index_path, 'rb') as knowledge_index_file:
        index_dict = pickle.load(knowledge_index_file)
        vectorizer = index_dict["knowledge_vecs"]

    return vectorizer

def load_tfidf_knowledge_index(args):
    with open(args.knowledge_index_path, 'rb') as knowledge_index_file:
        index_dict = pickle.load(knowledge_index_file)
        vectorizer = index_dict["vectorizer"]

    return vectorizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--knowledge_selection_policy", type=str, default="bert", choices=["tf_idf", "infersent", "bert"])
    parser.add_argument('--conversations_path', type=str, default=
                        os.path.join(
                            'alexa-prize-topical-chat-dataset',
                            'conversations'
                        ))

    parser.add_argument('--reading_set_path', type=str, default=
                        os.path.join('alexa-prize-topical-chat-dataset',
                                     'reading_sets',
                                     'post-build'
                                     ), help='Path to topical chats reading set JSON files')
    parser.add_argument('--knowledge_index_path', type=str,
                        default=os.path.join(
                            'tc_processed',
                            'tc_knowledge_index_facebook_test_freq_shortened_arts.pkl'
                        ),
                        help='Path to knowledge index file/folder')
    parser.add_argument('--output_selection_path', type=str,
                        default=os.path.join(
                            'knowledge_selection',
                            'processed_knowledge_selection'),
                        help='Path to output file for match knowledge selections')
    args = parser.parse_args()

    if args.knowledge_selection_policy == "infersent":
        vectorizer = load_infersent_knowledge_index(args)
    if args.knowledge_selection_policy == "bert":
        vectorizer = load_bert_knowledge_index(args)
    else:
        vectorizer = load_tfidf_knowledge_index(args)



    test_freq_conversations, test_freq_reading_set = load_test_freq_data(args)

    if args.knowledge_selection_policy == "tf_idf":
        generate_knowledge_selections(test_freq_conversations, test_freq_reading_set, vectorizer)
    elif args.knowledge_selection_policy == "infersent":
        generate_knowledge_infersent(test_freq_conversations, test_freq_reading_set, vectorizer)
    else:
        # generate_knowledge_bert(test_freq_conversations, test_freq_reading_set, vectorizer)
        generate_knowledge_bert_sent(test_freq_conversations, test_freq_reading_set, vectorizer)