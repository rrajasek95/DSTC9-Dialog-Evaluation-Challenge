import string

import numpy as np

from elastic.topical_chats import TopicalChatsIndexRetriever
from glove.glove_utils import get_max_cosine_similarity, get_max_cosine_similarity_embs_models
from encoder.fb_models import InferSent
import torch
import os
import pickle


from sentence_transformers import SentenceTransformer


def clean(s):
    return ''.join([c if c not in string.punctuation else ' ' for c in s.lower()])

class TfIdfRankerRetriever(object):
    """
    A module that performs retrieval from an index and also performs top-n ranking.
    This forms a component of the heuristic knowledge selection policy for the KD-PD-NRG
    """

    def _clean(self, s):
        return ''.join([c for c in s.lower() if c not in string.punctuation])

    def __init__(self, knowledge_index, new_index=False):
        if new_index:
            self.tfidf_vec = knowledge_index["vectorizer"]
            self.knowledge_sentences = knowledge_index["knowledge"]
            self.vectorized_sentences = knowledge_index["knowledge_vecs"]
        else:
            self.tfidf_vec = knowledge_index["tfidf_vec"]
            self.knowledge_sentences = knowledge_index["knowledge_list"]
            self.vectorized_sentences = self.tfidf_vec.transform(self.knowledge_sentences)

    def get_top_n(self, query, n=5):
        # These two lines are derived from dynamic.py of the baseline code, with modifications
        # to enable top-n selection
        query_vector = self.tfidf_vec.transform([self._clean(query)])
        similarity = np.squeeze(np.asarray(query_vector.dot(self.vectorized_sentences.transpose()).todense()))
        top_n_indices = similarity.argsort()[-n:][::-1].tolist()
        retrieve_rank_list = [(self.knowledge_sentences[i], similarity[i]) for i in top_n_indices]
        return retrieve_rank_list

    # TODO : get_top_fact(query, convo_tag, threshold=False)


class EmbRankerRetriever(object):

    def _clean(self, s):
        return ''.join([c for c in s.lower() if c not in string.punctuation])

    def __init__(self, knowledge_index):
        self.tokenizer = knowledge_index["tokenizer"]
        self.emb_matrix = knowledge_index["emb_matrix"]
        self.knowledge_vecs = knowledge_index["knowledge_vecs"]
        self.threshold = 0.7

    # TODO: Will need to implement way to select best knowledge without access to convo_tag
    def get_top_n(self, query, convo_tag, n=1):
        knowledge = self.knowledge_vecs[convo_tag]
        fact = get_max_cosine_similarity(query, knowledge, self.emb_matrix, self.tokenizer)
        return fact

    def get_top_fact(self, query, convo_id, threshold=False):
        knowledge = self.knowledge_vecs[convo_id]
        fact, sim = get_max_cosine_similarity(query, knowledge, self.emb_matrix, self.tokenizer)

        if threshold:
            knowledge_sentence = fact if sim > self.threshold else ""
        else:
            knowledge_sentence = fact
        return knowledge_sentence


class ElasticRankerRetriever(object):
    """
    Used to implement a more general knowledge retriever
    for querying elasticsearch for data
    """
    def __init__(self, host, port, alias):
        self.tc_retriever = TopicalChatsIndexRetriever(host, port, alias)

    def get_top_n(self, query, n=5):
        return self.tc_retriever.retrieve_facts_with_score(query)

    # TODO : get_top_fact(query, convo_tag, threshold=False)


class BertRankerRetriever(object):
    def __init__(self, knowledge_index):
        self.knowledge_vecs = knowledge_index["knowledge_vecs"]
        self.model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        self.threshold = 0.35

    def get_top_fact(self, query, convo_id, threshold=False):
        knowledge = self.knowledge_vecs[convo_id]
        fact, sim = get_max_cosine_similarity_embs_models(clean(query), knowledge, self.model, knowledge_policy="bert")

        if threshold:
            knowledge_sentence = fact if sim > self.threshold else ""
        else:
            knowledge_sentence = fact

        return knowledge_sentence


class InfersentRankerRetriever(object):
    def __init__(self, knowledge_vecs):
        self.knowledge_vecs = knowledge_vecs
        self.model = self.load_infersent_model()
        self.threshold = 0.6

    def get_top_fact(self, query, convo_id, threshold=False):
        knowledge = self.knowledge_vecs[convo_id]
        fact, sim = get_max_cosine_similarity_embs_models(clean(query), knowledge, self.model)

        if threshold:
            knowledge_sentence = fact if sim > self.threshold else ""
        else:
            knowledge_sentence = fact

        return knowledge_sentence


    def load_infersent_model(self):
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
