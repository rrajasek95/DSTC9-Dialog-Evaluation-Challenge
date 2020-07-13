import string
import numpy as np
from glove.glove_utils import get_max_cosine_similarity


class TfIdfRankerRetriever(object):
    """
    A module that performs retrieval from an index and also performs top-n ranking.
    This forms a component of the heuristic knowledge selection policy for the KD-PD-NRG
    """

    def _clean(self, s):
        return ''.join([c for c in s.lower() if c not in string.punctuation])

    def __init__(self, knowledge_index):
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


class EmbRankerRetriever(object):

    def _clean(self, s):
        return ''.join([c for c in s.lower() if c not in string.punctuation])

    def __init__(self, knowledge_index):
        self.tokenizer = knowledge_index["tokenizer"]
        self.emb_matrix = knowledge_index["emb_matrix"]
        self.knowledge_vecs = knowledge_index["knowledge_vecs"]

    # TODO: Will need to implement way to select best knowledge without access to convo_tag
    def get_top_n(self, query, convo_tag, n=1):
        knowledge = self.knowledge_vecs[convo_tag]
        fact = get_max_cosine_similarity(query, knowledge, self.emb_matrix, self.tokenizer)
        return fact
