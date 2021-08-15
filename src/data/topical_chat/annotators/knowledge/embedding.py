import pickle
import string

import torch
from sentence_transformers import SentenceTransformer, util


class SentenceTransformerRetriever:
    def __init__(self,
                 knowledge_index_path,
                 device="cpu"
                 ):

        model_name, conversation_index = self._load_index(knowledge_index_path)

        self.embedder = SentenceTransformer(model_name)
        self.conversation_index = conversation_index
        self.device = device


    def _load_index(self, knowledge_embeddings_path):
        with open(knowledge_embeddings_path, "rb") as knowledge_embeddings_file:
            knowledge_embedding_data = pickle.load(knowledge_embeddings_file)

        return knowledge_embedding_data["model_name"], knowledge_embedding_data["conversation_index"]


    def _preprocess_query(self, query_text):
        return ''.join([c if c not in string.punctuation else ' ' for c in query_text.lower()])

    def query(self, query_text, conversation_id):
        preprocessed_query = self._preprocess_query(query_text)
        query_embedding = self.embedder.encode(preprocessed_query, convert_to_tensor=True, device=self.device)

        corpus = self.conversation_index[conversation_id]["corpus"]
        corpus_embeddings = self.conversation_index[conversation_id]["embeddings"].to(self.device)

        scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)

        best_match = torch.topk(scores, 1)
        score = float(best_match[0])
        index = best_match[1]

        return corpus[index], score
