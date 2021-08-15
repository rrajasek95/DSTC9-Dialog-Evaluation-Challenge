from tqdm.auto import tqdm

from .base_annotator import AnnotatorBase
from .knowledge.embedding import SentenceTransformerRetriever

class KnowledgeRetriever(AnnotatorBase):
    def __init__(self, retriever_index_path):
        self.retriever = SentenceTransformerRetriever(retriever_index_path)

    def annotate_df(self, messages_df):
        tqdm.pandas()

        print("Retrieving knowledge")
        knowledge_sentence_scores = messages_df.progress_apply(lambda row: self.retriever.query(row['message'], row['conversation_id']), axis=1)
        print("Retrieval completed!")

        return knowledge_sentence_scores