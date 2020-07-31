from elasticsearch import Elasticsearch

from elastic.topical_chats import TopicalChatsIndexRetriever

"""
Adapted from Jiaqi's code from cobot
"""


class ElasticSearchIndexRetriever:
    domain_specific_index_classes = [
        TopicalChatsIndexRetriever
    ]

    def __init__(self,
                 host="localhost",
                 port=9200,
                 alias="default"):
        self.host = host
        self.port = port

        self.initialize_indices()
        self.es = Elasticsearch(host=host, port=port)

    def initialize_indices(self):
        self.indices = {}

        for clazz in self.domain_specific_index_classes:
            self.add_index(index_clazz=clazz)

    def add_index(self, index_clazz):
        self.indices[index_clazz.identifier] = index_clazz(host=self.host, port=self.port)