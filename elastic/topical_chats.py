import csv
import json

import requests
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Document

"""
Adapted from Jiaqi's code from cobot
"""


class TopicalChatsIndex(Document):
    @classmethod
    def exists(cls, using):
        return cls._index.exists(using=using)


class TopicalChatsIndexRetriever:

    def __init__(self,
                 host="localhost",
                 port=9200,
                 alias="default"):
        self.host = host
        self.port = port
        self.alias = alias

        self.es = Elasticsearch(host=self.host, port=self.port)

        self.identifier = "topical_chats"
        self.doctype = "doc"

    def create_index(self):
        if TopicalChatsDocument.exists(using=self.alias):
            return False

        body = TopicalChatsDocument._index.to_dict()
        body["mappings"] = {self.doctype: body["mappings"]}
        self.create_request(json.dumps(body))
        return True

    def create_request(self, mappings, timeout=1):
        if not mappings:
            return False

        headers = {
            "Content-Type": 'application/json'
        }

        url = f"http://{self.host}:{self.port}/{self.identifier}"
        requests.put(url, data=mappings, headers=headers, timeout=timeout)

    def save_doc(self, data_dict):
        self.es.index(index=self.identifier, doc_type=self.doctype, body=data_dict)

    def add_documents(self, infilename):
        with open(infilename, "r") as infile:
            reader = csv.DictReader(infile)

            for row in reader:
                data_dict = {}
                data_dict.update(row)
                self.save_doc(data_dict)