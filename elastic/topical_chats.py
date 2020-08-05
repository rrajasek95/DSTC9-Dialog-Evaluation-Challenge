import argparse
import csv

import requests
from elasticsearch_dsl import Document, Keyword, Text, connections, Q

"""
Adapted from Jiaqi's code from cobot
"""


class TopicalChatsFact(Document):
    source = Keyword()
    reddit_thread_id = Keyword()
    normalized_entity = Keyword()
    text = Text()

    class Index:
        name = 'topical_chats'

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

        self.connection = connections.create_connection(alias=alias,
                                                        hosts=[host], timeout=1)

        self.identifier = "topical_chats"
        self.doctype = "doc"

    def create_index(self):
        if TopicalChatsFact.exists(using=self.alias):
            return False

        TopicalChatsFact.init()
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
        fact = TopicalChatsFact(**data_dict)
        fact.save()

    def add_documents(self, data):
        if isinstance(data, str):
            with open(data, "r") as infile:
                reader = csv.DictReader(infile)

                for row in reader:
                    data_dict = {}
                    data_dict.update(row)
                    self.save_doc(data_dict)
        else:
            for doc in data:
                insert_doc = {}
                if isinstance(doc, str):
                    insert_doc["text"] = doc
                self.save_doc(insert_doc)

    def query_facts_from_entities(self, entities):
        q = None
        for ner_text in entities:
            ner_text = ner_text.lower()
            # Exact match for the phrases.
            if q:
                # https://www.elastic.co/guide/en/elasticsearch/reference/current/full-text-queries.html
                # should match
                q |= Q("match_phrase", reddit_thread_id=ner_text)
            else:
                q = Q("match_phrase", reddit_thread_id=ner_text)

        s = TopicalChatsFact.search(using=self.alias).query(q)
        response = s.execute()
        print(response)
        for hit in s:
            print(hit.text)

    def retrieve_facts_matching_utterance(self, utterance):
        q = Q("match", text=utterance)

        s = TopicalChatsFact.search(using=self.alias).query(q)
        response = s.execute()

        hits = [hit.text for hit in s]

        return hits

    def retrieve_facts_with_score(self, utterance):
        q = Q("match", text=utterance)

        s = TopicalChatsFact.search(using=self.alias).query(q)
        response = s.execute()
        hits = [(hit.text, hit.meta.score) for hit in s]
        print(hits)
        return hits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-host", "--host", help="host", type=str, default="localhost")
    parser.add_argument("-port", "--port", help="port", type=int, default=9200)
    parser.add_argument("-alias", "--alias", help="testing,default,product", type=str, default="default")

    args = parser.parse_args()

    tc_retriever = TopicalChatsIndexRetriever(args.host, args.port, args.alias)
    tc_retriever.create_index()
    # tc_retriever.add_documents('v2_entity_funfacts_texts.csv')
    # tc_retriever.query_facts_from_entities(['t3_3nhy9j'])
    tc_retriever.retrieve_facts_matching_utterance("jellyfish")