import argparse
import os
import string

import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm.auto import tqdm
import pickle

SPLITS = (
    "train",
    "valid_freq",
    "valid_rare",
    "test_freq",
    "test_rare"
)


def clean(s):
    return ''.join([c if c not in string.punctuation else ' ' for c in s.lower()])


def embed_knowledge(args):
    embedder = SentenceTransformer(args.model_name)

    os.makedirs(args.output_path, exist_ok=True)

    for split in SPLITS:
        split_reading_set_file = os.path.join(args.reading_sets_path, f"{split}.parquet")
        reading_set_dataframe = pd.read_parquet(split_reading_set_file)

        split_conversation_knowledge = defaultdict(list)

        for item in reading_set_dataframe.itertuples():
            if item.fact_type not in ["url", "entity"]:
                if item.list_data:
                    for data in item.list_data:
                        split_conversation_knowledge[item.conversation_id].append(data)
                else:
                    split_conversation_knowledge[item.conversation_id].append(item.data)

        split_knowledge_dict = {
            "model_name": args.model_name,
            "conversation_index": {}
        }

        for conversation_id, conversation_knowledge in tqdm(split_conversation_knowledge.items()):
            split_knowledge_dict["conversation_index"][conversation_id] = {
                "corpus": conversation_knowledge,
                "embeddings": embedder.encode(conversation_knowledge, convert_to_tensor=True)
            }

        output_split_knowledge_path = os.path.join(args.output_path, f"{split}_embeddings.pkl")

        with open(output_split_knowledge_path, "wb") as output_split_knowledge_file:
            pickle.dump(split_knowledge_dict, output_split_knowledge_file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='paraphrase-MiniLM-L6-v2')
    parser.add_argument('--reading_sets_path', default='data/intermediate/topical_chat/parquet/reading_sets')
    parser.add_argument('--output_path', default='data/intermediate/topical_chat/knowledge_embeddings/')

    args = parser.parse_args()

    embed_knowledge(args)