import argparse
import os

import pandas as pd
from tqdm.auto import tqdm

from utilities import load_json_from_path

TOPICAL_CHAT_SPLITS = (
    "train",
    "valid_freq",
    "valid_rare",
    "test_freq",
    "test_rare"
)


def convert_tc_conversations_to_parquet(conversations_path, output_base_path, splits=TOPICAL_CHAT_SPLITS):

    output_conversations_path = os.path.join(output_base_path, "conversations")
    output_metadata_path = os.path.join(output_conversations_path, "meta")

    os.makedirs(output_conversations_path, exist_ok=True)
    os.makedirs(output_metadata_path, exist_ok=True)

    for split in splits:
        split_conversation_path = os.path.join(conversations_path, f"{split}.json")

        split_conversations = load_json_from_path(split_conversation_path)

        # Typically, we don't need to worry too much about the conversation metadata,
        # except using it as an auxiliary field to filter on for our experiments.
        # Usually, this is ok to ignore
        conversations_metadata = []

        messages = []

        for conversation_id, conversation_data in tqdm(split_conversations.items()):
            conversation_metadata = {
                "conversation_id": conversation_id,
                "article_url": conversation_data['article_url'],
                "config": conversation_data['config'],
                "agent_1_conversation_rating": conversation_data['conversation_rating']['agent_1'],
                "agent_2_conversation_rating": conversation_data['conversation_rating']['agent_2']
            }

            conversations_metadata.append(conversation_metadata)

            for turn_index, message in enumerate(conversation_data['content']):
                conversation_message = {
                    "conversation_id": conversation_id,
                    "turn_index": (turn_index + 1)  # We use 1-based indexing
                }
                # Retain the existing fields as-is
                conversation_message.update(message)

                messages.append(conversation_message)

        split_metadata_path = os.path.join(output_metadata_path, f"{split}_meta.parquet")
        split_messages_path = os.path.join(output_conversations_path, f"{split}.parquet")

        conversation_metadata_dataframe = pd.DataFrame.from_records(conversations_metadata)
        conversation_metadata_dataframe.to_parquet(split_metadata_path)

        conversation_messages_dataframe = pd.DataFrame.from_records(messages)

        conversation_messages_dataframe.set_index(["conversation_id", "turn_index"])
        conversation_messages_dataframe.to_parquet(split_messages_path)

        print(f"Successfully saved data for split '{split}'")

    print("Successfully saved all conversation data!")


def create_fact_set_entry(conversation_id, agent, fact_set, fact_type, fact_data):


    return {
        "conversation_id": conversation_id,
        "agent": agent,
        "fact_set_id": fact_set,
        "fact_type": fact_type,
        "data": fact_data if isinstance(fact_data, str) else None,
        "list_data": fact_data if isinstance(fact_data, list) else None  # Parquet needs data types to be homogeneous
    }

def extract_conversation_reading_set_items(conversation_id, conversation_reading_set):



    conversation_items = []

    # Extracting agent items
    for agent in ["agent_1", "agent_2"]:
        agent_reading_set = conversation_reading_set[agent]

        for fact_set_id, fact_set in agent_reading_set.items():
            for fact_type, fact_data in fact_set.items():
                conversation_items.append(
                    create_fact_set_entry(conversation_id, agent, fact_set_id, fact_type, fact_data))

    article_data = conversation_reading_set["article"]

    for item_type, data in article_data.items():
        conversation_items.append(create_fact_set_entry(conversation_id, "both", "article", item_type, data))

    return conversation_items






def convert_tc_reading_sets_to_parquet(reading_sets_path, output_base_path, splits=TOPICAL_CHAT_SPLITS):
    output_reading_set_paths = os.path.join(output_base_path, "reading_sets")
    output_metadata_path = os.path.join(output_reading_set_paths, "meta")

    os.makedirs(output_reading_set_paths, exist_ok=True)
    os.makedirs(output_metadata_path, exist_ok=True)

    for split in splits:
        split_reading_set_path = os.path.join(reading_sets_path, "post-build", f"{split}.json")

        split_reading_set = load_json_from_path(split_reading_set_path)

        reading_sets_metadata = []

        reading_set_items = []

        for conversation_id, conversation_reading_set in tqdm(split_reading_set.items()):

            reading_set_metadata = {
                "conversation_id": conversation_id,
                "config": conversation_reading_set['config']
            }

            reading_set_items.extend(extract_conversation_reading_set_items(conversation_id, conversation_reading_set))
            reading_sets_metadata.append(reading_set_metadata)

        reading_set_dataframe = pd.DataFrame.from_records(reading_set_items)
        reading_set_metadata_dataframe = pd.DataFrame.from_records(reading_sets_metadata)

        reading_set_dataframe.to_parquet(os.path.join(output_reading_set_paths, f"{split}.parquet"))
        reading_set_metadata_dataframe.to_parquet(os.path.join(output_metadata_path, f"{split}.parquet"))

        print(f"Successfully saved data for split '{split}'")


def prepare_topical_chats_parquet(conversations_path, reading_sets_path, output_path):
    convert_tc_conversations_to_parquet(args.conversations_path, args.output_path)
    convert_tc_reading_sets_to_parquet(args.reading_sets_path, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--conversations_path', default="data/external/alexa-prize-topical-chat-dataset/conversations",
                        help="Path to Topical Chat conversations")
    parser.add_argument('--reading_sets_path', default="data/external/alexa-prize-topical-chat-dataset/reading_sets",
                        help="Path to Topical Chat reading sets")
    parser.add_argument('--output_path', default="data/intermediate/topical_chat_parquet/",
                        help="Path to the processed output files"
                        )

    args = parser.parse_args()

    prepare_topical_chats_parquet(args.conversations_path, args.reading_sets_path, args.output_path)