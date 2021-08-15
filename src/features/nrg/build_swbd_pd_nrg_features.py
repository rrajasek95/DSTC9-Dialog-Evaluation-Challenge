import argparse
import os
from collections import namedtuple
from tqdm.auto import tqdm

import pandas as pd
from transformers import AutoTokenizer

from utils import add_tokens_to_vocabulary

from constants import SWBD_ADDITIONAL_TOKENS

TOPICAL_CHAT_SPLITS = (
    "train",
    "valid_freq",
    "valid_rare",
    "test_freq",
    "test_rare"
)

Turn = namedtuple('Turn', ['context', 'response', 'knowledge', 'response_dialog_acts', 'negative_samples'])


def extract_conversation_examples(conversation,
                                  all_messages,
                                  max_context_size,
                                  num_negative_samples,
                                  knowledge_similarity_threshold
                                  ):
    conversation_examples = []

    context = []
    previous_response = None

    for turn in conversation:
        if previous_response:
            context = context[-max_context_size + 1:] + [previous_response]
        negative_sample_responses = all_messages.sample(num_negative_samples).tolist()
        knowledge = "" if turn.knowledge_similarity_score < knowledge_similarity_threshold else turn.knowledge

        example = Turn(context, turn.message, knowledge, turn.segment_predictions, negative_sample_responses)
        conversation_examples.append(example)

        previous_response = turn.message

    return conversation_examples


def generate_examples_from_conversations(conversation_dataframe, processing_parameters):
    max_context_size = processing_parameters["max_context_size"]
    num_negative_samples = processing_parameters["num_negative_samples"]
    knowledge_similarity_threshold = processing_parameters["knowledge_similarity_threshold"]

    # Ensure conversations are in order prior to processing
    conversation_dataframe.sort_values(by=["conversation_id", "turn_index"])

    # Group conversations together
    conversations = []
    current_conversation_id = conversation_dataframe.iloc[0]['conversation_id']
    conversation_turns = []
    for turn in conversation_dataframe.itertuples(name="Turn"):
        if turn.conversation_id != current_conversation_id:
            conversations.append(conversation_turns)
            conversation_turns = []
            current_conversation_id = turn.conversation_id
        else:
            conversation_turns.append(turn)
    conversations.append(conversation_turns)

    examples = []
    for conversation in tqdm(conversations):
        examples.extend(
            extract_conversation_examples(conversation,
                                          conversation_dataframe['message'],
                                          max_context_size,
                                          num_negative_samples,
                                          knowledge_similarity_threshold))

    # Eager cleanup
    del conversations

    return pd.DataFrame(examples)


def prepare_swbd_pd_nrg_training_features(conversations_path, output_path, processing_parameters):
    os.makedirs(output_path, exist_ok=True)
    for split in TOPICAL_CHAT_SPLITS:
        conversation_split_dataframe = pd.read_parquet(os.path.join(conversations_path, f"{split}.parquet"))
        examples_dataframe = generate_examples_from_conversations(conversation_split_dataframe, processing_parameters)

        examples_dataframe.to_parquet(os.path.join(output_path, f"{split}.parquet"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--annotated_conversations_path',
                        default='data/intermediate/topical_chat/annotation/conversations',
                        help='Path to annotated conversations file')
    parser.add_argument('--output_path', default='data/processed/swbd_pd_nrg/features',
                        help='Path to emit the processed data to')

    parser.add_argument('--max_context_size', default=2,
                        help='Maximum of previous turns of context to condition on')
    parser.add_argument('--num_negative_samples', default=1,
                        help='Number of negative sample responses')
    parser.add_argument('--knowledge_similarity_threshold', default=0.7,
                        help='Similarity threshold for inclusion of knowledge')

    args = parser.parse_args()

    conversation_processing_parameters = {
        "max_context_size": args.max_context_size,
        "num_negative_samples": args.num_negative_samples,
        "knowledge_similarity_threshold": args.knowledge_similarity_threshold
    }

    prepare_swbd_pd_nrg_training_features(
        args.annotated_conversations_path,
        args.output_path,
        conversation_processing_parameters
    )