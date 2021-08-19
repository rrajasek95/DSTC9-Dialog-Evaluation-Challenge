import argparse
import os
import pickle
from itertools import chain
from tqdm.auto import tqdm

import pandas as pd
from transformers import GPT2Tokenizer

from constants import SWBD_ADDITIONAL_TOKENS, SPECIAL_TOKENS
from utils import add_tokens_to_vocabulary


def prepare_input_from_example_and_response(tokenizer, example, response, data_prep_parameters, is_ground_truth=False):
    bos, eos, speaker1, speaker2, end, pad, eot = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    encoded_dialog_acts = tokenizer.encode([f"<{da}>" for da in example.response_dialog_acts])

    if example.knowledge:
        # Some knowledge sentences can greatly exceed the length of the input sequence
        # truncate upto a maximum length
        knowledge_tokens = tokenizer.encode(example.knowledge)
        truncated_knowledge = knowledge_tokens[:min(data_prep_parameters["max_knowledge_length"], len(knowledge_tokens))]
        encoded_knowledge = truncated_knowledge + tokenizer.convert_tokens_to_ids(["_fact"])
    else:
        encoded_knowledge = tokenizer.convert_tokens_to_ids(["_nofact"])

    conversation_segments = []

    for i, segment in enumerate(example.context):
        speaker_segment = tokenizer.encode(f"<speaker{i % 2 + 1}>" + segment)
        conversation_segments.append(speaker_segment + [eot])

    system_segment = tokenizer.encode(f"<speaker{len(example.context) % 2 + 1}>" + response)
    conversation_segments.append(system_segment + [eot])

    sequence = [[bos] + encoded_dialog_acts + encoded_knowledge] + conversation_segments + [[eos]]

    instance = {}
    instance["input_ids"] = list(chain.from_iterable(sequence))

    # Apply speaker embeddings to the input sequence
    instance["token_type_ids"] = [0 for _ in sequence[0]] + [speaker1 if i % 2 else speaker2 for (i, s) in
                                                             enumerate(sequence[1:-1]) for _ in s] + [0 for _ in
                                                                                                      sequence[-1]]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1

    # Generate the ground truth token labels for the system response
    # used for computing language modeling loss; does not calculate LM-loss for non-ground-truth responses
    if is_ground_truth:
        instance["lm_labels"] = [-100] * (sum(len(s) for s in sequence[:-2]) + 1) + sequence[-2][1:] + [-100]
    else:
        instance["lm_labels"] = [-100] * len(instance['input_ids'])

    return instance


def prepare_training_instance_from_example(tokenizer, example, data_prep_parameters):
    """
    Constructs ground-truth and negative samples from the provided example.

    The input can be succinctly represented as the following scheme:
        [INPUT] := <bos> [ACTION_PLAN] <speaker1> [RESPONSE] <speaker2> [RESPONSE] ... <speaker1/2> [RESPONSE] <eos>
        [ACTION_PLAN] := <da_1> <da_2> ... <da_n> FACT _fact
                       | <da_1> <da_2> ... <da_n> _nofact
        [RESPONSE] := RESPONSE <eot>

    :param example:
    :return:
    """

    training_instances = [
        prepare_input_from_example_and_response(tokenizer, example, negative_sample, data_prep_parameters, is_ground_truth=False)
        for negative_sample in example.negative_samples]
    training_instances.append(
        prepare_input_from_example_and_response(tokenizer, example, example.response, data_prep_parameters, is_ground_truth=True))

    return training_instances


def prepare_training_data(training_features_path, pretrained_model_checkpoint, path_to_output_tokenizer,
                          path_to_output_train_data, data_prep_parameters):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_checkpoint)

    add_tokens_to_vocabulary(tokenizer, SWBD_ADDITIONAL_TOKENS)

    tokenizer.save_pretrained(path_to_output_tokenizer)

    training_features = pd.read_parquet(training_features_path)

    training_instances = []
    for example in tqdm(training_features.itertuples(name='Example')):
        training_instances.append(prepare_training_instance_from_example(tokenizer, example, data_prep_parameters))

    # Make parent directory
    os.makedirs(os.path.dirname(path_to_output_train_data), exist_ok=True)
    # Save data
    with open(path_to_output_train_data, 'wb') as training_data_file:
        pickle.dump(training_instances, training_data_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_features_path', default='data/processed/swbd_pd_nrg/features/train.parquet',
                        help='Path to parquet file containing training features')
    parser.add_argument('--pretrained_model_checkpoint', default='gpt2-medium')
    parser.add_argument('--path_to_output_tokenizer', default='data/processed/swbd_pd_nrg/tokenizer')
    parser.add_argument('--path_to_output_train_data', default='data/processed/swbd_pd_nrg/training_data.pkl')
    parser.add_argument('--max_knowledge_length', default=200, help='Maximum number of knowledge tokens to include')

    args = parser.parse_args()

    data_prep_parameters = {
        "max_knowledge_length": args.max_knowledge_length
    }
    prepare_training_data(args.training_features_path,
                          args.pretrained_model_checkpoint,
                          args.path_to_output_tokenizer,
                          args.path_to_output_train_data,
                          data_prep_parameters)
