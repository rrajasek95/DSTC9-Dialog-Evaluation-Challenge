"""
 Created by diesel
 10/9/20
"""


import pickle
from itertools import chain
import random

import spacy
from torch.utils.data import Dataset
import torch
from collections import defaultdict



class DatasetBase(Dataset):

    def __init__(self, dataset, tokenizer, special_tokens, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens

        # Args to control memory footprint
        self.max_history = args.max_history
        self.num_candidates = args.num_candidates
        self.max_fact_length = args.max_fact_length

    def __getitem__(self, index):
        """
        Baseline sentence data format.

        Each example comprises of the following:
        1. history_tuple:
            1. conversation_history - List[List[int]]
                1. Highest list level corresponds to turns in the conversation
                2. Lowest list level are the individual tokens in the segment
                Example:
            2. conversation_history_da - (TODO: fill type)
                1. dialog acts of conversation history - not relevant to baseline config
            3. knowledge history - (TODO: fill type)
                1. knowledge sentences corresponding to conv history - not relevant to baseline config

        2. target_tuple:
            1. response: List[int] - tokens of the expected response which is a single turn
            2. DA_info - not relevant to baseline config
            3. fact: List[int] - tokens of knowledge sentence corresponding to the sentence we are generating

        :return: instance: Dict[str, object]
                    - "input_ids": the sequence of tokens of our prepared input
                    - "token_type_ids":
                        - tokens indicating which parts of input are 'sentence_plan', 'speaker1 response', 'speaker2 response'
                    - "mc_token_ids":
                        - tokens indicating whether the response is a true follow-on to the context (multiple choice selection)
                    - "lm_labels":
                        - tokens which indicate which parts of the sequence represent the predicted output (for language modeling)
        """
        # For the baseline implementation, we don't need to consider the DA
        item = self.dataset[index]

        history = item["src"]
        response = item["tgt"]
        fact = item["fct"]


        history, fact = self.truncate_sequences(history, fact)

        candidates = self.sample_candidates(self.dataset, index)
        candidates.append(response)

        instances = []
        for j, candidate in enumerate(candidates):
            lm_labels = bool(j == self.num_candidates - 1)
            #instance = self.build_input_from_segments(history, candidate, fact, self.tokenizer, lm_labels)

            bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids((self.special_tokens[:4]))
            sequence = (
                # fact
                    [[bos] + fact] +
                    # history
                    history +
                    # response
                    [candidate + [eos]]
            )

            # add speaker token to beginning of turns
            sequence = [sequence[0]] + [
                [speaker2 if (len(sequence) - i) % 2 else speaker1] + s
                for i, s in enumerate(sequence[1:])
            ]

            instance = {}
            instance["input_ids"] = list(chain(*sequence))
            instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
            instance["mc_token_ids"] = len(instance["input_ids"]) - 1
            instance["lm_labels"] = [-100] * len(instance["input_ids"])
            if lm_labels:
                instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]


            instances.append(instance)
        return instances


    def __len__(self):
        return len(self.dataset)


    def get_num_batches(self, batch_size):
        return len(self) // batch_size


    def sample_candidates(self, dataset, current_conversation_index):
        # Lets just hope that the number of cases where the true responses gets included in the
        # candidates is vanishingly small
        candidates = [ex["tgt"] for ex in random.sample(dataset, self.num_candidates - 1)]
        return candidates

    def build_input_from_segments(self, history, response, fact, tokenizer, lm_labels=False):
        """
        Input construction (may change):
        <bos> FACT <speaker1/2> UTT1 <speaker1/2> ... <speaker2> RESPONSE <eos>
        Considerations for design:
        1. Topical chat examples are created by adding a response every turn
        2. Last turn is always speaker2

        Reference:
        https://huggingface.co/transformers/model_doc/gpt2.html?highlight=gpt2#transformers.GPT2DoubleHeadsModel
        https://huggingface.co/transformers/model_doc/gpt2.html?highlight=gpt2#transformers.GPT2LMHeadModel
        """

        bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids((self.special_tokens[:4]))


        sequence = [[bos] + fact] + history + [response + [eos]]

        sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                    enumerate(sequence[1:])]

        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        """
        Explanation:
        lm_labels is token-wise mask that is used to compute language modeling loss 
        We want the language modeling loss to propagate only when we generate
        incorrectly on the true response and not on the distractor responses
        """
        instance["lm_labels"] = [-100] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        return instance


    def truncate_sequences(self, history, fact):
        # Truncate history turns to reduce memory requirement
        if len(history) > (2 * self.max_history + 1):
            history = history[-(2 * self.max_history + 1):]

        # Truncate facts to decrease overall input length
        trunc_facts = fact[:min(len(fact), self.max_fact_length)]
        return history, trunc_facts


    @staticmethod
    def pad_batch_input(dataset, padding=0):
        """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
        PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
        max_l = max(len(x) for x in dataset["input_ids"])
        for name in PADDED_INPUTS:
            dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
        return dataset


    @classmethod
    def collate_batch_elements(cls, batch, tokenizer, args, special_tokens):
        """
        Topical chats is a ridiculously large dataset (2GB+ including facts/reading set).
        Maintaining an entire tensor dataset in memory is a terrible idea
        since *every* input is padded to the size of the largest element.
        The training dataset has 179k instances, so imagine padding all
        of those to max_length (RIP RAM!)

        Another point to note is that the actual number of instances per batch in this
        implementation is num_candidates*batch_size. I haven't thought about how to
        optimize this but I guess it'll take a bit more effort
        - Rishi

        """
        batch_inputs = defaultdict(list)
        chained_batch = chain(*batch)

        for instance in chained_batch:
            for field, data in instance.items():
                batch_inputs[field].append(data)

        padded_dataset = cls.pad_batch_input(batch_inputs, padding=tokenizer.convert_tokens_to_ids(special_tokens[-2]))

        tensorized_input = []
        # Verify input sent the same way:
        #
        # "input_ids": [Batch size, num_cands, seq_len]
        # "mc_token_ids": [Batch size, num cands],
        # "lm_labels": [batch size, num_cands, seq_len]
        # "mc_labels": [batch_size]
        # "token_type_ids": [batch_size, num_cands, seq_len]

        MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]

        batch_size = tuple([len(batch_inputs[MODEL_INPUTS[0]])//args.num_candidates])
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(padded_dataset[input_name])

            if input_name != "mc_labels":
                tensor = tensor.view((-1, args.num_candidates) + tensor.shape[1:])
            else:
                tensor = torch.ones(size=batch_size, dtype=torch.long) * (args.num_candidates - 1)
            tensorized_input.append(tensor)
        return tensorized_input



def main():
    pass


if __name__ == "__main__":
    main()
