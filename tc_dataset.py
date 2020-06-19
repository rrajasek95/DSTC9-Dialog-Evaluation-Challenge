from itertools import chain
import random

from torch.utils.data import Dataset

class TopicalChatsDataset(Dataset):
    """
    It's absolutely necessary to create a dataset class since
    the amount of data is huge.

    I wonder if there are other optimization opportunities
    - Rishi
    """
    def __init__(self, dataset, tokenizer, special_tokens, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.max_history = args.max_history
        self.num_candidates = args.num_candidates

    def __getitem__(self, index):
        (history, response, fact) = self.dataset[index]

        # Truncate history turns to reduce memory requirement
        if len(history) > (2 * self.max_history + 1):
            history = history[-(2 * self.max_history + 1):]

        candidates = self.sample_candidates(self.dataset, index)
        candidates.append(response)

        instances = []
        for j, candidate in enumerate(candidates):
            lm_labels = bool(j == self.num_candidates - 1)
            instance = self.build_input_from_segments(history, candidate, fact, self.tokenizer, lm_labels)
            instances.append(instance)
        return instances

    def __len__(self):
        return len(self.dataset)

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def sample_candidates(self, dataset, current_conversation_index):
        # Lets just hope that the number of cases where the true responses gets included in the
        # candidates is vanishingly small
        candidates = [response for (_, response, _) in random.sample(dataset, self.num_candidates - 1)]

        return candidates

    def build_input_from_segments(self, history, response, fact, tokenizer, lm_labels=False):
        bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids((self.special_tokens[:-1]))

        """
        Input construction (may change):
        <bos> FACT <speaker1/2> UTT1 <speaker1/2> ... <speaker2> RESPONSE
        Considerations for design:
        1. Topical chat examples are created by adding a response every turn
        2. Last turn is always speaker2

        To my knowledge, the position of the fact in input is mostly immaterial due to
        the self-attention mechanism (since all tokens are equidistant). The positional
        embeddings affect only the contextual representation (I think!)
          - Rishi
        """
        sequence = [[bos] + fact] + history + [response + [eos]]

        sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                    enumerate(sequence[1:])]

        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        # I have no idea what this part refers to, Cargo Culting for now
        instance["lm_labels"] = [-100] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        return instance
