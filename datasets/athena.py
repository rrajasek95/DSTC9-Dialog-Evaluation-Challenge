from itertools import chain

from torch.utils.data import Dataset

class AthenaUserQuestionsDataset(Dataset):

    def __init__(self, data, tokenizer, special_tokens, inference=True):
        self.data = data
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.inference = inference

    def __getitem__(self, index):
        history, response = self.data[index]

        if self.inference:
            instance = self.build_input_from_segments(history, [])
        else:
            instance = self.build_input_from_segments(history, response)

        return [instance]

    def build_input_from_segments(self, history, response, lm_labels=False):
        """
        Input construction:
        <bos> _nofact <speaker1/2> UTT1 <speaker1/2> ... <speaker2> RESPONSE <eos>
        """

        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids((self.special_tokens[:4]))

        # There is no fact
        sequence = [[bos]] + history + [response + [eos]]

        sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                    enumerate(sequence[1:])]

        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        instance["lm_labels"] = [-100] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

        return instance

    def __len__(self):
        return len(self.data)

    def get_num_batches(self, batch_size):
        return len(self) // batch_size