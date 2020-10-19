from collections import Counter
from itertools import chain

from torch.utils.data.dataset import Dataset
from torchtext.vocab import Vocab

class AthenaDaDataset(Dataset):
    def __init__(self, df):
        self.dataset = df
        counter = Counter(df['label'].tolist())
        self.vocab = Vocab(counter)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset.iloc[index]

        return row['text'], self.vocab.stoi[row['label']]

    def num_labels(self):
        return len(self.vocab)

class SwdaDataset(Dataset):
    def __init__(self, conversations):
        self.conversations = conversations
        all_utterances = chain.from_iterable([conv['utterances'] for conv in conversations])
        counter = Counter(turn['damsl_act_tag'] for turn in all_utterances)
        self.vocab = Vocab(counter)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, item):
        dialog = self.conversations[item]['utterances']
        utterances = []
        tags = []

        for turn in dialog:
            utterances.append(turn["text"])
            tags.append(self.vocab.stoi[turn["damsl_act_tag"]])

        return utterances, tags, len(tags)

    def num_labels(self):
        return len(self.vocab)
