from collections import Counter

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
