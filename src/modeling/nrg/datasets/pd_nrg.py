from torch.utils.data import Dataset


class PdNrgDataset(Dataset):
    """
    Implements the version of the PD-NRG dataset that works with pre-tokenized data
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]

        return example