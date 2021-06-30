from torch.utils.data import Dataset

class PdNrgDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        example = self.dataset[item]

        pass