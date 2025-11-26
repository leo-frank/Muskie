import torch
from torch.utils.data import Dataset
from abc import ABC
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset

class ComposedDataset(Dataset, ABC):
    def __init__(self, dataset_configs):
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict)
            base_dataset_list.append(baseset)
        self.base_dataset = ConcatDataset(base_dataset_list)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx):
        batch = self.base_dataset[idx]
        return batch