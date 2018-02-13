import torch
from torch.utils import data


class SliceSet(data.Dataset) :

    def __init__(self, dataset, start_index, end_index) :
        self.start_index = start_index
        self.end_index = end_index
        self.dataset = dataset


    def __len__(self) :
        return self.end_index - self.start_index

    def __getitem__(self, index) :
        return self.dataset[index - self.start_index]

