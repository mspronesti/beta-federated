from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class LocalDataset(ABC, Dataset):

    def __init__(self,
                 data,
                 labels,
                 num_classes,
                 client_id):
        """
        Args:
            data: the training data
            labels: the labels to train
            num_classes: the distinct classes present in the client
            client_id: the id associated with the client
        """
        self.data = data
        self.labels = labels
        self.num_classes = num_classes
        self.client_id = client_id

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass
