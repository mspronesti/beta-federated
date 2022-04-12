from torchvision import transforms
import PIL.Image as Image
from .local_dataset import LocalDataset


class CifarLocalDataset(LocalDataset):
    """This is the local dataset used
    for each client for the cifar dataset."""

    def __init__(self, images, labels, num_classes=-1, client_id=-1):
        """
        Args:
            images: the training images
            labels: the associated labels
            num_classes:the number of UNIQUE classes present in this client
            client_id: torch transform for the images
        """
        super().__init__(images, labels, num_classes=-1, client_id=-1)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index].reshape((32, 32, 3)), mode='RGB')
        img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)
