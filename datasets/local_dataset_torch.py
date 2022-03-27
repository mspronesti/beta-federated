from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image as Image


class CifarLocalDataset(Dataset):
    def __init__(self, images, labels, num_classes, client_id=-1):
        self.images = images
        self.labels = labels.astype(int)
        self.num_classes = num_classes
        self.client_id = client_id
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index].reshape((32, 32, 3)), mode='RGB')
        img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)
