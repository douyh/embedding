import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
class dataLoader(Dataset):
    def __init__(self, path, listName, dataset = '', data_transforms = None, loader = None):
        self.path = path
        self.listName = listName
        self.images = [os.path.join(self.path, line.strip().split()[0]) for line in open(self.listName)]
        self.labels = [int(line.strip().split()[1]) for line in open(self.listName)]
        self.data_transforms = data_transforms
        self.dataset = dataset
        if loader:
            self.loader = loader
        else:
            self.loader = self.default_loader
    def default_loader(self, imageName):
        try:
            image = Image.open(imageName)
            return image.convert('RGB')
        except:
            print("Cannot read image", path)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        imageName = self.images[item]
        label = self.labels[item]
        image = self.loader(imageName)
        if self.data_transforms is not None:
            try:
                image = self.data_transforms[self.dataset](image)
            except:
                print("Cannot transform image", imageName)
        return image, label

class dataAugmentation():
    def __init__(self):
        self.data_transforms = {
            "trainImages": transforms.Compose([ 
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
            "testImages": transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
        }
