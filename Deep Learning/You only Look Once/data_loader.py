from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torchvision.datasets as dset
from PIL import Image
import torch
import os
import random

class VOC(data.Dataset):
    def __init__(self, train_image_dir, test_image_dir, transform, mode):
        self.train_image_dir = train_image_dir
        self.test_image_dir = test_image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.train_dataset = dset.VOCDetection(root = self.train_image_dir, year = '2012', image_set='train', download=False)
        self.test_dataset = dset.VOCDetection(root = self.test_image_dir, year = '2012', image_set='val', download=False)
        print(self.train_dataset[0])
        if self.mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        labels = [label]
        return self.transform(filename), torch.FloatTensor(labels)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(train_image_dir='pascalData/train', test_image_dir='pascalData/val', image_size=448, batch_size=16, dataset='VOC', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    
    dataset = VOC(train_image_dir, test_image_dir, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
