import os
import torch
from torch.utils import data
from PIL import Image
import torch.nn as nn
import torchvision
import numpy as np
import operator
from medmnist import *
import csv
import cv2
import torchvision.transforms as transforms
import torchvision.datasets as dset
# Med MNIST

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=1):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if(self.n_views==1):
            return self.base_transform(x.convert('RGB'))
        elif(self.n_views==2):
            return [self.base_transform(x.convert('RGB')) for i in range(self.n_views)]

class ContrastiveLearningDataset:
    def __init__(self, transform, n_views):
        self.transforms = transform
        self.n_views = n_views

    def get_dataset(self, name):
        # OCTMNIST 
        dataset_t = PathMNIST(split=name, transform=ContrastiveLearningViewGenerator(self.transforms,self.n_views),download=True)
        return dataset_t


class IndexedDataset(data.Dataset):
    """Wrap a dataset so each item also returns its stable sample index."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        return sample, target, index

    def __len__(self):
        return len(self.dataset)


def build_imagefolder_dataset(root, transform, n_views):
    """Build an ImageFolder dataset with one or two augmented views.

    Args:
        root: directory in ImageFolder format, e.g. root/class_name/*.jpg.
        transform: torchvision transform applied to each PIL image.
        n_views: 1 for test view, 2 for teacher/student train views.

    Returns:
        torchvision.datasets.ImageFolder.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            "ImageFolder path not found: {}\n"
            "Expected structure: {}/<class_name>/*.jpg\n"
            "Use --data_root, --train_dir, or --test_dir to set the correct path.".format(root, root)
        )
    return dset.ImageFolder(
        root=root,
        transform=ContrastiveLearningViewGenerator(transform, n_views=n_views),
    )

