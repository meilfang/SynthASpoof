from typing import Callable
import os
import os.path
from os.path import exists
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict
import random

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.distributed as dist
import torch.utils.data
import torchvision

import albumentations
from albumentations.pytorch import ToTensorV2

PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]

def ApplyWeightedRandomSampler(dataset_csv):
    dataframe = pd.read_csv(dataset_csv) # head: image_path, label
    class_counts = dataframe.label.value_counts()
    sample_weights = [1/class_counts[i] for i in dataframe.label.values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataframe), replacement=True)
    return sampler

# map_size is for PixBis
class TrainDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224), map_size=14):
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            albumentations.HorizontalFlip(),
            albumentations.RandomGamma(gamma_limit=(80, 180)), # 0.5, 1.5
            albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, p=0.5),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),
        ])
        self.map_size = map_size

    def __len__(self):
        return len(self.dataframe)

    def get_labels(self):
        return self.dataframe.iloc[:, 1]

    def __getitem__(self, idx):

        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if label_str == 'bonafide' else 0
        map_x = torch.ones((self.map_size,self.map_size)) if label == 1 else torch.zeros((self.map_size, self.map_size))

        image = self.composed_transformations(image = image)['image']

        return {
            "images": image,
            "labels": torch.tensor(label, dtype = torch.float),
            "map": map_x
        }

class TestDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224)):
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = 1 if label_str == 'bonafide' else 0

        image = self.composed_transformations(image=image)['image']

        return {
            "images": image,
            "labels": torch.tensor(label, dtype = torch.float),
            "img_path": img_path
        }
