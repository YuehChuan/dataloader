# -*- coding: utf-8 -*-
#data https://huggingface.co/sd-dreambooth-library/nasa-space-v2-768/tree/main/concept_images
#https://discuss.pytorch.org/t/how-to-set-labels-for-images-in-a-dataset/148367
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

class SpaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        return image

#https://minerl.io/docs/notes/windows.html
#https://blog.csdn.net/u014546828/article/details/109235539
if __name__ == '__main__':
    TRAIN_IMG_DIR = "data/train_images/"
    VAL_IMG_DIR = "data/val_images/"

    BATCH_SIZE = 5
    NUM_EPOCHS = 3
    NUM_WORKERS = 2
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_ds = SpaceDataset(
        image_dir=TRAIN_IMG_DIR,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader):
            # get the inputs; data is a tensor (batchsize,weight,height,channel) 5 images (768,768,3)
            inputs = data



