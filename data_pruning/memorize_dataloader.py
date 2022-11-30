import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import pickle
from PIL import Image
import numpy as np



class MemorizationDataloader(Dataset):


    def __init__(self, training_dataset, m, transforms):
        
        self.training_dataset = training_dataset
        self.m = m
        self.transforms = transforms

    def __len__(self):
        return self.m

    def __getitem__(self, idx):

        class_idx = int(idx/len(self.training_dataset[0]))
        img_idx = int(idx%len(self.training_dataset[0]))

        x = self.training_dataset[class_idx][img_idx]
        x = np.reshape(x, (3, 32, 32)).astype(np.uint8)
        x = np.transpose(x, (1, 2 ,0))
        x = Image.fromarray(x)

        y = torch.tensor(class_idx)

        if self.transforms != None:
            x = self.transforms(x)

        # torchvision.utils.save_image(x, './{}.png'.format(idx))
        return x, y