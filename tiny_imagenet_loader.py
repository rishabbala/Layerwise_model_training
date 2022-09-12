import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import pickle
from PIL import Image
import numpy as np
import csv



class TinyImagenetDataset(Dataset):


    def __init__(self, split='train', transforms=None):
        
        self.split = split
        self.transforms = transforms

        self.id2onehot = {}
        self.onehot2id = {}

        class_counter = 0
        with open('./data/tiny-imagenet-200/wnids.txt') as data_file:
            for line in data_file:
                self.id2onehot[line.rstrip()] = class_counter
                self.onehot2id[class_counter] = line.rstrip()
                class_counter += 1

        if self.split == 'val':
            
            self.val_file = './data/tiny-imagenet-200/val/val_annotations.txt'
            self.val_data = []

            with open('./data/tiny-imagenet-200/val/val_annotations.txt') as data_file:
                self.val_data = list(csv.reader(data_file, delimiter="\t"))



    def __len__(self):
        if self.split == 'train':
            return 100000
        elif self.split == 'val':
            return 10000



    def __getitem__(self, idx):

        if self.split == 'train':

            class_idx_onehot = int(idx/500)
            class_idx = self.onehot2id[class_idx_onehot]
            img_idx = int(idx%500)

            image = Image.open('./data/tiny-imagenet-200/train/' + str(class_idx) + '/images/' + str(class_idx) + '_' + str(img_idx) + '.JPEG').convert('RGB')
            
            image = self.transforms(image)

            if image.shape[0] != 3:
                image = torch.cat((image, image, image), dim=0)

            label = class_idx_onehot


        elif self.split == 'val':

            img_idx = self.val_data[idx][0]
            image = Image.open('./data/tiny-imagenet-200/val/images/' + img_idx).convert('RGB')
            image = self.transforms(image)

            label = self.val_data[idx][1]
            label = self.id2onehot[label]

        # torchvision.utils.save_image(image, './{}.png'.format(idx))
        
        return image, label