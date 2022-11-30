import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import pickle
from PIL import Image
import numpy as np



class ImagenetDataset(Dataset):


    def __init__(self, split='train', transforms=None):
        
        self.split = split
        self.transforms = transforms

        if self.split == 'train':
            for i in range(10):
                self.data_file = './data/imagenet/train_data_batch_{}'.format(str(i+1))
                d = self.unpickle()
                if i == 0:
                    self.data = d['data']
                    self.labels = d['labels']
                else:
                    self.data = np.append(self.data, d['data'], axis=0)
                    self.labels.extend(d['labels'])
                

        if self.split == 'val':
            self.data_file = './data/imagenet/val_data'
            d = self.unpickle()
            self.data = d['data']
            self.labels = d['labels']


    def __len__(self):
        return len(self.labels)


    def unpickle(self):
        with open(self.data_file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def __getitem__(self, idx):

        x = self.data[idx]
        x = np.reshape(x, (3, 32, 32)).astype(np.uint8)
        x = np.transpose(x, (1, 2 ,0))
        x = Image.fromarray(x)

        y = torch.tensor(self.labels[idx]-1)

        if self.transforms != None:
            x = self.transforms(x)

        # torchvision.utils.save_image(x, './{}.png'.format(idx))
        return x, y