import torch
import torchvision
from torch import nn
import os
import math
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR

import shutil
from collections import OrderedDict
import random
from helper_functions import OptimScheduler
from memorize_dataloader import MemorizationDataloader



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def CreateDataLoaderMemorize(dataset):
    
    if dataset == 'cifar10':
        data_dir = './data/cifar-10-batches-py/data_batch_'

        labeled_images = {} 
        num_images = 0
        num_classes = 10

        for i in range(5):
            d = unpickle(data_dir+str(i+1))
            for l in range(len(d[b'labels'])):
                num_images += 1
                if d[b'labels'][l] in labeled_images.keys():
                    labeled_images[d[b'labels'][l]].append(d[b'data'][l])
                else:
                    labeled_images[d[b'labels'][l]] = [d[b'data'][l]]

    else:
        data_dir = './data/cifar-100-python/train'

        labeled_images = {} 
        num_images = 0
        num_classes = 100

        d = unpickle(data_dir)

        for l in range(len(d[b'fine_labels'])):
            num_images += 1
            if d[b'fine_labels'][l] in labeled_images.keys():
                labeled_images[d[b'fine_labels'][l]].append(d[b'data'][l])
            else:
                labeled_images[d[b'fine_labels'][l]] = [d[b'data'][l]]

    for l in labeled_images.keys():
        print(l, len(labeled_images[l]))


    m = int(num_images/num_classes * 0.7)
    t = 200
    training_dataset_all = {}
    images_in_dataset_all = {}

    for k in range(t):
        print(k)
        single_training_dataset = {}
        single_images_in_dataset = {}
        for l in labeled_images.keys():
            r_sample = random.sample(range(0, len(labeled_images[l])), m)
            l_sample = [labeled_images[l][r] for r in r_sample]
            single_training_dataset[l] = l_sample
            single_images_in_dataset[l] = r_sample
        training_dataset_all[k] = single_training_dataset
        images_in_dataset_all[k] = single_images_in_dataset

    return training_dataset_all, labeled_images, images_in_dataset_all, m, t