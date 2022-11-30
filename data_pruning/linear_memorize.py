import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import math
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR
import pickle
from PIL import Image
import numpy as np
import copy

import shutil
from collections import OrderedDict
from helper_functions import OptimScheduler
from split_dataset import CreateDataLoaderMemorize
from memorize_dataloader import MemorizationDataloader


def str2bool(v):
    
    """ 
    Function to convert input string from argparse to bool
    
    input: v -> string
    output: v -> bool
    """
    
    if isinstance(v, bool):
        return v
    elif v.lower() == 'true':
        return True
    else:
        return False


class LinearModel(nn.Module):

    def __init__(self, args, output_size):
        super().__init__()

        if 'cifar' in args.dataset:

            self.model = nn.Sequential(OrderedDict([
                ('linear0', nn.Linear(in_features=3072, out_features=1024)),
                ('relu0', nn.ReLU(inplace=True)),
                ('linear1', nn.Linear(in_features=1024, out_features=512)),
                ('relu1', nn.ReLU(inplace=True)),
                ('linear2', nn.Linear(in_features=512, out_features=256)),
                ('relu2', nn.ReLU(inplace=True)),
                ('linear3', nn.Linear(in_features=256, out_features=128)),
                ('relu3', nn.ReLU(inplace=True)),
                ('linear4', nn.Linear(in_features=128, out_features=output_size)),
            ]))

        else:
            raise NotImplementedError("Oracle not implemented for current dataset")


    def forward(self, x):

        x = x.view(x.shape[0], -1)
        x = self.model(x)
        return x




def CreateDataLoader(args, pos_memorize, training_dataset_all):

    '''
    Create the dataloader
    '''

    if args.dataset == "cifar10":
        mean_dataset_norm = [0.4914, 0.4822, 0.4465]
        std_dataset_norm = [0.247, 0.2434, 0.2615]

    if args.dataset == "cifar100":
        mean_dataset_norm = [0.5071, 0.4867, 0.4408]
        std_dataset_norm = [0.2675, 0.2565, 0.2761]

    # if args.dataset == "imagenet":
    #     mean_dataset_norm = [0.485, 0.456, 0.406]
    #     std_dataset_norm = [0.229, 0.224, 0.225]

    # # must recompute imagenet values for now
    # if args.dataset == "tiny_imagenet":
    #     mean_dataset_norm = [0.485, 0.456, 0.406]
    #     std_dataset_norm = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                                # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)
                            ])

    test_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)
                            ])

    train_dataset = MemorizationDataloader(training_dataset_all[pos_memorize], m, train_transforms)

    if args.dataset == "cifar10":
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transforms)

    if args.dataset == "cifar100":
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=2000, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=2000, shuffle=False, num_workers=16)

    return train_dataloader, test_dataloader




def train():

    model.train()

    train_loss = 0
    train_acc = 0
    num_train_images = 0
    train_ce_loss = 0

    for idx, (images, labels) in enumerate(train_dataloader):
        
        optim.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        out = model(images)

        ce_loss = cross_entropy_loss(out, labels)
        pred = torch.argmax(out, dim=1)
        loss = ce_loss
        train_acc += torch.sum(torch.where(pred==labels, 1, 0))

        train_loss += loss.item() * images.shape[0]
        train_ce_loss += ce_loss.item() * images.shape[0]
        num_train_images += images.shape[0]

        loss.backward()
        optim.step()

    train_acc = train_acc/num_train_images
    train_loss = train_loss/num_train_images
    train_ce_loss = train_ce_loss/num_train_images

    return train_acc, train_loss, train_ce_loss




def test():

    test_loss = 0
    test_acc = 0
    num_test_images = 0
    test_ce_loss = 0

    for _,  (images, labels) in enumerate(test_dataloader):
    
        images = images.to(device)
        labels = labels.to(device)
        
        out = model(images)

        ce_loss = cross_entropy_loss(out, labels)
        pred = torch.argmax(out, dim=1)
        loss = ce_loss
        test_acc += torch.sum(torch.where(pred==labels, 1, 0))

        test_loss += loss.item() * images.shape[0]
        test_ce_loss += ce_loss.item() * images.shape[0]
        num_test_images += images.shape[0]

    test_acc = test_acc/num_test_images
    test_loss = test_loss/num_test_images
    test_ce_loss = test_ce_loss/num_test_images

    return test_acc, test_loss, test_ce_loss



def rank_images():

    if args.dataset == "cifar10":
        mean_dataset_norm = [0.4914, 0.4822, 0.4465]
        std_dataset_norm = [0.247, 0.2434, 0.2615]

    if args.dataset == "cifar100":
        mean_dataset_norm = [0.5071, 0.4867, 0.4408]
        std_dataset_norm = [0.2675, 0.2565, 0.2761]

    test_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)
                            ])

    in_class = {}
    out_class = {}

    for k in range(t):
        model.load_state_dict(torch.load('./weights/{}/memorization_oracle/model{}.pth'.format(args.dataset, k)))
        model.to(device)
        for class_idx in labeled_images.keys():

            if class_idx not in in_class.keys():
                in_class[class_idx] = {}
            if class_idx not in out_class.keys():
                out_class[class_idx] = {}


            for i in range(len(labeled_images[class_idx])):
                img = labeled_images[class_idx][i]
                img = np.reshape(img, (3, 32, 32)).astype(np.uint8)
                img = np.transpose(img, (1, 2 ,0))
                img = Image.fromarray(img)
                img = test_transforms(img).to(device).unsqueeze(0)
                out = model(img)
                pred = torch.argmax(out, dim=1)

                if i not in in_class[class_idx].keys():
                    in_class[class_idx][i] = [0, 0]
                if i not in out_class[class_idx].keys():
                    out_class[class_idx][i] = [0, 0]
                
                if i in images_in_dataset_all[k][class_idx]:
                    in_class[class_idx][i][1] += 1
                    if pred == class_idx:
                        in_class[class_idx][i][0] += 1

                else:
                    out_class[class_idx][i][1] += 1
                    if pred == class_idx:
                        out_class[class_idx][i][0] += 1

    prob = {}

    for class_idx in labeled_images.keys():
        if class_idx not in prob.keys():
            prob[class_idx] = {}
        for i in range(len(labeled_images[class_idx])):
            prob[class_idx][i] = np.abs(min(1, in_class[class_idx][i][0]/(1e-5+in_class[class_idx][i][1])) - min(1, out_class[class_idx][i][0]/(1e-5+out_class[class_idx][i][1])))

        ## easy examples first
        prob[class_idx] = list(dict(sorted(prob[class_idx].items(), key=lambda item: item[1])).keys()) #, reverse=True

        imgs = []
        for i in prob[class_idx]:
            imgs.append(labeled_images[class_idx][i])
        prob[class_idx] = imgs

    print(prob)

    with open('./data/{}_memorized_easy.pkl'.format(args.dataset), 'wb') as f:
        pickle.dump(prob, f)

    keys = list(prob.keys())
    prob_rev = copy.deepcopy(prob)

    for key in keys:
        prob_rev[key] = prob_rev[key][::-1]

    with open('./data/{}_memorized_hard.pkl'.format(args.dataset), 'wb') as f:
        pickle.dump(prob_rev, f)
    
    

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device", device)

    parser = argparse.ArgumentParser(description='Arguements for training Memorization Oracle Model')
    parser.add_argument('--dataset', default='cifar100')
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--early_stop', default=False, type=str2bool)
    args = parser.parse_args()

    ### Remove previous tfevents Create directory for weights
    if os.path.isdir('./runs/{}/memorization_oracle'.format(args.dataset)):
        shutil.rmtree('./runs/{}/memorization_oracle'.format(args.dataset))
        os.makedirs('./runs/{}/memorization_oracle'.format(args.dataset))
    if not os.path.isdir('./weights/{}/memorization_oracle'.format(args.dataset)):
        os.makedirs('./weights/{}/memorization_oracle'.format(args.dataset))

    writer = SummaryWriter('./runs/{}/memorization_oracle/'.format(args.dataset))

    # Set output size depending on the dataset and create dataloader
    if args.dataset == 'cifar10':
        output_size = 10
    elif args.dataset == 'cifar100':
        output_size = 100 
    elif args.dataset == 'imagenet':
        output_size = 1000
    elif args.dataset == 'tiny_imagenet':
        output_size = 200
    else:
        raise ValueError("Dataset not availabe")

    model = LinearModel(args, output_size).to(device)

    training_dataset_all, labeled_images, images_in_dataset_all, m, t = CreateDataLoaderMemorize(args.dataset)

    cross_entropy_loss = nn.CrossEntropyLoss()
    test_loss_max = 1e9
    prev_epoch = 0
    

    for key, value in model.named_parameters():
        print(key, value.requires_grad)


    for k in range(t):
        train_dataloader, test_dataloader = CreateDataLoader(args, k, training_dataset_all)
        optim_scheduler = OptimScheduler(args, 1)
        optim_scheduler.change = False
        optim = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)
        lr_scheduler = LinearLR(optim, start_factor=0.05, end_factor=1, total_iters=args.warmup_epochs)
        test_loss_max = 1e9
        prev_epoch = 0

        for epoch in range (args.n_epochs):

            train_acc, train_loss, train_ce_loss = train()
                
            with torch.no_grad():
                model.eval()
                test_acc, test_loss, test_ce_loss = test()
                model.train()

            if test_loss < test_loss_max:

                test_loss_max = test_loss
                torch.save(model.state_dict(), './weights/{}/memorization_oracle/model{}.pth'.format(args.dataset, k))

                # Because of warmup
                torch.save({'train_loss': train_loss,
                            'train_ce_loss': train_ce_loss,
                            'train_acc': train_acc, 
                            'test_loss': test_loss,
                            'test_ce_loss': test_ce_loss,
                            'test_acc': test_acc,
                            'epoch': epoch}, './weights/{}/memorization_oracle/info{}.pth'.format(args.dataset, k)) #'lr': lr_scheduler.get_last_lr()[0]

            print("Epoch: {}/{}, model: memorization_oracle{}, lr:{}".format(epoch, args.n_epochs, k, optim.param_groups[0]['lr']))

            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar('Train Cross Entropy Loss', train_ce_loss, epoch)
            writer.add_scalar('Train Acc', train_acc, epoch)
            writer.add_scalar('Val Loss', test_loss, epoch)
            writer.add_scalar('Val Cross Entropy Loss', test_ce_loss, epoch)
            writer.add_scalar('Val Acc', test_acc, epoch)

            lr_scheduler.step()

            loss_to_use = train_loss
            if epoch-prev_epoch>args.warmup_epochs:
                optim_scheduler.update(loss_to_use, optim)

            if optim_scheduler.pos == 1:
                break

    rank_images()