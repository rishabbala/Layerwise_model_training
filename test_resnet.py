import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, MultiStepLR
import argparse
import os
import math
from torchvision import transforms
import time

from helper_functions import CreateModelName, OptimScheduler
from conf import get_args
from imagenet_loader import ImagenetDataset
from tiny_imagenet_loader import TinyImagenetDataset
from create_model import ModelClass
from conloss import ContrastiveLoss
from resnet import MakeResnet
from resnet_blockwise import MakeResnetBlockwise


device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

if args.dataset == 'cifar10':
    output_size = 10
    mean_dataset_norm = [0.4914, 0.4822, 0.4465]
    std_dataset_norm = [0.247, 0.2434, 0.2615]
elif args.dataset == 'cifar100':
    output_size = 100 
    mean_dataset_norm = [0.5071, 0.4867, 0.4408]
    std_dataset_norm = [0.2675, 0.2565, 0.2761]
elif args.dataset == 'imagenet':
    output_size = 1000
    mean_dataset_norm = [0.485, 0.456, 0.406]
    std_dataset_norm = [0.229, 0.224, 0.225]
elif args.dataset == 'tiny_imagenet':
    output_size = 200
    mean_dataset_norm = [0.485, 0.456, 0.406]
    std_dataset_norm = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)
                        ])

test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)
                        ])

if args.dataset == "cifar10":
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transforms)

if args.dataset == "cifar100":
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transforms)

if args.dataset == "imagenet":
    train_dataset = ImagenetDataset(split='train', transforms=train_transforms)
    test_dataset = ImagenetDataset(split='val', transforms=test_transforms)

# must recompute imagenet values for now
if args.dataset == "tiny_imagenet":
    train_dataset = TinyImagenetDataset(split='train', transforms=train_transforms)
    test_dataset = TinyImagenetDataset(split='val', transforms=test_transforms)

args.batch_size = 256
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)


model_large = MakeResnet(num_linear_layers=2, output_size=output_size, dataset=args.dataset, block_size=[2, 2, 2, 2, 0])
model_init = MakeResnet(num_linear_layers=2, output_size=output_size, dataset=args.dataset, block_size=[2, 2, 2, 2, 0])
model_small = MakeResnet(num_linear_layers=2, output_size=output_size, dataset=args.dataset, block_size=[2, 2, 2, 2, 0])


new_sd = torch.load('./weights/cifar100/combined_resnet18/model.pth')
old_sd = torch.load('./weights/cifar100/combined_resnet18/model.pth')

model_large.load_state_dict(new_sd)
model_small.load_state_dict(old_sd)

# new_sd.update(old_sd)

model_init.load_state_dict(new_sd)


for key, values in model_init.named_parameters():
    txt = key.split('.')

    try:
        if int(txt[1][-1]) == 2 and int(txt[2]) == 1 and 'norm' in key:
            print(key)
            values.data = torch.zeros(values.data.shape)
    except:
        pass

    # print(key, torch.sqrt(torch.sum(torch.square(values))))

# for key, values in model_large.named_parameters():
#     print(key, torch.sqrt(torch.sum(torch.square(values))))

model_small.cuda()
model_small.eval()

model_init.cuda()
model_init.eval()

model_large.cuda()
model_large.eval()

val_acc_small = 0
val_acc_init = 0
val_acc_large = 0
num_val_images = 0
for _,  (images, labels) in enumerate(test_dataloader):

    images = images.to(device)
    labels = labels.to(device)
    
    out_small, _ = model_small(images)
    pred_small = torch.argmax(out_small, dim=1)

    out_init, _ = model_init(images)
    pred_init = torch.argmax(out_init, dim=1)

    out_large, _ = model_large(images)
    pred_large = torch.argmax(out_large, dim=1)

    val_acc_small += torch.sum(torch.where(pred_small==labels, 1, 0))
    val_acc_init += torch.sum(torch.where(pred_init==labels, 1, 0))
    val_acc_large += torch.sum(torch.where(pred_large==labels, 1, 0))

    num_val_images += images.shape[0]

val_acc_small = val_acc_small/num_val_images
val_acc_init = val_acc_init/num_val_images
val_acc_large = val_acc_large/num_val_images

print("Val Acc small: {}, init: {}, large: {}".format(val_acc_small, val_acc_init, val_acc_large))