import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, MultiStepLR
import argparse
import os
import math

from resnet import MakeResnet
from vgg import MakeVGG
from helper_functions import CreateModelName, CreateDataLoader
from conf import get_args
from imagenet_loader import ImagenetDataset
from tiny_imagenet_loader import TinyImagenetDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()
train_dataloader, test_dataloader = CreateDataLoader(args)  

if args.dataset == 'cifar10':
    output_size = 10
elif args.dataset == 'cifar100':
    output_size = 100 
elif args.dataset == 'imagenet':
    output_size = 1000
elif args.dataset == 'tiny_imagenet':
    output_size = 200

model_large = MakeVGG(num_linear_layers=2, output_size=output_size, dataset=args.dataset, block_size=[1, 1, 2, 2, 2])
model_init = MakeVGG(num_linear_layers=2, output_size=output_size, dataset=args.dataset, block_size=[1, 1, 2, 2, 2])
model_small = MakeVGG(num_linear_layers=2, output_size=output_size, dataset=args.dataset, block_size=[1, 1, 1, 1, 1])

new_sd = torch.load('./weights/cifar10/sandwich_vgg11/model.pth')
old_sd = torch.load('./weights/cifar10/sandwich_vgg11/model_1.pth')

model_large.load_state_dict(new_sd)
model_small.load_state_dict(old_sd)

new_sd.update(old_sd)

for key in new_sd.keys():
    txt = key.split('.')

    if int(txt[2]) > 0:
        if 'conv.weight' in key:
            print(key, "weight")
            kernel = torch.zeros(new_sd[key].shape).cuda()
            for i in range(new_sd[key].shape[0]):
                kernel[i, i, int(kernel.shape[2]/2), int(kernel.shape[3]/2)] = 1
            new_sd[key] = kernel

        elif 'conv.bias' in key:
            ## bias is fully 0
            print(key, "bias")
            new_sd[key] = torch.zeros(new_sd[key].shape).cuda()

        elif 'norm.weight' in key or 'norm.bias' in key:
            print(key, "norm wb")
            ## current layer name vgg.layerx.y.block.norm.weight. Set the weight from vgg.layerx.y-1.block.norm.weight so that the output after norm is same
            txt[2] = str(int(txt[2])-1)
            key2 = '.'.join(txt)
            new_sd[key] = new_sd[key2].detach().clone()

        elif 'running_mean' in key:
            print(key, "rmean")
            txt[-1] = 'running_mean'
            key2 = '.'.join(txt)
            new_sd[key2] = new_sd[key].detach().clone()

        elif 'running_var' in key:
            print(key, "rvar")
            txt[-1] = 'running_var'
            key2 = '.'.join(txt)
            new_sd[key2] = torch.square(new_sd[key].detach().clone())

model_init.load_state_dict(new_sd)
            
for key, values in model_init.named_parameters():
    print(key, torch.sqrt(torch.sum(torch.square(values))))


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
    
    out_small = model_small(images)
    pred_small = torch.argmax(out_small, dim=1)

    out_init = model_init(images)
    pred_init = torch.argmax(out_init, dim=1)

    out_large = model_large(images)
    pred_large = torch.argmax(out_large, dim=1)

    val_acc_small += torch.sum(torch.where(pred_small==labels, 1, 0))
    val_acc_init += torch.sum(torch.where(pred_init==labels, 1, 0))
    val_acc_large += torch.sum(torch.where(pred_large==labels, 1, 0))

    num_val_images += images.shape[0]

val_acc_small = val_acc_small/num_val_images
val_acc_init = val_acc_init/num_val_images
val_acc_large = val_acc_large/num_val_images

print("Val Acc small: {}, init: {}, large: {}".format(val_acc_small, val_acc_init, val_acc_large))