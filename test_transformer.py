import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import copy
import pickle
import os
import shutil
import math
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, MultiStepLR
import numpy as np
from config import get_args

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.optim import create_optimizer_v2, optimizer_kwargs

from models.config import get_block_size, get_model_func
from data_pruning.memorize_dataloader import MemorizationDataloader

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

if args.dataset == "cifar10":
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=None)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=None)

if args.dataset == "cifar100":
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=None)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=None)

if args.dataset == "imagenet":
    train_dataset = ImagenetDataset(split='train', transforms=None)
    test_dataset = ImagenetDataset(split='val', transforms=None)

# must recompute imagenet values for now
if args.dataset == "tiny_imagenet":
    train_dataset = TinyImagenetDataset(split='train', transforms=None)
    test_dataset = TinyImagenetDataset(split='val', transforms=None)


args.batch_size = 256
train_dataloader = create_loader(
                            train_dataset,
                            input_size=[3, 32, 32],
                            batch_size=args.batch_size,
                            is_training=True,
                            use_prefetcher=True,
                            no_aug=False,
                            re_prob=0.25, ## prob of erasing area
                            re_mode='pixel', ## method of replacing erased area
                            re_count=1,  ## num of erased area
                            re_split=False, ## bool for of different splits per batch
                            scale=[0.8, 1.0],
                            ratio=[3/4, 4/3],
                            hflip=0.5,
                            vflip=0,
                            color_jitter=0.4,
                            auto_augment='rand-m9-mstd0.5-inc1', ## rand augment
                            num_aug_splits=0, ## bool for of different splits per batch
                            interpolation='random',
                            mean=mean_dataset_norm,
                            std=std_dataset_norm,
                            num_workers=16,
                            distributed=False,
                            # device=self.device,
    )


test_dataloader = create_loader(
    test_dataset,
    input_size=[3, 32, 32],
    batch_size=args.batch_size,
    is_training=False,
    use_prefetcher=True,
    interpolation='bicubic',
    mean=mean_dataset_norm,
    std=std_dataset_norm,
    num_workers=16,
    crop_pct=1.0
)

func = get_model_func(args)
block, num_epochs_per_block = get_block_size(args)

model_large = func(feature_dim=256, mlp_dim=512, num_blocks=block[-1][0], num_heads=4, output_size=args.output_size)
model_init = func(feature_dim=256, mlp_dim=512, num_blocks=block[-1][0], num_heads=4, output_size=args.output_size)
model_small = func(feature_dim=256, mlp_dim=512, num_blocks=block[-1][0], num_heads=4, output_size=args.output_size)

new_sd = torch.load('./weights/cifar100/combined_vit_v2/model.pth')
old_sd = torch.load('./weights/cifar100/combined_vit_v2/model.pth')

model_large.load_state_dict(new_sd)
model_small.load_state_dict(old_sd)

# new_sd.update(old_sd)

model_init.load_state_dict(new_sd)


# for key, values in model_init.named_parameters():
#     txt = key.split('.')

#     try:
#         if 'transformer' in key and int(txt[1][-1]) == 6 and 'norm' in key:
#             values.data = torch.zeros(values.data.shape)
#     except:
#         pass

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
    
    # out_small = model_small(images)
    # pred_small = torch.argmax(out_small, dim=1)

    # out_init = model_init(images)
    # pred_init = torch.argmax(out_init, dim=1)

    out_large = model_large(images)
    pred_large = torch.argmax(out_large, dim=1)

    # val_acc_small += torch.sum(torch.where(pred_small==labels, 1, 0))
    # val_acc_init += torch.sum(torch.where(pred_init==labels, 1, 0))
    val_acc_large += torch.sum(torch.where(pred_large==labels, 1, 0))

    num_val_images += images.shape[0]

# val_acc_small = val_acc_small/num_val_images
# val_acc_init = val_acc_init/num_val_images
val_acc_large = val_acc_large/num_val_images

print("Val Acc small: {}, init: {}, large: {}".format(val_acc_small, val_acc_init, val_acc_large))