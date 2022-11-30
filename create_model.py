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

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.optim import create_optimizer_v2, optimizer_kwargs

from models.config import get_block_size, get_model_func
from data_pruning.memorize_dataloader import MemorizationDataloader



class ModelClass():

    '''
    Create the model, optimizer and lr_scheduler
    '''

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.cur_pos = -1
        self.child_weight_path = './weights/{}/{}/model.pth'.format(self.args.dataset, self.args.model_name)

        self.func = get_model_func(self.args)
        self.block, self.num_epochs_per_block = get_block_size(self.args)

        if args.memorize:
            if args.easy:
                with open('./data/{}_memorized_easy.pkl'.format(self.args.dataset), 'rb') as f:
                    self.dataset = pickle.load(f)
            else:
                with open('./data/{}_memorized_hard.pkl'.format(self.args.dataset), 'rb') as f:
                    self.dataset = pickle.load(f)


    def create_custom_model(self, optim=None):
        '''
        Models created here

        input: 
                args: the arguments -> dict
                output_size: the final output size of the network -> int

        output:
                model: model to be trained
                optim: the optimizer
        '''

        self.cur_pos += 1

        if self.cur_pos > len(self.block):
            raise ValueError("Training for more blocks than expected")

        if self.cur_pos == len(self.block):
            print("Finished Training, Exiting")
            exit()

        if 'resnet' in self.args.model_name:
            model = self.func(num_linear_layers=self.args.num_linear_layers, output_size=self.args.output_size, dataset=self.args.dataset, block_size=self.block[self.cur_pos])
        elif 'vit' in self.args.model_name:
            model = self.func(feature_dim=256, mlp_dim=512, num_blocks=self.block[self.cur_pos][0], num_heads=4, output_size=self.args.output_size)
        elif 'cct_2' in self.args.model_name:
            model = self.func(feature_dim=128, mlp_dim=128, num_blocks=self.block[self.cur_pos][0], num_heads=2, output_size=self.args.output_size, n_conv=2)
        elif 'cct_4' in self.args.model_name:
            model = self.func(feature_dim=128, mlp_dim=128, num_blocks=self.block[self.cur_pos][0], num_heads=2, output_size=self.args.output_size, n_conv=2)
        elif 'cct_7' in self.args.model_name:
            model = self.func(feature_dim=256, mlp_dim=512, num_blocks=self.block[self.cur_pos][0], num_heads=4, output_size=self.args.output_size, n_conv=1)

        else:
            raise NotImplementedError("Model Unknown")

        model, new_optim = self.weight_share(model, optim)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of Trainable parameters", num_params)

        return model, new_optim

    
    def CreateDataLoader(self):

        '''
        Create the dataloader
        '''

        print("Batch size", self.args.batch_size)

        if self.args.dataset == "cifar10":
            mean_dataset_norm = [0.4914, 0.4822, 0.4465]
            std_dataset_norm = [0.247, 0.2434, 0.2615]

        if self.args.dataset == "cifar100":
            mean_dataset_norm = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
            std_dataset_norm = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

        if self.args.dataset == "imagenet":
            mean_dataset_norm = [0.485, 0.456, 0.406]
            std_dataset_norm = [0.229, 0.224, 0.225]

        # must recompute imagenet values for now
        if self.args.dataset == "tiny_imagenet":
            mean_dataset_norm = [0.485, 0.456, 0.406]
            std_dataset_norm = [0.229, 0.224, 0.225]

        # if self.args.memorize:

        #     percent_data = (self.cur_pos+1)/len(self.block)
        #     prev_percent_data = (self.cur_pos)/len(self.block)

        #     dataset_pruned = copy.deepcopy(self.dataset)

        #     num_imgs = 0
        #     for key in dataset_pruned.keys():
        #         if self.args.exclusive:
        #             dataset_pruned[key] = dataset_pruned[key][int(prev_percent_data * len(dataset_pruned[key])):int(percent_data * len(dataset_pruned[key]))]
        #         else:
        #             dataset_pruned[key] = dataset_pruned[key][:int(percent_data * len(dataset_pruned[key]))]
        #         num_imgs += len(dataset_pruned[key])
        #         print(key, len(dataset_pruned[key]))

        #     train_dataset = MemorizationDataloader(dataset_pruned, num_imgs, train_transforms)

        #     if self.args.dataset == "cifar10":
        #         test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transforms)

        #     if self.args.dataset == "cifar100":
        #         test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transforms)

        #     # if self.args.dataset == "imagenet":
        #     #     test_dataset = ImagenetDataset(split='val', transforms=test_transforms)

        #     # # must recompute imagenet values for now
        #     # if self.args.dataset == "tiny_imagenet":
        #     #     test_dataset = TinyImagenetDataset(split='val', transforms=test_transforms)

        # else:

        if self.args.dataset == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=None)
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=None)

        if self.args.dataset == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=None)
            test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=None)

        if self.args.dataset == "imagenet":
            train_dataset = ImagenetDataset(split='train', transforms=None)
            test_dataset = ImagenetDataset(split='val', transforms=None)

        # must recompute imagenet values for now
        if self.args.dataset == "tiny_imagenet":
            train_dataset = TinyImagenetDataset(split='train', transforms=train_transforms)
            test_dataset = TinyImagenetDataset(split='val', transforms=test_transforms)

        mixup_args = dict(
                mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
                prob=1.0, switch_prob=0.5, mode='batch',
                label_smoothing=0.1, num_classes=self.args.output_size)

        collate_fn = FastCollateMixup(**mixup_args)


        train_dataloader = create_loader(
                                    train_dataset,
                                    input_size=[3, 32, 32],
                                    batch_size=self.args.batch_size,
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
                                    collate_fn=collate_fn
                                    # device=self.device,
            )


        test_dataloader = create_loader(
            test_dataset,
            input_size=[3, 32, 32],
            batch_size=self.args.batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=mean_dataset_norm,
            std=std_dataset_norm,
            num_workers=16,
            crop_pct=1.0
        )


        return train_dataloader, test_dataloader


    def weight_share(self, model, optim=None, optim_params0_keys=[]):

        """ 
        Function to share weights between smaller and larger models
        
        output:
                model: the model to be trained -> torch model
                optim_parameters: subset of model parameters to optimize -> list
        """

        if optim == None:
            new_optim = create_optimizer_v2(model, opt='adamw', lr=6e-4, weight_decay=3e-2, filter_bias_and_bn=True)

        else:
            ## Load all previous layers with weight from smaller model
            sd = model.state_dict()
            try:
                child_sd = torch.load(self.child_weight_path)
                keys = list(child_sd.keys()).copy()
                sd.update(child_sd)
            except:
                raise ValueError("Base model not trained yet")
            model.load_state_dict(sd)

            new_optim = create_optimizer_v2(model, opt='adamw', lr=6e-4, weight_decay=3e-2, filter_bias_and_bn=True)

        return model, new_optim