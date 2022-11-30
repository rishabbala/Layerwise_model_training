import torch
import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import shutil
import copy



def CreateModelName(args):

    '''
    Create a distinct model name
    Remove the folder in runs and weights if that model was previously trained 
    '''

    if args.model_name == None:
        raise NotImplementedError("Model name not provided")

    # Set the name of the model for storing
    if args.memorize:
        if args.easy:
            args.model_name = 'memorize_easy_' + args.model_name
        else:
            args.model_name = 'memorize_hard_' + args.model_name
        if args.exclusive:
            args.model_name = 'exclusive_' + args.model_name

    if args.combined:
        args.model_name = 'combined_' + args.model_name
    if args.early_stop:
        args.model_name += '_early_stop'  

    args.model_name += args.temp_str  


    ### Remove previous tfevents Create directory for weights
    if os.path.isdir('./runs/{}/{}'.format(args.dataset, args.model_name)):
        shutil.rmtree('./runs/{}/{}'.format(args.dataset, args.model_name))
        os.makedirs('./runs/{}/{}'.format(args.dataset, args.model_name))
    if not os.path.isdir('./weights/{}/{}'.format(args.dataset, args.model_name)):
        os.makedirs('./weights/{}/{}'.format(args.dataset, args.model_name))

    if args.dataset == 'cifar10':
        args.output_size = 10
    elif args.dataset == 'cifar100':
        args.output_size = 100 
    elif args.dataset == 'imagenet':
        args.output_size = 1000
    elif args.dataset == 'tiny_imagenet':
        args.output_size = 200
    else:
        raise ValueError("Dataset not availabe")


class OptimScheduler():

    '''
    Optimizer Scheduler to reduce learning rate if there exists too much bounce, and stop if accuracy doesnt go up
    
    num_additions: the number of blocks to be added, so that we dont train the early blocks for too long
    window: window over which we consider if the accuracy changes
    acc_list: test accuracy list
    acc_swing: difference in current and prev accuracy over the window
    min_lr: to stop training
    change: to indicate if we can add the next layers
    '''

    def __init__(self, args, num_additions):
        self.window = 5
        self.acc_list = []
        self.acc_swing = []
        self.args = args
        
        self.acc_swing_thresh = 0.03

        if self.args.early_stop:
            self.min_lr = 2e-4
            self.window = 3
        else:
            self.min_lr = 5e-5
            self.window = 5
        
        self.change = True
        self.flag = 0
        self.pos = 0
        self.decay_epochs = 0
        self.num_additions = num_additions

        self.lr = 0


    def update(self, acc, optim):

        acc *= 100

        if self.decay_epochs > 0:
            print("Cosine decay final")
            for op in optim.param_groups:
                op['lr'] = self.lr * np.cos(np.pi*self.decay_epochs/(2*10)) ## 10 epochs decay to zero
                if op['lr'] <= 0:
                    exit()
                
            self.decay_epochs += 1
            return

        if self.pos == self.num_additions-1 and self.flag == 0:
            print("Last Training")
            self.min_lr = 5e-5
            self.window = 5
            self.flag += 1

        self.change = False

        if len(self.acc_list) >= self.window:
            self.acc_list.pop(0)
            self.acc_swing.pop(0)
        
        self.acc_list.append(acc)

        if len(self.acc_list) > 1:
            self.acc_swing.append(self.acc_list[-1]-self.acc_list[-2])

        if len(self.acc_list) == self.window and abs(sum(self.acc_swing)) < self.acc_swing_thresh:

            self.acc_swing_thresh /= 2

            for op in optim.param_groups:
                op['lr'] /= 2
                self.lr = op['lr']

                if op['lr'] <= self.min_lr:
                    self.change = True
            
        if self.change:
            if self.pos != self.num_additions-1:
                self.acc_swing_thresh = 0.1
                print("Changing from {}".format(self.pos))
                
            else:
                self.decay_epochs = 1
                self.change = False
                print("Converged")

            self.acc_list = []
            self.acc_swing = []
            self.pos += 1



import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, MultiStepLR
import argparse
import os
import math
from torchvision import transforms
import time
from timm.scheduler import create_scheduler_v2
from timm.loss import SoftTargetCrossEntropy
from sgd_hd import SGDHD
from adam_hd import AdamHD
from adamw_hd import AdamWHD
import math


from helper_functions import CreateModelName, OptimScheduler
from config import get_args
from dataloader.imagenet_loader import ImagenetDataset
from dataloader.tiny_imagenet_loader import TinyImagenetDataset
from create_model import ModelClass
from loss.conloss import ContrastiveLoss



def train():
    '''
    train one epoch:
        model: taken from main()
    '''

    model.train()

    train_loss = 0
    train_acc = 0
    num_train_images = 0
    train_ce_loss = 0

    for idx, (images, labels) in enumerate(train_dataloader):

        optim.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        out, features = model(images)

        ce_loss = soft_cross_entropy_loss(out, labels)

        # ce_loss = cross_entropy_loss(out, labels)

        pred = torch.argmax(out, dim=1)
        loss = ce_loss

        labels_max = torch.argmax(labels, dim=1)
        train_acc += torch.sum(torch.where(pred==labels_max, 1, 0))

        # train_acc += torch.sum(torch.where(pred==labels, 1, 0))

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
    test_con_loss = 0
    test_ce_loss = 0
    test_ortho_loss = 0

    for _,  (images, labels) in enumerate(test_dataloader):

        images = images.to(device)
        labels = labels.to(device)
        
        out, features = model(images)

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


if __name__ == "__main__":
    '''
    main:
        CreateModelName: generates the model name from the input args
    '''


    start = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device", device)

    args = get_args()
    CreateModelName(args)
    writer = SummaryWriter('./runs/{}/{}/'.format(args.dataset, args.model_name))

    mc = ModelClass(args, device)

    soft_cross_entropy_loss = SoftTargetCrossEntropy()
    cross_entropy_loss = nn.CrossEntropyLoss()

    test_loss_max = 1e9
    prev_epoch = 0
    optim = None

    # model, optim, lr_scheduler, train_dataloader, test_dataloader = mc.create_custom_model(optim)

    # no_decay = []
    # decay = []

    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         continue

    #     if param.ndim <= 1 or name.endswith(".bias"):
    #         no_decay.append(param)
    #     else:
    #         decay.append(param)

    # ## turn off weight decay for batch norm and filter bias
    # """ Optimizer Factory w/ Custom Weight Decay
    # Hacked together by / Copyright 2021 Ross Wightman
    # """
    # param_groups = [
    #     {'params': no_decay, 'weight_decay': 0.},
    #     {'params': decay, 'weight_decay': 6e-2}
    #     ]

    # ## 1e-8 for 11507082 => 1e-15 for 1
    # ## 
    
    # # optim = SGDHD(model.parameters(), lr=args.lr, momentum=0.9, hypergrad_lr=3e-5)
    # # optim = AdamHD(model.parameters(), lr=6e-5, hypergrad_lr=1e-11)
    # optim = AdamWHD(param_groups, lr=1e-3, hypergrad_lr=1e-5, weight_decay=6e-2)

    # model.to(device)
    # for key, value in model.named_parameters():
    #     print(key, value.requires_grad)


    # for epoch in range (args.n_epochs):

    #     # # if optim_scheduler.change:
    #     # # prev_epoch = epoch

    #     # # lr_scheduler, _ = create_scheduler_v2(optimizer=optim, sched='cosine', num_epochs=args.n_epochs, decay_epochs=30, decay_milestones=[30, 60], cooldown_epochs=10, patience_epochs=10, decay_rate=0.1, min_lr=1e-5, warmup_lr=1e-5, warmup_epochs=10, warmup_prefix=False, noise=None)

    #     # # optim_scheduler.change = False

        
    #     # # print(optim_scheduler.loss_swing_thresh)
    #     # print(args.batch_size)

    #     if epoch > 0 and epoch%50 == 0:
    #         model, _, _, _, _ = mc.create_custom_model(optim)

    #         for name, param in model.named_parameters():
    #             if not param.requires_grad:
    #                 continue

    #             if param.ndim <= 1 or name.endswith(".bias"):
    #                 no_decay.append(param)
    #             else:
    #                 decay.append(param)

    #         param_groups = [
    #             {'params': no_decay, 'weight_decay': 0.}, # , 'lr': optim.param_groups[0]['lr']
    #             {'params': decay, 'weight_decay': 6e-2}
    #         ]

    #         optim = AdamWHD(param_groups, lr=1e-3, hypergrad_lr=1e-5, weight_decay=6e-2)
    #         model.to(device)
    #         for key, value in model.named_parameters():
    #             print(key, value.requires_grad)


    #     train_acc, train_loss, train_ce_loss = train()
            
    #     with torch.no_grad():
    #         model.eval()
    #         test_acc, test_loss, test_ce_loss = test()
    #         model.train()
        
    #     if test_loss < test_loss_max:

    #         test_loss_max = test_loss
    #         torch.save(model.state_dict(), './weights/{}/{}/model.pth'.format(args.dataset, args.model_name))

    #         # Because of warmup
    #         torch.save({'train_loss': train_loss,
    #                     'train_ce_loss': train_ce_loss,
    #                     'train_acc': train_acc, 
    #                     'test_loss': test_loss,
    #                     'test_ce_loss': test_ce_loss,
    #                     'test_acc': test_acc,
    #                     'epoch': epoch}, './weights/{}/{}/info.pth'.format(args.dataset, args.model_name)) #'lr': lr_scheduler.get_last_lr()[0]

    #     print("Epoch: {}/{}, model: {}, lr:{}".format(epoch, args.n_epochs, args.model_name, optim.param_groups[1]['lr']))

    #     writer.add_scalar('Train Loss', train_loss, epoch)
    #     writer.add_scalar('Train Cross Entropy Loss', train_ce_loss, epoch)
    #     writer.add_scalar('Train Acc', train_acc, epoch)
    #     writer.add_scalar('Val Loss', test_loss, epoch)
    #     writer.add_scalar('Val Cross Entropy Loss', test_ce_loss, epoch)
    #     writer.add_scalar('Val Acc', test_acc, epoch)
    #     writer.add_scalar('Learning Rate', optim.param_groups[1]['lr'], epoch)

    #     # if optim.param_groups[0]['lr'] < 1e-6:
    #     #     exit()

    #     # lr_scheduler.step(epoch)

    #     # lr_scheduler.step()

    #     # if mc.block[mc.cur_pos][-1] == -1:
    #     # loss_to_use = train_loss
    #     # else:
    #     #     loss_to_use = test_loss
        
    #     # if epoch-prev_epoch>args.warmup_epochs:
    #     # optim_scheduler.update(loss_to_use, optim)

    #     # if optim_scheduler.change:
    #     #     torch.save(model.state_dict(), './weights/{}/{}/model_{}.pth'.format(args.dataset, args.model_name, optim_scheduler.pos))

    test_loss_max = 1e9
    prev_epoch = 0
    optim = None
    optim_scheduler = OptimScheduler(args, len(mc.block))

    for epoch in range (args.n_epochs):
        
        if epoch == 0:
        # if optim_scheduler.change:
        # if epoch%mc.num_epochs_per_block == 0 and mc.cur_pos != len(mc.block)-1:
            test_loss_max = 1e9
            prev_epoch = epoch

            model, optim, _ = mc.create_custom_model(optim)
            # model, optim, lr_scheduler = mc.create_custom_model(optim)


            # if epoch == 0:
            #decay_milestones=[30, 60]
            # if epoch == 0:
            lr_scheduler, _ = create_scheduler_v2(optimizer=optim, sched='cosine', num_epochs=args.n_epochs, min_lr=1e-5, warmup_lr=1e-6, warmup_epochs=10, cooldown_epochs=10, patience_epochs=10, warmup_prefix=False, noise=None)
            # else:
            #     lr_scheduler, _ = create_scheduler_v2(optimizer=optim, sched='cosine', num_epochs=150, min_lr=1e-5, warmup_lr=1e-5, warmup_epochs=75, warmup_prefix=False, noise=None)

            optim_scheduler.change = False

            model.to(device)
            for key, value in model.named_parameters():
                print(key, value.requires_grad) 

        # for opt in optim.param_groups:
        #     ## make the last descent go to min lr
        #     ## first l-1 blocks get x epochs, final block 2x epochs
        #     if epoch < (mc.num_epochs_per_block) * (len(mc.block)-1) + math.floor(mc.num_epochs_per_block/2):
        #         opt['lr'] = 2e-4 * math.sin((epoch%mc.num_epochs_per_block)*math.pi/mc.num_epochs_per_block) + 3e-4
        #     # elif epoch >= (mc.num_epochs_per_block) * (len(mc.block)-1) and epoch < (mc.num_epochs_per_block) * (len(mc.block)-1) + mc.num_epochs_per_block:
        #     #     opt['lr'] = 2e-4 * math.sin((epoch%(mc.num_epochs_per_block))*math.pi/(2*mc.num_epochs_per_block)) + 3e-4
        #     else:
        #         opt['lr'] = 5e-4 * math.sin(math.pi/2 + (epoch%(1.5*mc.num_epochs_per_block))*math.pi/(3*mc.num_epochs_per_block)) + 1e-5

        if optim_scheduler.change_batch_size:
            optim_scheduler.change_batch_size = False
            print("Changeing Batch size to", optim_scheduler.args.batch_size)
            train_dataloader, test_dataloader = mc.CreateDataLoader(optim_scheduler.args.batch_size)

        train_acc, train_loss, train_ce_loss = train()
            
        with torch.no_grad():
            model.eval()
            test_acc, test_loss, test_ce_loss = test()
            model.train()
        
        if test_loss < test_loss_max:

            test_loss_max = test_loss
            torch.save(model.state_dict(), './weights/{}/{}/model.pth'.format(args.dataset, args.model_name))

            # Because of warmup
            torch.save({'train_loss': train_loss,
                        'train_ce_loss': train_ce_loss,
                        'train_acc': train_acc, 
                        'test_loss': test_loss,
                        'test_ce_loss': test_ce_loss,
                        'test_acc': test_acc,
                        'epoch': epoch}, './weights/{}/{}/info.pth'.format(args.dataset, args.model_name)) #'lr': lr_scheduler.get_last_lr()[0]

        print("Epoch: {}/{}, model: {}, lr:{}".format(epoch, args.n_epochs, args.model_name, optim.param_groups[1]['lr']))

        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Cross Entropy Loss', train_ce_loss, epoch)
        writer.add_scalar('Train Acc', train_acc, epoch)
        writer.add_scalar('Val Loss', test_loss, epoch)
        writer.add_scalar('Val Cross Entropy Loss', test_ce_loss, epoch)
        writer.add_scalar('Val Acc', test_acc, epoch)
        writer.add_scalar('Learning Rate', optim.param_groups[0]['lr'], epoch)

        lr_scheduler.step(epoch)

        # lr_scheduler.step()

        # acc_to_use = test_acc
        
        # # if epoch-prev_epoch>args.warmup_epochs:
        # optim_scheduler.update(acc_to_use, optim)

        # # if optim_scheduler.change:
        # #     torch.save(model.state_dict(), './weights/{}/{}/model_{}.pth'.format(args.dataset, args.model_name, optim_scheduler.pos))



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
            model = self.func(feature_dim=768, mlp_dim=3072, num_blocks=self.block[self.cur_pos][0], num_heads=12, output_size=self.args.output_size)
        elif 'cct' in self.args.model_name:
            model = self.func(feature_dim=128, mlp_dim=128, num_blocks=self.block[self.cur_pos][0], num_heads=2, output_size=self.args.output_size)

        model, new_optim, lr_scheduler = self.weight_share(model, optim)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of Trainable parameters", num_params)

        return model, new_optim, lr_scheduler

    
    def CreateDataLoader(self, batch_size):

        '''
        Create the dataloader
        '''

        self.args.batch_size = batch_size

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

        # Defined transforms
        # if self.args.contrastive and self.cur_pos != len(self.block)-1:
        #     ## transforms for contrastive learning
        #     if 'cifar' in self.args.dataset:
        #         train_transforms = TwoCropTransform(
        #                                             transforms.Compose([
        #                                                 transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #                                                 transforms.RandomHorizontalFlip(),
        #                                                 transforms.RandomApply([
        #                                                     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #                                                 ], p=0.8),
        #                                                 transforms.RandomGrayscale(p=0.2),
        #                                                 transforms.ToTensor(),
        #                                                 transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)
        #                                             ])
        #         )

        #         test_transforms = TwoCropTransform(
        #                                             transforms.Compose([
        #                                                 # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #                                                 # transforms.RandomHorizontalFlip(),
        #                                                 # transforms.RandomApply([
        #                                                 #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #                                                 # ], p=0.8),
        #                                                 # transforms.RandomGrayscale(p=0.2),
        #                                                 transforms.ToTensor(),
        #                                                 transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)
        #                                             ])
        #         )
                                    
        #     else:
        #         raise ValueError("Transform for dataset not implemented")

        # else:
            ## for cross entropy training

        train_transforms = None
        test_transforms = None

        # train_transforms = transforms.Compose([
        #                             transforms.RandomCrop(32, padding=4),
        #                             # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #                             transforms.RandomHorizontalFlip(),
        #                             transforms.RandomRotation(15),
        #                             transforms.RandomApply([
        #                                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #                             ], p=0.8),
        #                             transforms.RandomGrayscale(p=0.2),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)
        #                         ])

        # test_transforms = transforms.Compose([
        #                             transforms.ToTensor(),
        #                             transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)
        #                         ])

        if self.args.memorize:

            percent_data = (self.cur_pos+1)/len(self.block)
            prev_percent_data = (self.cur_pos)/len(self.block)

            dataset_pruned = copy.deepcopy(self.dataset)

            num_imgs = 0
            for key in dataset_pruned.keys():
                if self.args.exclusive:
                    dataset_pruned[key] = dataset_pruned[key][int(prev_percent_data * len(dataset_pruned[key])):int(percent_data * len(dataset_pruned[key]))]
                else:
                    dataset_pruned[key] = dataset_pruned[key][:int(percent_data * len(dataset_pruned[key]))]
                num_imgs += len(dataset_pruned[key])
                print(key, len(dataset_pruned[key]))

            train_dataset = MemorizationDataloader(dataset_pruned, num_imgs, train_transforms)

            if self.args.dataset == "cifar10":
                test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transforms)

            if self.args.dataset == "cifar100":
                test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transforms)

            # if self.args.dataset == "imagenet":
            #     test_dataset = ImagenetDataset(split='val', transforms=test_transforms)

            # # must recompute imagenet values for now
            # if self.args.dataset == "tiny_imagenet":
            #     test_dataset = TinyImagenetDataset(split='val', transforms=test_transforms)

        else:

            if self.args.dataset == "cifar10":
                train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transforms)
                test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transforms)

            if self.args.dataset == "cifar100":
                train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=train_transforms)
                test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transforms)

            if self.args.dataset == "imagenet":
                train_dataset = ImagenetDataset(split='train', transforms=train_transforms)
                test_dataset = ImagenetDataset(split='val', transforms=test_transforms)

            # must recompute imagenet values for now
            if self.args.dataset == "tiny_imagenet":
                train_dataset = TinyImagenetDataset(split='train', transforms=train_transforms)
                test_dataset = TinyImagenetDataset(split='val', transforms=test_transforms)

        # if not self.args.contrastive or (self.args.contrastive and self.block[self.cur_pos][-1] != -1):
        #     ## reduce batch size for non contrastive training
        #     self.args.batch_size = 128

        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=16)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=16)


        # if 'cct' in self.args.model_name or 'vit' in self.args.model_name:
        mixup = 0.8
        cutmix = 1.0
        mixup_prob = 1.0
        mixup_switch_prob = 0.5
        mixup_mode = 'batch'
        mixup_off_epoch = 175
        smoothing = 0.1

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

        # if self.args.reuse_encoder:
        #     if 'resnet18' in self.args.model_name:
        #         base_model = 'resnet18'
        #     elif 'resnet18' in self.args.model_name:
        #         base_model = 'resnet34'

        #     sd = model.state_dict()

        #     base_dict = torch.load('./weights/{}/{}/model.pth'.format(self.args.dataset, base_model))
        #     keys = list(base_dict.keys()).copy()

        #     for key in keys:
        #         if 'encoder' not in key:
        #             del base_dict[key]
            
        #     sd.update(base_dict)
        #     model.load_state_dict(sd)

        #     for key, values in model.named_parameters():
        #         if 'encoder' in key:
        #             values.requires_grad = False


        if optim == None:

            # if self.args.contrastive:
            #     if self.block[self.cur_pos][-1] == -1:
            #         for key, values in model.named_parameters():
            #             if 'linear_layers' in key:
            #                 values.requires_grad = False

            #     elif self.block[self.cur_pos][-1] == 0:
            #         for key, values in model.named_parameters():
            #             if 'linear_layers' not in key:
            #                 values.requires_grad = False

            # if self.block[self.cur_pos][-1] != -1 and self.args.contrastive:
            #     weight_decay = 0
            # else:
            #     weight_decay = 5e-4

            # params_groups = []
            # for key, params in model.named_parameters():
            #     if params.requires_grad:
            #         params_groups.append({
            #             'params': params,
            #             'lr': self.args.lr,
            #             'momentum': 0.9
            #         })

            # new_optim = SGD(params_groups)


            ## update only params with non blocked grad in optim
            # new_optim = SGD(filter(lambda param: param.requires_grad, model.parameters())
            #                 , lr=self.args.lr, momentum=0.9) #, weight_decay=5e-4

            # new_optim = AdamW(filter(lambda param: param.requires_grad, model.parameters())
            #                 , lr=55e-5) # , weight_decay=6e-2

            new_optim = create_optimizer_v2(model, opt='adamw', lr=6e-4, weight_decay=6e-2, filter_bias_and_bn=True)

        else:

            # self.args.lr = 0.2

            ## Load all previous layers with weight from smaller model
            sd = model.state_dict()
            try:
                child_sd = torch.load(self.child_weight_path)
                keys = list(child_sd.keys()).copy()

                ## Dont update the fp and lin layers
                # if self.args.blockwise:
                #     for key in keys:
                #         if 'feature_projection_layers' in key or 'linear_layers' in key:
                #             del child_sd[key]

                sd.update(child_sd)
            except:
                raise ValueError("Base model not trained yet")
            model.load_state_dict(sd)

            # ## Freeze gradients where required
            # for key, values in model.named_parameters():
            #     if self.args.frozen_upper:
            #         if 'resnet' in key:
            #             txt = key.split('.')
            #             try:
            #                 if int(txt[1][-1]) < self.cur_pos:
            #                     values.requires_grad = False
            #             except:
            #                 if self.cur_pos > 0 and 'encoder' in key:
            #                     values.requires_grad = False

            #     if self.args.frozen_prev:
            #         if 'resnet' in key:
            #             if key in child_sd.keys() and 'linear_layers' not in key:
            #                 values.requires_grad = False

            #     if self.args.contrastive:
            #         if self.block[self.cur_pos][-1] == -1 and 'linear_layers' in key:
            #             values.requires_grad = False

            #         elif self.block[self.cur_pos][-1] == 0 and 'linear_layers' not in key:
            #             values.requires_grad = False

            

            # new_optim = SGD(filter(lambda param: param.requires_grad, model.parameters())
            #                 , lr=self.args.lr, momentum=0.9) #, weight_decay=5e-4

            # new_optim = AdamW(filter(lambda param: param.requires_grad, model.parameters())
            #                 , lr=55e-5) # , weight_decay=6e-2
            
            new_optim = create_optimizer_v2(model, opt='adamw', lr=6e-4, weight_decay=6e-2, filter_bias_and_bn=True)

            # print(new_optim.state_dict()['param_groups'])

        #0.05
        lr_scheduler = LinearLR(new_optim, start_factor=1, end_factor=1, total_iters=self.args.warmup_epochs)

        return model, new_optim, lr_scheduler