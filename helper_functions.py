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
    Optimizer Scheduler to reduce learning rate if there exists too much bounce, and stop if loss doesnt go down
    
    num_additions: the number of blocks to be added, so that we dont train the early blocks for too long
    window: window over which we consider if the accuracy changes
    acc_list: test accuracy list
    acc_swing: difference in current and prev accuracy over the window
    min_lr: to stop training
    change: to indicate if we can add the next layers
    '''

    def __init__(self, args, num_additions):
        self.window = 10
        self.loss_list = []
        self.loss_swing = []
        self.args = args
        
        self.loss_swing_thresh = 1e-3

        if self.args.combined:
            self.min_lr = 2e-4
            self.window = 5
        else:
            self.min_lr = 1e-5
            self.window = 10
        
        self.change = True
        self.flag = 0
        self.pos = 0
        self.num_additions = num_additions


    def update(self, loss, optim):

        self.change = False

        if self.pos == self.num_additions-1 and self.flag == 0:
            print("Last Training")
            self.min_lr = 1e-5
            self.window = 10
            self.flag += 1

        if len(self.loss_list) >= self.window:
            self.loss_list.pop(0)
            self.loss_swing.pop(0)
        
        self.loss_list.append(loss)

        if len(self.loss_list) > 1:
            self.loss_swing.append(self.loss_list[-1]-self.loss_list[-2])

        if len(self.loss_list) == self.window and abs(sum(self.loss_swing)) < self.loss_swing_thresh:
            print("Changing Learning Rate")
            self.loss_list = []
            self.loss_swing = []
            self.loss_swing_thresh /= 2
            for op in optim.param_groups:
                op['lr'] /= 2
                if op['lr'] < self.min_lr:
                    self.change = True
 
        if self.change:
            if self.pos != self.num_additions-1:
                self.loss_swing_thresh = 0.1
                print("Changing from {}".format(self.pos))
                
            else:
                print("Converged")
                exit()

            self.pos += 1
            