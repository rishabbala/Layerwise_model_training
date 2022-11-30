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
            self.min_lr = 1e-5
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

            # for op in optim.param_groups:
            #     op['lr'] /= 5
            #     self.lr = op['lr']

            #     if op['lr'] <= self.min_lr:
            #         self.change = True
            
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