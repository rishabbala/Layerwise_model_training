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
from helper_functions import CreateModelName, CreateDataLoader
from conf import get_args
from imagenet_loader import ImagenetDataset
from tiny_imagenet_loader import TinyImagenetDataset
from create_model import CreateBlockwiseResnet



def train():

    model.train()
    train_loss = 0
    train_acc = 0
    num_train_images = 0
    for idx, (images, labels) in enumerate(train_dataloader):
        
        optim.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        
        out = model(images)
        pred = torch.argmax(out, dim=1)

        loss = cross_entropy_loss(out, labels)
        train_acc += torch.sum(torch.where(pred==labels, 1, 0))
        train_loss += loss.item()
        num_train_images += images.shape[0]

        loss.backward()
        optim.step()

        # print(model.state_dict()['resnet.layer0.2.block.0.weight'])

    train_acc = train_acc/num_train_images
    train_loss = train_loss/num_train_images

    return train_acc, train_loss
    


def test():

    model.eval()
    test_loss = 0
    test_acc = 0
    num_test_images = 0

    for _,  (images, labels) in enumerate(test_dataloader):
    
        images = images.to(device)
        labels = labels.to(device)
        
        out = model(images)
        pred = torch.argmax(out, dim=1)

        loss = cross_entropy_loss(out, labels)
        test_loss += loss.item()
        test_acc += torch.sum(torch.where(pred==labels, 1, 0))
        num_test_images += images.shape[0]
    
    test_acc = test_acc/num_test_images
    test_loss = test_loss/num_test_images

    return test_acc, test_loss



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = get_args()
    parameters = CreateModelName(args)
    writer = SummaryWriter('./runs/{}/{}/'.format(args.dataset, args.model_name))

    train_dataloader, test_dataloader = CreateDataLoader(args)  

    model = parameters['model']
    optim_parameters = parameters['optim_parameters']
    num_layers_below = parameters['num_layers_below']
    base_size = parameters['base_size']
    layer_increase = parameters['layer_increase']
    share_pos = parameters['share_pos']
    num_epochs_per_each_block = parameters['num_epochs_per_each_block']
    output_size = parameters['output_size']


    model = model.to(device)
    cross_entropy_loss = nn.CrossEntropyLoss()

    # Set the optimizer and scheduler for non-blockwise training
    if not args.train_solo_layers:
        if not args.share_weights:
            optim = Adam(optim_parameters, lr=args.lr, weight_decay=1e-5)
            lr_scheduler = MultiStepLR(optim, milestones=[0.3*args.n_epochs, 0.6*args.n_epochs, 0.8*args.n_epochs], gamma=0.2, last_epoch=-1)
        else:
            ## Warmup when weight sharing
            optim = Adam(optim_parameters, lr=args.lr, weight_decay=1e-5)
            lr_scheduler = ChainedScheduler([MultiStepLR(optim, milestones=[0.3*args.n_epochs, 0.6*args.n_epochs, 0.8*args.n_epochs], gamma=0.2, last_epoch=-1),
                                             LinearLR(optim, start_factor=1, end_factor=1, total_iters=0.1*args.n_epochs)])
    
    model.train()
    test_loss_max = 1e9
    prev_epoch = 0


    for epoch in range (args.n_epochs):

        # If adding and training layers sequentially, update the optimizer, scheduler, and num epochs accordingly
        if args.train_solo_layers and args.share_weights and (share_pos == -1 or (share_pos != -1 and epoch-prev_epoch >= num_epochs_per_each_block[share_pos])):
            test_loss_max = 1e9
            prev_epoch = epoch
            share_pos += 1

            if share_pos > 0:
                base_size[share_pos] += layer_increase[share_pos]
                model, optim_parameters = CreateBlockwiseResnet(args, base_size, output_size, './weights/{}/{}/model.pth'.format(args.dataset, args.model_name), share_pos)
                model.to(device)
            
            for key, value in model.named_parameters():
                print(key, value.requires_grad)

            optim = Adam(optim_parameters, lr=args.lr, weight_decay=1e-5)
            
            # Warmup
            lr_scheduler = ChainedScheduler([MultiStepLR(optim, milestones=[int(0.3*num_epochs_per_each_block[share_pos]), int(0.6*num_epochs_per_each_block[share_pos]), int(0.8*num_epochs_per_each_block[share_pos])], gamma=0.2, last_epoch=-1),
                                             LinearLR(optim, start_factor=1, end_factor=1, total_iters=0.2*num_epochs_per_each_block[share_pos])])

        train_acc, train_loss = train()
        test_acc, test_loss = test()

        if test_loss < test_loss_max:

            test_loss_max = test_loss
            torch.save(model.state_dict(), './weights/{}/{}/model.pth'.format(args.dataset, args.model_name))

            if args.share_weights:
                # Because of warmup
                torch.save({'train_loss': train_loss,
                            'train_acc': train_acc, 
                            'test_loss': test_loss,
                            'test_acc': test_acc,
                            'epoch': epoch, 
                            'lr': lr_scheduler._schedulers[-1].get_last_lr()}, './weights/{}/{}/info.pth'.format(args.dataset, args.model_name))
                    
            else:
                # Because of no warmup
                torch.save({'train_loss': train_loss,
                            'train_acc': train_acc, 
                            'test_loss': test_loss,
                            'test_acc': test_acc,
                            'epoch': epoch, 
                            'lr': lr_scheduler.get_last_lr()}, './weights/{}/{}/info.pth'.format(args.dataset, args.model_name))


        if args.share_weights:
            print("Epoch: {}/{}, model: {}, lr multi step: {}, lr final: {}".format(epoch, args.n_epochs, args.model_name, lr_scheduler._schedulers[0].get_last_lr()[0], lr_scheduler._schedulers[-1].get_last_lr()[0]))
        else:
            print("Epoch: {}/{}, model: {}, lr: {}".format(epoch, args.n_epochs, args.model_name, lr_scheduler.get_last_lr()[0]))        

        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Acc', train_acc, epoch)
        writer.add_scalar('Val Loss', test_loss, epoch)
        writer.add_scalar('Val Acc', test_acc, epoch)

        lr_scheduler.step()