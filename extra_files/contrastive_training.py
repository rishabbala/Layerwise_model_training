import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, MultiStepLR
import argparse
import os
import math
import shutil


from resnet import MakeResnet
from helper_functions import CreateModelName
from conf import get_args
from imagenet_loader import ImagenetDataset
from tiny_imagenet_loader import TinyImagenetDataset
from create_model import CreateBlockwiseResnet


# from helper_functions import CreateModelName
# from imagenet_loader import ImagenetDataset
# from tiny_imagenet_loader import TinyImagenetDataset


def CreateDataLoaderlocal(args):

    if args.dataset == "cifar10":
        mean_dataset_norm = [0.4914, 0.4822, 0.4465]
        std_dataset_norm = [0.247, 0.2434, 0.2615]

    if args.dataset == "cifar100":
        mean_dataset_norm = [0.5071, 0.4867, 0.4408]
        std_dataset_norm = [0.2675, 0.2565, 0.2761]

    if args.dataset == "imagenet":
        mean_dataset_norm = [0.485, 0.456, 0.406]
        std_dataset_norm = [0.229, 0.224, 0.225]

    # must recompute imagenet values for now
    if args.dataset == "tiny_imagenet":
        mean_dataset_norm = [0.485, 0.456, 0.406]
        std_dataset_norm = [0.229, 0.224, 0.225]

    # train_transform = transforms.Compose([transforms.ToTensor(),
    #                                       transforms.Resize((32, 32)),
    #                                       transforms.RandomCrop(32, padding=4),
    #                                       transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm),
    #                                       transforms.RandomHorizontalFlip(0.5)])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((32, 32)),
                                         transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)])

    # Defined transforms
    if args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=test_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)

    if args.dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=test_transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform)

    if args.dataset == "imagenet":
        train_dataset = ImagenetDataset(split='train', transforms=test_transform)
        test_dataset = ImagenetDataset(split='val', transforms=test_transform)

    # must recompute imagenet values for now
    if args.dataset == "tiny_imagenet":
        train_dataset = TinyImagenetDataset(split='train', transforms=test_transform)
        test_dataset = TinyImagenetDataset(split='val', transforms=test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, test_dataloader



def ConLoss(features1, features2, labels):

    temperature = 0.1

    features = torch.cat((features1, features2), dim=0)
    features = torch.nn.functional.normalize(features, dim=1)
    features = features.view(features.shape[0], features.shape[1], -1)

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    mask = torch.cat((mask, mask), dim=1)
    mask = torch.cat((mask, mask), dim=0)
    mask = mask - (torch.eye(mask.shape[0]).float().to(device).detach()).detach()
    
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) ## reshaped such that first bsz rows are first channel of all inputs, second bsz rows is 2nd channel ...  torch.unbind(features, dim=1) has shape (bsz, feature.shape[-1])

    f_dot_f = torch.matmul(contrast_feature, contrast_feature.T)/temperature

    f_max, _ = torch.max(f_dot_f, dim=1, keepdim=True)
    f_dot_f = f_dot_f - f_max.detach()
    
    # norm = torch.sqrt(torch.sum(torch.square(f_dot_f), dim=1, keepdim=True)).detach()
    # f_dot_f = f_dot_f/norm

    mask = torch.tile(mask, (features.shape[1], features.shape[1]))
    anchor_dot_pos = torch.mul(f_dot_f, mask)
    
    mask_all = (torch.ones_like(f_dot_f) - torch.eye(f_dot_f.shape[0]).to(device)).detach()
    anchor_dot_all = torch.exp(torch.mul(f_dot_f, mask_all))

    # print(features.shape, mask.shape, contrast_feature.shape, f_dot_f.shape, anchor_dot_all.shape)
    # print()
    # exit()

    loss = anchor_dot_pos - torch.log(torch.sum(anchor_dot_all, dim=1, keepdim=True))
    loss = loss * (-1/torch.sum(mask, dim=1, keepdim=True))

    loss = loss.mean()

    return loss



def train():

    train_loss = 0
    train_acc = 0
    num_train_images = 0
    for idx, (images, labels) in enumerate(train_dataloader):
                
        optim.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        images1 = transform_extra(images)
        images2 = transform_extra(images)
        
        out1 = model(images1)
        out2 = model(images2)

        pred1 = torch.argmax(out1, dim=1)
        pred2 = torch.argmax(out2, dim=1)
        
        ce_loss = (cross_entropy_loss(out1, labels) + cross_entropy_loss(out2, labels))/2
        con_loss = ConLoss(out1, out2, labels)
        loss = (ce_loss + con_loss)/2

        train_acc += (torch.sum(torch.where(pred1==labels, 1, 0)) + torch.sum(torch.where(pred2==labels, 1, 0)))/2
        train_loss += loss.item()
        num_train_images += images.shape[0]

        loss.backward()
        optim.step()

    train_acc = train_acc/num_train_images
    # train_loss = train_loss/num_train_images_set

    return train_acc, train_loss, ce_loss, con_loss


def test():

    test_loss = 0
    test_acc = 0
    num_test_images = 0

    for idx,  (images, labels) in enumerate(test_dataloader):
    
        images = images.to(device)
        labels = labels.to(device)

        images1 = transform_extra(images)
        images2 = transform_extra(images)
        
        out1 = model(images1)
        out2 = model(images2)
        
        pred1 = torch.argmax(out1, dim=1)
        pred2 = torch.argmax(out2, dim=1)
        
        ce_loss = (cross_entropy_loss(out1, labels) + cross_entropy_loss(out2, labels))/2
        con_loss = ConLoss(out1, out2, labels)
        loss = (ce_loss + con_loss)/2

        test_acc += (torch.sum(torch.where(pred1==labels, 1, 0)) + torch.sum(torch.where(pred2==labels, 1, 0)))/2
        test_loss += loss.item()
        num_test_images += images.shape[0]

        # num_test_images += images.shape[0]
        # num_test_images_set += 1
    
    test_acc = test_acc/num_test_images
    # test_loss = test_loss/num_test_images_set

    return test_acc, test_loss, ce_loss, con_loss



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = get_args()
    parameters = CreateModelName(args)
    writer = SummaryWriter('./runs/{}/{}/'.format(args.dataset, args.model_name))

    train_dataloader, test_dataloader = CreateDataLoaderlocal(args)  

    model = parameters['model']
    optim_parameters = parameters['optim_parameters']
    num_layers_below = parameters['num_layers_below']
    base_size = parameters['base_size']
    layer_increase = parameters['layer_increase']
    share_pos = parameters['share_pos']
    num_epochs_per_each_block = parameters['num_epochs_per_each_block']
    output_size = parameters['output_size']

    transform_extra = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.RandomVerticalFlip(0.5)])


    model = model.to(device)
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

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

        train_acc, train_loss, ce_loss_train, con_loss_train = train()
        test_acc, test_loss, ce_loss_test, con_loss_test = test()

        if test_loss < test_loss_max:

            test_loss_max = test_loss
            torch.save(model.state_dict(), './weights/{}/{}/model.pth'.format(args.dataset, args.model_name))

            if args.share_weights:
                # Because of warmup
                torch.save({'train_loss': train_loss,
                            'train_acc': train_acc, 
                            'test_loss': test_loss,
                            'test_acc': test_acc,
                            'ce_loss_train': ce_loss_train,
                            'con_loss_train': con_loss_train, 
                            'ce_loss_test': ce_loss_test,
                            'con_loss_test': con_loss_test,
                            'epoch': epoch, 
                            'lr': lr_scheduler._schedulers[-1].get_last_lr()}, './weights/{}/{}/info.pth'.format(args.dataset, args.model_name))
                    
            else:
                # Because of no warmup
                torch.save({'train_loss': train_loss,
                            'train_acc': train_acc, 
                            'test_loss': test_loss,
                            'test_acc': test_acc,
                            'ce_loss_train': ce_loss_train,
                            'con_loss_train': con_loss_train, 
                            'ce_loss_test': ce_loss_test,
                            'con_loss_test': con_loss_test,
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







        if mc.block[mc.cur_pos][-1] == -1:

            # images = torch.cat((images[0], images[1]), dim=0)
            # labels = torch.cat((labels, labels), dim=0)

            # images = images.to(device)
            # labels = labels.to(device)

            # out, features = model(images)



            # _, features1 = model(images1)
            # _, features2 = model(images2)

            # features = torch.stack((features1, features2), dim=1)



            bsz = labels.shape[0]
            images = torch.cat((images[0], images[1]), dim=0)

            images = images.to(device)
            labels = labels.to(device)

            _, f = model(images)

            f1, f2 = torch.split(f, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            ce_loss = torch.tensor(0.).to(device)
            train_acc = torch.tensor(0.).to(device)

            con_loss = contrastive_loss(features, labels)
            
            loss = con_loss
