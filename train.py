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
from timm.utils import ApexScaler, NativeScaler



def train():
    '''
    train one epoch:
    model: taken from main()
    '''


    model.train()

    train_loss = 0
    train_acc = 0
    num_train_images = 0

    for idx, (images, labels) in enumerate(train_dataloader):

        optim.zero_grad()

        out = model(images)
        loss = soft_cross_entropy_loss(out, labels)
        pred = torch.argmax(out, dim=1)

        labels_max = torch.argmax(labels, dim=1)
        train_acc += torch.sum(torch.where(pred==labels_max, 1, 0))

        loss.backward()
        optim.step()

        train_loss += loss.item() * images.shape[0]
        num_train_images += images.shape[0]

    train_acc = train_acc/num_train_images
    train_loss = train_loss/num_train_images

    return train_acc, train_loss


def test():
    """
    Evaluate One Epoch
    """

    test_loss = 0
    test_acc = 0
    num_test_images = 0

    model.eval()
    with torch.no_grad():
        for _,  (images, labels) in enumerate(test_dataloader):

            out = model(images)
            loss = cross_entropy_loss(out, labels)

            pred = torch.argmax(out, dim=1)
            test_acc += torch.sum(torch.where(pred==labels, 1, 0))

            test_loss += loss.item() * images.shape[0]
            num_test_images += images.shape[0]

        test_acc = test_acc/num_test_images
        test_loss = test_loss/num_test_images

    return test_acc, test_loss


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
    train_dataloader, test_dataloader = mc.CreateDataLoader()

    soft_cross_entropy_loss = SoftTargetCrossEntropy().cuda()
    cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()

    test_acc_min = -1e9
    prev_epoch = 0
    optim = None
    lr_scheduler = None
    # optim_scheduler = OptimScheduler(args, len(mc.block))

    for epoch in range (args.n_epochs):
        
        # if optim_scheduler.change:
        if epoch%mc.num_epochs_per_block == 0 and mc.cur_pos != len(mc.block)-1:
            test_acc_min = -1e9
            prev_epoch = epoch

            model, optim = mc.create_custom_model(optim)

            if not args.combined:
                lr_scheduler, _ = create_scheduler_v2(optimizer=optim, sched='cosine', num_epochs=args.n_epochs, min_lr=1e-4, warmup_lr=1e-6, warmup_epochs=10, cooldown_epochs=10, patience_epochs=10, warmup_prefix=False, noise=None)

            # optim_scheduler.change = False
            for key, value in model.named_parameters():
                print(key, value.requires_grad) 

            model.to(device)

        if lr_scheduler is None:
            for opt in optim.param_groups:
                e = (epoch%mc.num_epochs_per_block)-5
                n = args.n_epochs - (len(mc.block)-1) * mc.num_epochs_per_block - 5
                n2 = epoch - (len(mc.block)-1) * mc.num_epochs_per_block - 5

                if epoch%mc.num_epochs_per_block <= 5 and mc.cur_pos != len(mc.block)-1:
                    opt['lr'] = 2e-4 * (epoch%mc.num_epochs_per_block) / 5 + 4e-4
                elif mc.cur_pos != len(mc.block)-1:
                    opt['lr'] = 4e-4 + 0.5 * (2e-4) * (1 + math.cos(e*math.pi/(mc.num_epochs_per_block-5)))
                else:
                    opt['lr'] = 1e-4 + 0.5 * (5e-4) * (1 + math.cos(n2*math.pi/n))
                    
        train_acc, train_loss = train()
        test_acc, test_loss = test()

        if test_acc > test_acc_min:

            test_acc_min = test_acc
            torch.save(model.state_dict(), './weights/{}/{}/model.pth'.format(args.dataset, args.model_name))

            # Because of warmup
            torch.save({'train_loss': train_loss,
                        'train_acc': train_acc, 
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'epoch': epoch}, './weights/{}/{}/info.pth'.format(args.dataset, args.model_name)) #'lr': lr_scheduler.get_last_lr()[0]

        print("Epoch: {}/{}, model: {}, lr:{}".format(epoch, args.n_epochs, args.model_name, optim.param_groups[0]['lr']))

        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Acc', train_acc, epoch)
        writer.add_scalar('Val Loss', test_loss, epoch)
        writer.add_scalar('Val Acc', test_acc, epoch)
        writer.add_scalar('Learning Rate', optim.param_groups[0]['lr'], epoch)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch+1, test_loss)

        # lr_scheduler.step()

        # acc_to_use = test_acc
        
        # # if epoch-prev_epoch>args.warmup_epochs:
        # optim_scheduler.update(acc_to_use, optim)

        # # if optim_scheduler.change:
        # #     torch.save(model.state_dict(), './weights/{}/{}/model_{}.pth'.format(args.dataset, args.model_name, optim_scheduler.pos))