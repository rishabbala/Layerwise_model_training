import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adadelta, SGD, Adam
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, MultiStepLR
import argparse
import os
import math
import shutil
import copy

from helper_functions import CreateModelName, CreateDataLoader
from imagenet_loader import ImagenetDataset
from tiny_imagenet_loader import TinyImagenetDataset




def str2bool(v):
    
    """ 
    Function to convert input string from argparse to bool
    
    input: v -> string
    output: v -> bool
    """
    
    if isinstance(v, bool):
        return v
    elif v.lower() == 'true':
        return True
    else:
        return False




class MakeResnet(nn.Module):


    def __init__(self, model_name, num_linear_layers, output_size, dataset, other_size=None):

        super().__init__()
        self.resnet = ResnetBasic(num_layers=other_size, num_linear_layers=2, output_size=output_size, dataset=dataset)
    
    def forward(self, x):
        
        x = self.resnet(x)

        return x



class ResnetBasic(nn.Module):


    def __init__(self, num_layers, num_linear_layers, output_size, dataset):
        
        super().__init__()

        # if 'cifar' in dataset:
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU(inplace=True))
                            
        # else:
        #     self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
        #                                 nn.BatchNorm2d(num_features=64), 
        #                                 nn.ReLU(inplace=True),
        #                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer0 = self._make_layer(num_layers[0], in_channels=64, out_channels=64, padding=1, stride=1)
        self.layer1 = self._make_layer(num_layers[1], in_channels=64, out_channels=128, padding=1, stride=2)
        self.layer2 = self._make_layer(num_layers[2], in_channels=128, out_channels=256, padding=1, stride=2)
        self.layer3 = self._make_layer(num_layers[3], in_channels=256, out_channels=512, padding=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_layers = []

        num_features_per_lin_layer = abs(int((512-output_size)/num_linear_layers))
        sign = int((512-output_size)/abs(512-output_size))
        for i in range(num_linear_layers-1):
            self.linear_layers.append(nn.Linear(in_features=512-sign*i*num_features_per_lin_layer, out_features=512-sign*(i+1)*num_features_per_lin_layer))
            self.linear_layers.append(nn.ReLU(inplace=True))
            final_size = 512-sign*(i+1)*num_features_per_lin_layer

        self.linear_layers.append(nn.Linear(in_features=final_size, out_features=output_size))
        self.linear_layers = nn.Sequential(*self.linear_layers)



    def _make_layer(self, n_layer, in_channels, out_channels, padding, stride):

        layer = []
        stride_layer = []
        channel_layer = []

        for _ in range(n_layer-1):
            stride_layer.append(1)
            channel_layer.append(out_channels)

        stride_layer.insert(0, stride)
        channel_layer.insert(0, in_channels)

        for i in range(n_layer):
            layer.append(Block(channel_layer[i], out_channels, padding=padding, stride=stride_layer[i]))

        return nn.Sequential(*layer)


    def forward(self, x):

        x = self.encoder(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x).view(x.shape[0], x.shape[1])
        x = self.linear_layers(x)

        return x



class Block(nn.Module):


    def __init__(self, in_channels, out_channels, padding, stride):
        
        super().__init__()

        self.downsample = None
        self.relu = nn.ReLU(inplace=True)

        self.block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding, stride=stride),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(num_features=out_channels)
                                    )

        if stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(num_features=out_channels))

    
    def forward(self, x):

        y = self.block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        y = x + y
        y = self.relu(y)

        return y


def WeightShare(args, model, child_weight_path=None, block_number=None, optim=None, optim_params0_keys=[]):

    """ 
    Function to share weights between smaller and larger models

    input:
            args: arguments passed to the program -> from argparse
            model: the model to be trained -> torch model
            child_weight_path: dir of weights of a smaller model -> str
            block_number: the block number to which new layers are added so we can freeze the layers above it and unfreeze those below it -> int

    output:
            model: the model to be trained -> torch model
            optim_parameters: subset of model parameters to optimize -> list
    """

    ## Load all previous layers with weight from smaller model
    sd = model.state_dict()
    try:
        child_sd = torch.load(child_weight_path)
        sd.update(child_sd)
    except:
        raise ValueError("Base model not trained yet")
    model.load_state_dict(sd)


    ## to ensure parms are in right order. After block_num we have enc, l11, l21, l31, l41, lin. Then after block_num2 we have enc, l11, l21, l31, l41, lin, l12, l22, l32, l42. To ensure this order, we have to extend the optim params0 before adding new ones to optim_params 1. This should match order of param exp values from Adam optimizer

    optim_params0 = []
    optim_params1 = []

    for key in optim_params0_keys:
        optim_params0.append(model.state_dict()[key])

    model.to(device)
    ## Initialize just the weights and bias ONLY FOR RESNETS, MAKE SURE TO CHANGE FOR OTHERS. bias comes after weight so change just the layer jsut after weight
    for key, values in model.named_parameters():
        if key in child_sd.keys() and key not in optim_params0_keys:
            optim_params0_keys.append(key)
            optim_params0.append(values)
        
        elif key not in child_sd.keys():
            if 'downsample' in key:
                pass

            elif ('weight' in key and len(list(values.shape)) == 1) or 'bias' in key:
                values.data = torch.zeros(values.shape).to(device)

            optim_params1.append(values)

    # # Dont train other layers except newly added
    # if not args.train_up_layers or not args.train_low_layers:
    #     for (key, values) in model.named_parameters():
    #         txt = key.split(".")
    #         try:
    #             t = int(txt[1][-1])
    #             if not args.train_low_layers and t < block_number:
    #                 values.requires_grad = False
    #             elif not args.train_up_layers and t > block_number:
    #                 values.requires_grad = False
    #         except:
    #             if 'encoder' in key and not args.train_low_layers:
    #                 values.requires_grad = False
    #             else:
    #                 values.requires_grad = True

    new_optim = Adam([{'params': optim_params0}, 
                      {'params': optim_params1}
                     ], lr=args.lr, weight_decay=1e-5)
    
    optim_sd = optim.state_dict()
    sd = new_optim.state_dict()
    
    for key in optim_sd['state'].keys():
        sd['state'][key] = optim_sd['state'][key]

    # ## Newly added params
    # pos = 0
    # for key in sd['param_groups'][1]['params']:
    #     ## same size as the values tensor
    #     t = {'step': step, 'square_avg': torch.zeros(optim_params1[pos].shape).to(device), 'acc_delta': torch.zeros(optim_params1[pos].shape).to(device)}
    #     sd['state'][key] = t
    #     pos += 1

    new_optim.load_state_dict(sd)
    del optim

    print(new_optim.state_dict()['param_groups'])
                
    return model, new_optim, optim_params0_keys



def train():

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

    train_acc = train_acc/num_train_images
    train_loss = train_loss/num_train_images

    return train_acc, train_loss
    


def test():

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


class Scheduler():

    def __init__(self, args, max_pos=None):
        self.window = 5
        self.test_acc_list = []

        self.acc_swing = []

        self.smoothened_acc_swing = 0
        self.avg_acc_swing = 0

        self.acc_swing_thresh = 0.01 ## 1%
        self.min_lr = 1e-5
        self.change = True
        self.max_pos = mp
        self.pos = 0

        self.args = args

    def update(self, test_acc, optim):

        self.change = False

        if len(self.test_acc_list) >= self.window:
            self.test_acc_list.pop(0)
            self.acc_swing.pop(0)
        
        self.test_acc_list.append(test_acc)

        if len(self.test_acc_list) > 1:
            self.acc_swing.append(self.test_acc_list[-1]-self.test_acc_list[-2])

        if len(self.test_acc_list) == self.window and abs(sum(self.acc_swing)) < self.acc_swing_thresh and abs(abs(sum(self.acc_swing))-sum(list(map(abs, self.acc_swing)))) >self.acc_swing_thresh:

            for op in optim.param_groups:
                op['lr'] /= 5

                if op['lr'] <= self.min_lr:
                    self.change = True
                    self.acc_swing_thresh = 0.01 ## 1%
                    print("Finished")

                elif op['lr'] < 5e-4:
                    if self.pos != self.max_pos-1:
                        self.change = True
                        self.acc_swing_thresh = 0.01 ## 1%
                        print("Changing")
                    else:
                        self.acc_swing_thresh = 0.005 ## 0.5%
                self.test_acc_list = []
                self.acc_swing = []
                

            print("Changeing LR", optim.state_dict()['param_groups'][0]['lr'])

        if self.change:
            self.pos += 1
            
        return optim



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    parser = argparse.ArgumentParser(description='Arguements for training')

    # Input parameters
    parser.add_argument('--model_name', default='sandwich_resnet18')
    parser.add_argument('--num_linear_layers', default=2, type=int)
    # parser.add_argument('--train_low_layers', default=True, type=str2bool)
    # parser.add_argument('--train_up_layers', default=True, type=str2bool)
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    args = parser.parse_args()

    # if args.train_up_layers:
    #     args.model_name += '_up'
    # if args.train_low_layers:
    #     args.model_name += '_low'

    writer = SummaryWriter('./runs/{}/{}/'.format(args.dataset, args.model_name))

    if os.path.isdir('./runs/{}/{}'.format(args.dataset, args.model_name)):
        shutil.rmtree('./runs/{}/{}'.format(args.dataset, args.model_name))
        os.makedirs('./runs/{}/{}'.format(args.dataset, args.model_name))
    if not os.path.isdir('./weights/{}/{}'.format(args.dataset, args.model_name)):
        os.makedirs('./weights/{}/{}'.format(args.dataset, args.model_name))

    train_dataloader, test_dataloader = CreateDataLoader(args)

    if args.dataset == 'cifar10':
        output_size = 10
    elif args.dataset == 'cifar100':
        output_size = 100 
    elif args.dataset == 'imagenet':
        output_size = 1000
    elif args.dataset == 'tiny_imagenet':
        output_size = 200

    cross_entropy_loss = nn.CrossEntropyLoss()

    test_acc_min = -1e9
    prev_epoch = 0

    pos = -1
    if 'resnet18' in args.model_name:
        mp = 2
        num_epochs_per_each_block = [args.n_epochs/2, args.n_epochs/2]
    elif 'resnet34' in args.model_name:
        mp = 5
        num_epochs_per_each_block = [args.n_epochs/5, args.n_epochs/5, args.n_epochs/5, args.n_epochs/5, args.n_epochs/5]

    scheduler = Scheduler(args, max_pos=mp)
    warmup_epochs = 2

    optim_params0_keys = []


    for epoch in range (args.n_epochs):

        # If adding and training layers sequentially, update the optimizer, scheduler, and num epochs accordingly
        if scheduler.change:
            test_acc_min = -1e9
            prev_epoch = epoch
            pos += 1

            if 'resnet18' in args.model_name:
                if pos == 0:
                    other_size = [1, 1, 1, 1]
                elif pos == 1:
                    other_size = [2, 2, 2, 2]
                else:
                    print("Completed training")
                    exit()

            elif 'resnet34' in args.model_name:
                if pos == 0:
                    other_size = [1, 1, 1, 1]
                elif pos == 1:
                    other_size = [2, 2, 2, 2]
                elif pos == 2:
                    other_size = [3, 3, 3, 3]
                elif pos == 3:
                    other_size = [3, 4, 4, 3]
                elif pos == 4:
                    other_size = [3, 4, 6, 3]
                else:
                    print("Completed training")
                    exit()

            scheduler.change = False

            model = MakeResnet(args.model_name+'_'+str(pos), args.num_linear_layers, output_size, args.dataset, other_size=other_size)

            if pos == 0:
                optim = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
                lr_scheduler = LinearLR(optim, start_factor=1, end_factor=1, total_iters=warmup_epochs)
            else:
                model, optim, optim_params0_keys = WeightShare(args, model, child_weight_path='./weights/{}/{}/model.pth'.format(args.dataset, args.model_name), block_number=pos, optim=optim, optim_params0_keys=optim_params0_keys)

                lr_scheduler = LinearLR(optim, start_factor=0.05, end_factor=1, total_iters=warmup_epochs)
           
            model.to(device)
            
            for key, value in model.named_parameters():
                print(key, value.requires_grad)
            
        train_acc, train_loss = train()
        
        model.eval()
        test_acc, test_loss = test()
        model.train()

        if test_acc > test_acc_min:

            test_acc_min = test_acc
            torch.save(model.state_dict(), './weights/{}/{}/model.pth'.format(args.dataset, args.model_name))

            # Because of warmup
            torch.save({'train_loss': train_loss,
                        'train_acc': train_acc, 
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'epoch': epoch}, './weights/{}/{}/info.pth'.format(args.dataset, args.model_name)) #'lr': lr_scheduler.get_last_lr()[0]

        print("Epoch: {}/{}, model: {}, lr:{}".format(epoch, args.n_epochs, args.model_name, optim.state_dict()['param_groups'][0]['lr']))  # , lr: {}, {} , lr_scheduler.get_last_lr()[0], lr_scheduler.get_last_lr()

        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Acc', train_acc, epoch)
        writer.add_scalar('Val Loss', test_loss, epoch)
        writer.add_scalar('Val Acc', test_acc, epoch)

        lr_scheduler.step()

        if epoch-prev_epoch>warmup_epochs or pos == 0:
            optim = scheduler.update(test_acc, optim)