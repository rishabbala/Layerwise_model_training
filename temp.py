import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, MultiStepLR
import argparse
import os
import math

from resnet import MakeResnet
from helper_functions import create_blockwise_resnet, weight_share, create_model_name
from conf import get_args
from imagenet_loader import ImagenetDataset
from tiny_imagenet_loader import TinyImagenetDataset



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



if __name__ == "__main__":
    optim_parameters = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = get_args()
    create_model_name(args)
    writer = SummaryWriter('./runs/{}/{}/'.format(args.dataset, args.model_name))


    # Defined transforms
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

    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((32, 32)),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm),
                                          transforms.RandomHorizontalFlip(0.5)])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((32, 32)),
                                         transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm)])

    # Set output size depending on the dataset and create dataloader
    if args.dataset == 'cifar10':
        output_size = 10
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)
    elif args.dataset == 'cifar100':
        output_size = 100
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform)
    elif args.dataset == 'imagenet':
        output_size = 1000
        train_dataset = ImagenetDataset(split='train', transforms=train_transform)
        test_dataset = ImagenetDataset(split='val', transforms=test_transform)
    elif args.dataset == 'tiny_imagenet':
        output_size = 200
        train_dataset = TinyImagenetDataset(split='train', transforms=train_transform)
        test_dataset = TinyImagenetDataset(split='val', transforms=test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.model_name == None:
        raise NotImplementedError("Model name not provided")

    # If the whole model is to be trained or a selected set of layers are to be trained together
    if not args.train_solo_layers:

        # Create the last linear layers for the torch model
        if args.use_torch_model == True:
            if output_size < 512:
                num_features_per_lin_layer = int((512-output_size)/args.num_linear_layers)
                lin = [nn.Flatten(start_dim=1)]
                for i in range(args.num_linear_layers-1):
                    lin.append(nn.Linear(in_features=512-i*num_features_per_lin_layer, out_features=512-(i+1)*num_features_per_lin_layer))
                    lin.append(nn.ReLU(inplace=True))
                    final_size = 512-(i+1)*num_features_per_lin_layer
            else:
                num_features_per_lin_layer = int((output_size-512)/args.num_linear_layers)
                lin = [nn.Flatten(start_dim=1)]
                for i in range(args.num_linear_layers-1):
                    lin.append(nn.Linear(in_features=512+i*num_features_per_lin_layer, out_features=512+(i+1)*num_features_per_lin_layer))
                    lin.append(nn.ReLU(inplace=True))
                    final_size = 512+(i+1)*num_features_per_lin_layer

            lin.append(nn.Linear(in_features=final_size, out_features=output_size))
            lin = nn.Sequential(*lin)

        # Resnet18
        if "resnet18" in args.model_name:
            # Create torch model
            if args.use_torch_model == True:
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
                model = torch.nn.Sequential(*(list(model.children())[:-1]))
                model = nn.Sequential(model, lin)
            # Our model
            else:
                model = MakeResnet('resnet18', args.num_linear_layers, output_size, args.dataset)
            if args.share_weights == True:
                raise ValueError("Cannot perform shared weight training for base model")

        # Resnet34
        if "resnet34" in args.model_name:
            # Create torch model
            if args.use_torch_model == True:
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
                model = torch.nn.Sequential(*(list(model.children())[:-1]))
                model = nn.Sequential(model, lin)
                if args.share_weights == True:
                    raise NotImplementedError("Cannot share weights for torch models")
            # Our model
            else:
                model = MakeResnet('resnet34', args.num_linear_layers, output_size, args.dataset)
                if args.share_weights == True:
                    model, optim_parameters = weight_share(args, model, './weights/'+str(args.dataset)+'/resnet18/model.pth')

    # If the blocks are to be trained sequentially
    else:
        if "resnet18" in args.model_name:
            raise NotImplementedError("Solo layer training for base models not implemented")
        elif 'resnet34' in args.model_name:
            # Assuming that the base model is resnet18
            num_layers_below = [8, 6, 4, 2]
            share_pos = -1
            base_size = [3, 2, 2, 2]
            layer_increase = [1, 2, 4, 1]
            if not args.train_upper_layers:
                num_new_layers = 8
            else:
                if not args.epochs_downstream:
                    num_new_layers = 8
                else:
                    print("Downstream")
                    num_train_layers = list(map(lambda a,b: a+b, num_layers_below, layer_increase))
                    num_new_layers = sum(num_train_layers)
            model, optim_parameters = create_blockwise_resnet(args, other_size=base_size, output_size=output_size, weight_share_dir='./weights/'+str(args.dataset)+'/resnet18/model.pth', block_number=0)

    model = model.to(device)

    # Loss and optimizers
    cross_entropy_loss = nn.CrossEntropyLoss()

    if optim_parameters == None or optim_parameters == []:
        optim_parameters = model.parameters()


    # Change the number of epochs for each individual blocks based on the number of layers to train.
    if args.train_solo_layers and args.share_weights:
        num_epochs_for_each_block = []
        temp = 0
        for i in range(len(layer_increase)):
            if not args.epochs_downstream:
                num_epochs_for_each_block.append(math.ceil(layer_increase[i]*args.n_epochs/num_new_layers))
                temp += math.ceil(layer_increase[i]*args.n_epochs/num_new_layers)
            else:
                if not args.train_upper_layers:
                    num_epochs_for_each_block.append(math.ceil(layer_increase[i]*args.n_epochs/num_new_layers))
                    temp += math.ceil(layer_increase[i]*args.n_epochs/num_new_layers)
                else:
                    num_epochs_for_each_block.append(math.ceil(num_train_layers[i]*args.n_epochs/num_new_layers))
                    temp += math.ceil(num_train_layers[i]*args.n_epochs/num_new_layers)

        args.n_epochs = temp

    # Set the optimizer and scheduler for non-blockwise training
    if not args.train_solo_layers:
        if not args.share_weights:
            optim = Adam(optim_parameters, lr=args.lr, weight_decay=1e-5)
            lr_scheduler = MultiStepLR(optim, milestones=[0.3*args.n_epochs, 0.6*args.n_epochs, 0.8*args.n_epochs], gamma=0.2, last_epoch=-1)
        else:
            ## Warmup
            optim = Adam(optim_parameters, lr=args.lr, weight_decay=1e-5)
            lr_scheduler = ChainedScheduler([MultiStepLR(optim, milestones=[0.3*args.n_epochs, 0.6*args.n_epochs, 0.8*args.n_epochs], gamma=0.2, last_epoch=-1),
                                             LinearLR(optim, start_factor=5e-3, end_factor=1, total_iters=0.1*args.n_epochs)])
    model.train()
    test_acc_min = -1e9
    prev_epoch = 0
    flag = 0


    for epoch in range (args.n_epochs):

        # If adding and training layers sequentially, update the optimizer, scheduler, and num epochs accordingly
        if args.train_solo_layers and args.share_weights and (epoch-prev_epoch == 0 or (share_pos != -1 and epoch-prev_epoch >= num_epochs_for_each_block[share_pos])):
            test_acc_min = -1e9
            prev_epoch = epoch
            share_pos += 1

            if share_pos > 0:
                flag = 0
                base_size[share_pos] += layer_increase[share_pos]
                model, optim_parameters = create_blockwise_resnet(args, other_size=base_size, output_size=output_size, weight_share_dir='./weights/{}/{}/model.pth'.format(args.dataset, args.model_name), block_number=share_pos)
                model.to(device)
            
            optim = Adam(optim_parameters, lr=args.lr, weight_decay=1e-5)
            # Warmup
            lr_scheduler = ChainedScheduler([MultiStepLR(optim, milestones=[int(0.3*num_epochs_for_each_block[share_pos]), int(0.6*num_epochs_for_each_block[share_pos]), int(0.8*num_epochs_for_each_block[share_pos])], gamma=0.2, last_epoch=-1),
                                             LinearLR(optim, start_factor=5e-3, end_factor=1, total_iters=0.2*num_epochs_for_each_block[share_pos])])

        train_acc, train_loss = train()
        
        model.eval()
        test_acc, test_loss = test()
        model.train()

        if test_acc > test_acc_min:

            test_acc_min = test_acc

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





import torch
import os
from resnet import MakeResnet
import shutil



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



def create_model_name(args):

    # Set the name of the model for storing
    if args.use_torch_model == True:
        args.model_name += '_torch'
    if args.share_weights == True:
        args.model_name += '_share_weights'
    if args.block_grad == True:
        args.model_name += '_block_grad'
    if args.train_solo_layers == True:
        args.model_name += '_train_solo_layers'
    if args.train_upper_layers == False:
        args.model_name += '_freeze_upper_layers'
    if args.train_lin_layers == False:
        args.model_name += '_freeze_lin_layer'
    if args.epochs_downstream:
        args.model_name += '_epoch_downstream'


    ### Remove previous tfevents Create directory for weights
    if os.path.isdir('./runs/{}/{}'.format(args.dataset, args.model_name)):
        shutil.rmtree('./runs/{}/{}'.format(args.dataset, args.model_name))
        os.makedirs('./runs/{}/{}'.format(args.dataset, args.model_name))
    if not os.path.isdir('./weights/{}/{}'.format(args.dataset, args.model_name)):
        os.makedirs('./weights/{}/{}'.format(args.dataset, args.model_name))



def create_blockwise_resnet(args, other_size, output_size, weight_share_dir, block_number):

    """ 
    If training blockwise, we build the model block by block and train them seperately

    input: 
            other_size: a list containing the size of each block in the resnet -> list
            weight_share_dir: the directory where weights of smaller model or the one without the current block is present -> str
    output:
            model: the model to train -> torch model
            optim_parameters: subset of model parameters to optimize -> list
    """

    model = MakeResnet('other', args.num_linear_layers, output_size, args. dataset, other_size=other_size)
    model, optim_parameters = weight_share(args, model, weight_share_dir, block_number)

    return model, optim_parameters



def weight_share(args, model, child_weight_path, block_number=None):

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

    optim_parameters = []
    linear_layer_values = []

    sd = model.state_dict()
    try:
        child_sd = torch.load(child_weight_path)
        sd.update(child_sd)
    except:
        raise ValueError("Base model not trained yet")
    model.load_state_dict(sd)

    ## Initialize just the weights and bias ONLY FOR RESNETS, MAKE SURE TO CHANGE FOR OTHERS. bias comes after weight so change just the layer jsut after weight
    for key, values in model.named_parameters():
        if key not in child_sd.keys():
            # weight of conv layer
            if 'weight' in key and len(list(values.shape)) > 1:
                kernel = torch.zeros(values.shape[2], values.shape[3])
                # kernel[int(values.shape[2]/2), int(values.shape[3]/2)] = 1
                kernel.unsqueeze(0).unsqueeze(0)
                kernel = torch.tile(kernel, (values.shape[0], values.shape[1], 1, 1))
                values.data = kernel

            # bias is 0, weight of batchnorm is 1
            else:
                if 'weight' in key:
                    values.data = torch.ones(values.shape)
                else:
                    values.data = torch.zeros(values.shape)


    # Choice of whether to train the batchnorm and linear layers
    if args.train_lin_layers == False:
        func = lambda key: False if key in child_sd else True
    elif args.train_lin_layers == True:
        func = lambda key: False if key in child_sd and 'linear_layers' not in key else True

    # The parameters to be trained are chosen and the others have their requires_grad set to False. Make sure downstream layers are are added to optimizer. This is done using the model nomenclature. We know that layer1 is 2 convs for resnet, which is always frozen. Layer 2 is the first block. So we check from layer 2 onwards. print(model.named_parameters()) to verify
    if args.block_grad:
        for (key, values) in model.named_parameters():
            values.requires_grad = func(key)
            txt = key.split(".")
            try:
                t = int(txt[1][-1])
                if (t > block_number+2 or values.requires_grad == True) and not args.train_upper_layers:
                    optim_parameters.append(values)
                elif t > block_number+2 and args.train_upper_layers:
                    values.requires_grad = True
                    optim_parameters.append(values)
            except:
                if values.requires_grad == True:
                    optim_parameters.append(values)
    else:
        optim_parameters = model.parameters()
        
    return model, optim_parameters
