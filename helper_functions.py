import torch
import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from resnet import MakeResnet
import shutil
from create_model import CreateTorchModel, CreateCustomModel, CreateBlockwiseResnet, WeightShare, SplitEpochs
from imagenet_loader import ImagenetDataset
from tiny_imagenet_loader import TinyImagenetDataset



def CreateModelName(args):

    if args.model_name == None:
        raise NotImplementedError("Model name not provided")

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

    args.model_name += '_trial'

    # Set output size depending on the dataset and create dataloader
    if args.dataset == 'cifar10':
        output_size = 10
    elif args.dataset == 'cifar100':
        output_size = 100 
    elif args.dataset == 'imagenet':
        output_size = 1000
    elif args.dataset == 'tiny_imagenet':
        output_size = 200


    ### Remove previous tfevents Create directory for weights
    if os.path.isdir('./runs/{}/{}'.format(args.dataset, args.model_name)):
        shutil.rmtree('./runs/{}/{}'.format(args.dataset, args.model_name))
        os.makedirs('./runs/{}/{}'.format(args.dataset, args.model_name))
    if not os.path.isdir('./weights/{}/{}'.format(args.dataset, args.model_name)):
        os.makedirs('./weights/{}/{}'.format(args.dataset, args.model_name))

    if args.use_torch_model:
        parameters = CreateTorchModel(args, output_size)
    else:
        parameters = CreateCustomModel(args, output_size)

    parameters['output_size'] = output_size

    return parameters



def CreateDataLoader(args):

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

    # Defined transforms
    if args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)

    if args.dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform)

    if args.dataset == "imagenet":
        train_dataset = ImagenetDataset(split='train', transforms=train_transform)
        test_dataset = ImagenetDataset(split='val', transforms=test_transform)

    # must recompute imagenet values for now
    if args.dataset == "tiny_imagenet":
        train_dataset = TinyImagenetDataset(split='train', transforms=train_transform)
        test_dataset = TinyImagenetDataset(split='val', transforms=test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, test_dataloader