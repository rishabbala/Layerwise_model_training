import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import argparse
import os
import math
import argparse
import shutil



class ResnetBasic(nn.Module):


    def __init__(self, num_layers, num_linear_layers, output_size, dataset, start_block):
        
        super().__init__()

        self.start_block = start_block
        self.linear_layers = []
        if output_size < 512:
            num_features_per_lin_layer = int((512-output_size)/num_linear_layers)
            for i in range(num_linear_layers-1):
                self.linear_layers.append(nn.Linear(in_features=512-i*num_features_per_lin_layer, out_features=512-(i+1)*num_features_per_lin_layer))
                self.linear_layers.append(nn.ReLU(inplace=True))
                final_size = 512-(i+1)*num_features_per_lin_layer

        else:
            num_features_per_lin_layer = int((output_size-512)/num_linear_layers)
            for i in range(num_linear_layers-1):
                self.linear_layers.append(nn.Linear(in_features=512+i*num_features_per_lin_layer, out_features=512+(i+1)*num_features_per_lin_layer))
                self.linear_layers.append(nn.ReLU(inplace=True))
                final_size = 512+(i+1)*num_features_per_lin_layer

        if self.start_block == 1:
            # if 'cifar' in dataset:
            self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features=64),
                                        nn.ReLU(inplace=True))
                                
            # else:
            #     self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            #                                 nn.BatchNorm2d(num_features=64), 
            #                                 nn.ReLU(inplace=True),
            #                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            self.layer2 = self._make_layer(num_layers[0], in_channels=64, out_channels=64, padding=1, stride=1)
            self.layer3 = self._make_layer(num_layers[1], in_channels=64, out_channels=128, padding=1, stride=2)
            self.layer4 = self._make_layer(num_layers[2], in_channels=128, out_channels=256, padding=1, stride=2)
            self.layer5 = self._make_layer(num_layers[3], in_channels=256, out_channels=512, padding=1, stride=2)

        elif self.start_block == 3:            
            self.layer3 = self._make_layer(num_layers[0], in_channels=64, out_channels=128, padding=1, stride=2)
            self.layer4 = self._make_layer(num_layers[1], in_channels=128, out_channels=256, padding=1, stride=2)
            self.layer5 = self._make_layer(num_layers[2], in_channels=256, out_channels=512, padding=1, stride=2)


        elif self.start_block == 4:            
            self.layer4 = self._make_layer(num_layers[0], in_channels=128, out_channels=256, padding=1, stride=2)
            self.layer5 = self._make_layer(num_layers[1], in_channels=256, out_channels=512, padding=1, stride=2)

        elif self.start_block == 5:            
            self.layer5 = self._make_layer(num_layers[0], in_channels=256, out_channels=512, padding=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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

        if self.start_block == 1:
            x = self.layer1(x)
            x = self.layer2(x)
            if self.start_block == 1:
                features = x
        if self.start_block <= 3:
            x = self.layer3(x)
            if self.start_block == 3:
                features = x
        if self.start_block <= 4:
            x = self.layer4(x)
            if self.start_block == 4:
                features = x
        if self.start_block <= 5:
            x = self.layer5(x)
            if self.start_block == 5:
                features = x

        x = self.avgpool(x).view(x.shape[0], x.shape[1])
        x = self.linear_layers(x)

        return x, features



class Block(nn.Module):


    def __init__(self, in_channels, out_channels, padding, stride):
        
        super().__init__()

        self.downsample = None
        self.relu = nn.ReLU(inplace=True)

        self.block1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding, stride=stride),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(num_features=out_channels)
                                    )

        if stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(num_features=out_channels))

    
    def forward(self, x):

        y = self.block1(x)

        if self.downsample is not None:
            x = self.downsample(x)

        y = x + y
        y = self.relu(y)

        return y



class LoadSavedFeatures(Dataset):


    def __init__(self, file_pos, dataset, model_name, train=True):
        if train:
            self.data = torch.load('./weights/{}/{}/save_output{}.pth'.format(dataset, model_name, file_pos))
            self.labels = torch.load('./weights/{}/{}/save_labels{}.pth'.format(dataset, model_name, file_pos))
        else:
            self.data = torch.load('./weights/{}/{}/save_output_test{}.pth'.format(dataset, model_name, file_pos))
            self.labels = torch.load('./weights/{}/{}/save_labels_test{}.pth'.format(dataset, model_name, file_pos))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        img = self.data[idx]
        label = self.labels[idx]

        return img, label



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description='Arguements for training')

    # Input parameters
    parser.add_argument('--dataset', default='cifar100')
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()

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

    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.Normalize(mean=mean_dataset_norm, std=std_dataset_norm),
                                          transforms.RandomHorizontalFlip(0.5)])

    test_transform = transforms.Compose([transforms.ToTensor(),
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

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ResnetBasic(num_layers=[3, 2, 2, 2], num_linear_layers=2, output_size=output_size, dataset=args.dataset, start_block=1).to(device)

    if args.dataset == 'cifar10':
        resnet18_state_dict = torch.load('./weights/cifar10/resnet18/model.pth')
    elif args.dataset == 'cifar100':
        resnet18_state_dict = torch.load('./weights/cifar100/resnet18/model.pth')
    for key, value in model.named_parameters():
        if 'resnet.'+key in resnet18_state_dict:
            value.data = torch.clone(resnet18_state_dict['resnet.'+key].data).to(device)

    del resnet18_state_dict

    model_name = 'multi_resnet'
    if os.path.isdir('./runs/{}/{}'.format(args.dataset, model_name)):
        shutil.rmtree('./runs/{}/{}'.format(args.dataset, model_name))
        os.makedirs('./runs/{}/{}'.format(args.dataset, model_name))
    if not os.path.isdir('./weights/{}/{}'.format(args.dataset, model_name)):
        os.makedirs('./weights/{}/{}'.format(args.dataset, model_name))

    writer = SummaryWriter('./runs/{}/{}/'.format(args.dataset, model_name))

    num_new_layers = 8
    layer_increase = [1, 2, 4, 1]
    num_epochs_for_each_block = []
    temp = 0

    # for i in range(len(layer_increase)):
    #     num_epochs_for_each_block.append(math.ceil(layer_increase[i]*args.n_epochs/num_new_layers))
    #     temp += math.ceil(layer_increase[i]*args.n_epochs/num_new_layers)

    num_epochs_for_each_block = [15, 45, 100, 45]
 
    args.n_epochs = 205

    optim = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    lr_scheduler = MultiStepLR(optim, milestones=[int(0.3*num_epochs_for_each_block[0]), int(0.6*num_epochs_for_each_block[0]), int(0.8*num_epochs_for_each_block[0])], gamma=0.2, last_epoch=-1)

    cross_entropy_loss = nn.CrossEntropyLoss()

    test_acc_min = -1e9
    prev_epoch = 0
    flag = 0
    pos = 0


    for epoch in range (args.n_epochs):

        save_output = []
        save_labels = []
        save_output_test = []
        save_labels_test = []

        if epoch-prev_epoch >= num_epochs_for_each_block[pos]:
            test_acc_min = -1e9
            pos += 1
            prev_epoch = epoch

            train_dataset = LoadSavedFeatures(file_pos=pos-1, dataset=args.dataset, model_name=model_name, train=True)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            test_dataset = LoadSavedFeatures(file_pos=pos-1, dataset=args.dataset, model_name=model_name, train=False)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            del model
            del optim
            del lr_scheduler

            if pos == 1:
                model = ResnetBasic(num_layers=[4, 2, 2], num_linear_layers=2, output_size=output_size, dataset=args.dataset, start_block=3).to(device)
            elif pos == 2:
                model = ResnetBasic(num_layers=[6, 2], num_linear_layers=2, output_size=output_size, dataset=args.dataset, start_block=4).to(device)
            elif pos == 3:
                model = ResnetBasic(num_layers=[3], num_linear_layers=2, output_size=output_size, dataset=args.dataset, start_block=5).to(device)

            child_state_dict = torch.load('./weights/{}/{}/resnet_{}.pth'.format(args.dataset, model_name, pos-1))
            for key, value in model.named_parameters():
                if key in child_state_dict:
                    value.data = torch.clone(child_state_dict[key].data).to(device)
            
            del child_state_dict

            optim = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
            lr_scheduler = MultiStepLR(optim, milestones=[int(0.3*num_epochs_for_each_block[0]), int(0.6*num_epochs_for_each_block[0]), int(0.8*num_epochs_for_each_block[0])], gamma=0.2, last_epoch=-1)

        train_loss = 0
        train_acc = 0
        num_train_images = 0
        for idx, (images, labels) in enumerate(train_dataloader):
            
            images = images.to(device)
            labels = labels.to(device)
            
            optim.zero_grad()
            out, features = model(images)

            pred = torch.argmax(out, dim=1)
            save_output.append(features.detach().cpu())
            save_labels.append(labels.detach().cpu())

            loss = cross_entropy_loss(out, labels)
            train_acc += torch.sum(torch.where(pred==labels, 1, 0))
            train_loss += loss.item()
            num_train_images += images.shape[0]

            loss.backward()
            optim.step()

        test_loss = 0
        test_acc = 0
        num_test_images = 0
        for _,  (images, labels) in enumerate(test_dataloader):
    
            images = images.to(device)
            labels = labels.to(device)
        
            out, features = model(images)
            pred = torch.argmax(out, dim=1)

            save_output_test.append(features.detach().cpu())
            save_labels_test.append(labels.detach().cpu())

            loss = cross_entropy_loss(out, labels)
            test_loss += loss.item()
            test_acc += torch.sum(torch.where(pred==labels, 1, 0))
            num_test_images += images.shape[0]
            
        test_acc = test_acc/num_test_images
        test_loss = test_loss/num_test_images

        train_acc = train_acc/num_train_images
        train_loss = train_loss/num_train_images

        if test_acc > test_acc_min:
            test_acc_min = test_acc
            save_output = torch.cat(save_output)
            save_labels = torch.cat(save_labels)
            save_output_test = torch.cat(save_output_test)
            save_labels_test = torch.cat(save_labels_test)

            print("Saving", save_output.shape, save_output_test.shape)

            torch.save(model.state_dict(), './weights/{}/{}/resnet_{}.pth'.format(args.dataset, model_name, pos))
            torch.save(save_output, './weights/{}/{}/save_output{}.pth'.format(args.dataset, model_name, pos))
            torch.save(save_labels, './weights/{}/{}/save_labels{}.pth'.format(args.dataset, model_name, pos))
            torch.save(save_output_test, './weights/{}/{}/save_output_test{}.pth'.format(args.dataset, model_name, pos))
            torch.save(save_labels_test, './weights/{}/{}/save_labels_test{}.pth'.format(args.dataset, model_name, pos))
        
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Acc', train_acc, epoch)
        writer.add_scalar('Val Loss', test_loss, epoch)
        writer.add_scalar('Val Acc', test_acc, epoch)

        lr_scheduler.step()
        print("Epoch: {}/{}, model: {}, lr:{}".format(epoch, args.n_epochs, model_name, lr_scheduler.get_last_lr()))