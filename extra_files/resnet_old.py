import torch
import torch.nn as nn
from collections import OrderedDict



class MakeResnet(nn.Module):


    def __init__(self, num_linear_layers, output_size, dataset, block_size=None):

        super().__init__()
        self.resnet = ResnetBasic(num_layers=block_size, num_linear_layers=num_linear_layers, output_size=output_size, dataset=dataset)
    
    def forward(self, x):
        
        z, x = self.resnet(x)

        return z, x



class ResnetBasic(nn.Module):


    def __init__(self, num_layers, num_linear_layers, output_size, dataset):
        
        super().__init__()

        # if 'cifar' in dataset:
        self.encoder = nn.Sequential(OrderedDict([
                                    ('conv', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)),
                                    ('norm', nn.BatchNorm2d(num_features=64)),
                                    ('relu', nn.ReLU(inplace=True))
                                    ]))
                            
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

        # self.linear_layers = []

        # num_features_per_lin_layer = abs(int((512-output_size)/num_linear_layers))
        # sign = int((512-output_size)/abs(512-output_size))
        # for i in range(num_linear_layers-1):
        #     self.linear_layers.append(nn.Linear(in_features=512-sign*i*num_features_per_lin_layer, out_features=512-sign*(i+1)*num_features_per_lin_layer))
        #     self.linear_layers.append(nn.ReLU(inplace=True))
        #     final_size = 512-sign*(i+1)*num_features_per_lin_layer

        # self.linear_layers.append(nn.Linear(in_features=final_size, out_features=output_size))
        # self.linear_layers = nn.Sequential(*self.linear_layers)

        self.linear_layers = nn.Sequential(OrderedDict([
                                    ('linear1', nn.Linear(in_features=512, out_features=256)),
                                    ('relu1', nn.ReLU(inplace=True)),
                                    ('linear2', nn.Linear(in_features=256, out_features=128)),
                                    ('relu2', nn.ReLU(inplace=True))
                                    ]))

        if num_layers[4] != -1:
            self.final_linear_layers = nn.Linear(in_features=128, out_features=output_size)
        else:
            self.final_linear_layers = None



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

        if self.final_linear_layers is not None:
            z = self.final_linear_layers(x)
        else:
            z = None

        return z, x



class Block(nn.Module):


    def __init__(self, in_channels, out_channels, padding, stride):
        
        super().__init__()

        self.downsample = None
        self.relu2 = nn.ReLU(inplace=True)

        self.block = nn.Sequential(OrderedDict([
                                    ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding, stride=stride)),
                                    ('norm1', nn.BatchNorm2d(num_features=out_channels)),
                                    ('relu1', nn.ReLU(inplace=True)),
                                    ('conv2', nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)),
                                    ('norm2', nn.BatchNorm2d(num_features=out_channels))
                                    ]))

        if stride != 1:
            self.downsample = nn.Sequential(OrderedDict([
                                            ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)),
                                            ('norm', nn.BatchNorm2d(num_features=out_channels))
                                            ]))

    
    def forward(self, x):

        y = self.block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        y = x + y
        y = self.relu2(y)

        return y