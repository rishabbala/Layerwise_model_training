import torch
import torch.nn as nn
from collections import OrderedDict



class MakeResnetBlockwise(nn.Module):


    def __init__(self, num_linear_layers, output_size, dataset, block_size=None):

        super().__init__()
        self.resnet = ResnetBasic(num_layers=block_size, num_linear_layers=num_linear_layers, output_size=output_size, dataset=dataset)
    
    def forward(self, x):
        
        z, f = self.resnet(x)

        return z, f



class ResnetBasic(nn.Module):


    def __init__(self, num_layers, num_linear_layers, output_size, dataset):
        
        super().__init__()

        self.num_layers = num_layers

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

        self.layer0 = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None


        self.layer0 = self._make_layer(num_layers[0], in_channels=64, out_channels=64, padding=1, stride=1)
        feature_size = 64
        if num_layers[1] != -1:
            self.layer1 = self._make_layer(num_layers[1], in_channels=64, out_channels=128, padding=1, stride=2)
            feature_size = 128
        if num_layers[2] != -1:
            self.layer2 = self._make_layer(num_layers[2], in_channels=128, out_channels=256, padding=1, stride=2)
            feature_size = 256
        if num_layers[3] != -1:
            self.layer3 = self._make_layer(num_layers[3], in_channels=256, out_channels=512, padding=1, stride=2)
            feature_size = 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_layers = []

        self.feature_projection_layers = nn.Sequential(OrderedDict([
                                                    ('linear1', nn.Linear(in_features=512, out_features=512)),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('linear2', nn.Linear(in_features=512, out_features=128)),
                                                    ]))

        self.linear_layers = nn.Linear(in_features=512, out_features=output_size)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         # nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


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
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)

        x = self.avgpool(x).view(x.shape[0], x.shape[1])

        f = self.feature_projection_layers(x)
        f = torch.nn.functional.normalize(f, dim=1)

        z = self.linear_layers(x)

        return z, f



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