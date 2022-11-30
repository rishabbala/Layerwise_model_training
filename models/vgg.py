import torch
import torch.nn as nn
from collections import OrderedDict



class MakeVGG(nn.Module):


    def __init__(self, num_linear_layers, output_size, dataset, block_size=None):

        super().__init__()
        self.vgg = VGGBasic(num_layers=block_size, num_linear_layers=num_linear_layers, output_size=output_size, dataset=dataset)
    
    def forward(self, x):
        
        x = self.vgg(x)

        return x



class VGGBasic(nn.Module):


    def __init__(self, num_layers, num_linear_layers, output_size, dataset):
        
        super().__init__()

        self.layer0 = self._make_layer(num_layers[0], in_channels=3, out_channels=64, padding=1)
        self.layer1 = self._make_layer(num_layers[1], in_channels=64, out_channels=128, padding=1)
        self.layer2 = self._make_layer(num_layers[2], in_channels=128, out_channels=256, padding=1)
        self.layer3 = self._make_layer(num_layers[3], in_channels=256, out_channels=512, padding=1)
        self.layer4 = self._make_layer(num_layers[4], in_channels=512, out_channels=512, padding=1)

        self.mp = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.adaptivemaxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.linear_layers = []

        num_features_per_lin_layer = abs(int((512-output_size)/num_linear_layers))
        sign = int((512-output_size)/abs(512-output_size))
        for i in range(num_linear_layers-1):
            self.linear_layers.append(nn.Linear(in_features=512-sign*i*num_features_per_lin_layer, out_features=512-sign*(i+1)*num_features_per_lin_layer))
            self.linear_layers.append(nn.ReLU(inplace=True))
            final_size = 512-sign*(i+1)*num_features_per_lin_layer

        self.linear_layers.append(nn.Linear(in_features=final_size, out_features=output_size))
        self.linear_layers = nn.Sequential(*self.linear_layers)


    def _make_layer(self, n_layer, in_channels, out_channels, padding):

        layer = []
        channel_layer = []

        for _ in range(n_layer-1):
            channel_layer.append(out_channels)

        channel_layer.insert(0, in_channels)

        for i in range(n_layer):
            layer.append(Block(channel_layer[i], out_channels, padding=padding))

        return nn.Sequential(*layer)


    def forward(self, x):

        x = self.layer0(x)
        x = self.mp(x)
        x = self.layer1(x)
        x = self.mp(x)
        x = self.layer2(x)
        x = self.mp(x)
        x = self.layer3(x)
        x = self.mp(x)
        x = self.layer4(x)

        x = self.adaptivemaxpool(x).view(x.shape[0], x.shape[1])
        x = self.linear_layers(x)

        return x



class Block(nn.Module):


    def __init__(self, in_channels, out_channels, padding):
        
        super().__init__()

        self.downsample = None
        self.relu = nn.ReLU(inplace=True)

        self.block = nn.Sequential(OrderedDict([
                                    ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding, stride=1)),
                                    ('norm', nn.BatchNorm2d(num_features=out_channels)),
                                    ('relu', nn.ReLU(inplace=True))
                                    ]))
    
    def forward(self, x):

        y = self.block(x)
        return y