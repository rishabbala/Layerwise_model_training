import torch
import torch.nn as nn



class MakeResnet(nn.Module):


    def __init__(self, model_name, num_linear_layers, output_size, dataset, other_size=None):

        super().__init__()

        if model_name == 'resnet18':
            self.resnet = ResnetBasic(num_layers=[2, 2, 2, 2], num_linear_layers=2, output_size=output_size, dataset=dataset)

        elif model_name == 'resnet34':
            self.resnet = ResnetBasic(num_layers=[3, 4, 6, 3], num_linear_layers=2, output_size=output_size, dataset=dataset)
        
        elif model_name == 'other':
            try:
                self.resnet = ResnetBasic(num_layers=other_size, num_linear_layers=2, output_size=output_size, dataset=dataset)
            except:
                raise ValueError("Model cant be created as resnet")
    
    def forward(self, x):
        
        x = self.resnet(x)

        return x



class ResnetBasic(nn.Module):


    def __init__(self, num_layers, num_linear_layers, output_size, dataset):
        
        super().__init__()

        if 'cifar' in dataset:
            self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features=64),
                                        nn.ReLU(inplace=True))
                            
        else:
            self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                                        nn.BatchNorm2d(num_features=64), 
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer2 = self._make_layer(num_layers[0], in_channels=64, out_channels=64, padding=1, stride=1)
        self.layer3 = self._make_layer(num_layers[1], in_channels=64, out_channels=128, padding=1, stride=2)
        self.layer4 = self._make_layer(num_layers[2], in_channels=128, out_channels=256, padding=1, stride=2)
        self.layer5 = self._make_layer(num_layers[3], in_channels=256, out_channels=512, padding=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x).view(x.shape[0], x.shape[1])
        x = self.linear_layers(x)

        return x



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