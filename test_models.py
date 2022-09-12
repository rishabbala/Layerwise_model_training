from resnet import MakeResnet
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR



device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter('./runs/resnet18/')

batch_size = 16


train_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618], 
                                                           std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628], inplace=True),
                                    
                                    #   transforms.RandomCrop(32, padding=4, padding_mode='reflect')
                                    
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomVerticalFlip(0.5),
                                      transforms.RandomGrayscale(0.5),
                                      transforms.RandomAffine(degrees=(-45, 45), translate=(0.5, 0.5)),
                                      transforms.RandomResizedCrop(size=(32, 32), scale=(0.5, 1)),
                                      transforms.Resize((32, 32))])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618], 
                                                          std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]),
                                     transforms.Resize((32, 32))])


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
# model = MakeResnet('resnet18').to(device)


model.eval()

train_acc = 0
num_train_images = 0
for i,  (images, labels) in enumerate(train_dataloader):
        
    images = images.to(device)
    labels = labels.to(device)
    
    out = model(images)
    pred = torch.argmax(out, dim=1)

    train_acc += torch.sum(torch.where(pred==labels, 1, 0))
    num_train_images += images.shape[0]

train_acc = train_acc/num_train_images    
    

val_acc = 0
num_val_images = 0
for _,  (images, labels) in enumerate(test_dataloader):

    images = images.to(device)
    labels = labels.to(device)
    
    out = model(images)
    pred = torch.argmax(out, dim=1)

    val_acc += torch.sum(torch.where(pred==labels, 1, 0))
    num_val_images += images.shape[0]

val_acc = val_acc/num_val_images

print("Train Acc: {}, Val Acc: {}".format(train_acc, val_acc))