import torch 
import torchvision.models as models 
import torch.nn as nn 

from .drn_origin import drn_d_54
from .drn_attention import drn_attention


class Resnet(nn.Module):
    def __init__(self, num_class=2):
        super(Resnet, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        modules = list(resnet18.children())[:-2]
        fc_input = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.fc = nn.Linear(2048, num_class)
        self.fc = nn.Linear(512, num_class)
    
    def forward(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 


class DenseNet(nn.Module):
    def __init__(self, num_class=2):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=False, num_classes=2)
    
    def forward(self, x):
        x = self.densenet(x)
        return x 


class DilatedResnet(nn.Module):
    def __init__(self):
        super(DilatedResnet, self).__init__()
        self.drn = drn_d_54(pretrained=True, num_classes=2)

    def forward(self, x):
        x = self.drn(x)
        return x 


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.attention = drn_attention(pretrained=True, num_classes=2)

    def forward(self, x):
        x = self.attention(x)
        return x