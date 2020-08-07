import torch 
import torchvision.models as models 
import torch.nn as nn 
import torch.utils.model_zoo as model_zoo

from .drn_origin import drn_d_54, drn_d_22
from .drn_attention import drn_attention
from .drn22 import drn_d_22_test


class Resnet18(nn.Module):
    def __init__(self, num_class=2):
        super(Resnet18, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        modules = list(resnet18.children())[:-2]
        fc_input = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.fc = nn.Linear(2048, num_class)
        # self.fc = nn.Linear(512, num_class)
        self.fc = nn.Linear(fc_input, num_class)
    
    def forward(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.model = self.resnet50()
    
    def resnet50(self, num_class=2):
        model = models.resnet50(pretrained=True)
        # read the parameter of resnet50
        pretrained_dict = model.state_dict()
        # change the last fc layer
        fc_input = model.fc.in_features
        model.fc = nn.Linear(fc_input, num_class)
        model_dict = model.state_dict()
        # filter out the last fc layer 
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 
            k in model_dict and k != 'fc.weight' and k != 'fc.bias'
        }
        # update the parameter
        model_dict.update(pretrained_dict)
        # load the parameter that we need
        model.load_state_dict(model_dict)
        return model

    def forward(self, x):
        x = self.model(x)
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

class DRN22(nn.Module):
    def __init__(self):
        super(DRN22, self).__init__()
        self.drn22 = drn_d_22(pretrained=False, num_classes=2)

    def forward(self, x):
        x = self.drn22(x)
        return x


class DRN22_test(nn.Module):
    def __init__(self):
        super(DRN22_test, self).__init__()
        self.drn22_test = drn_d_22_test(pretrained=False, num_classes=2, out_map=True)
        
    def forward(self, x):
        x, x_vector = self.drn22_test(x)
        # print(x_vector)
        return x, x_vector 


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.attention = drn_attention(pretrained=True, num_classes=2)

    def forward(self, x):
        x = self.attention(x)
        return x



class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(64, 192, kernel_size=5, padding=2), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(192, 384, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(384, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(), 
                nn.Linear(256 * 6 * 6, 4096), 
                nn.ReLU(inplace=True), 
                nn.Dropout(), 
                nn.Linear(4096, 4096), 
                nn.ReLU(inplace=True), 
                nn.Linear(4096, num_classes), 
            )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        if self.num_classes > 0:
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)

        return x



class LeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1), 
            nn.Tanh(), 
            nn.AvgPool2d(kernel_size=2), 
            nn.Conv2d(6, 16, kernel_size=5, stride=1), 
            nn.Tanh(), 
            nn.AvgPool2d(kernel_size=2), 
            nn.Conv2d(16, 120, kernel_size=5, stride=1), 
            nn.Tanh(), 
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(120 * 49 * 49, 84), 
                nn.Tanh(), 
                nn.Linear(84, num_classes), 
            )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        if self.num_classes > 0:
            x = self.classifier(x.view(x.shape[0], -1))
        
        return x