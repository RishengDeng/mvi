import torch 
import torchvision.models as models 
import torch.nn as nn 


class Resnet(nn.Module):
    def __init__(self, num_class=2):
        super(Resnet, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        modules = list(resnet50.children())[:-2]
        fc_input = resnet50.fc.in_features
        self.resnet50 = nn.Sequential(*modules)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, num_class)
    
    def forward(self, x):
        x = self.resnet50(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 