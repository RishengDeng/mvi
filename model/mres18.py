import torch 
import torch.nn as nn 
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F 


class MulRes18(nn.Module):
    def __init__(self, num_classes=2):
        super(MulRes18, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        modules_18 = list(resnet18.children())[:-2]
        fc_input = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules_18)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2 * 512 * 7 * 7, num_classes)
        
    def forward(self, x1, x2):
        x1 = self.resnet18(x1)
        # print('x1', x1.shape)
        x1 = self.flatten(x1)
        # print('x1_flatten', x1.shape)
        x2 = self.resnet18(x2)
        x2 = self.flatten(x2)
        x = torch.cat((x1, x2), dim=1)
        # print('x', x.shape)
        x = self.fc(x)
        return x 


class MulRes18Att(nn.Module):
    def __init__(self, num_classes=2):
        super(MulRes18Att, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        modules_18 = list(resnet18.children())[:-2]
        fc_input = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules_18)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_input, fc_input // 16)
        self.fc2 = nn.Linear(fc_input // 16, fc_input)
        self.fc = nn.Linear(2 * 512 * 7 * 7, num_classes)

    def forward(self, x1, x2):
        x1 = self.resnet18(x1)
        x1_fea = x1 
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x1 = x1.view(x1.size(0), x1.size(1), 1, 1)
        x1 = x1_fea * x1.expand_as(x1_fea)

        x2 = self.resnet18(x2)
        x2_fea = x2 
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1(x2)
        x2 = F.relu(x2)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x2 = x2.view(x2.size(0), x2.size(1), 1, 1)
        x2 = x2_fea * x2.expand_as(x2_fea)

        x = torch.cat((x1, x2), dim=1)
        x = self.flatten(x)
        x = self.fc(x) 
        return x 


class MulRes18AttRadio(nn.Module):
    def __init__(self, num_classes=2):
        super(MulRes18AttRadio, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        modules_18 = list(resnet18.children())[:-2]
        fc_input = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules_18)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_input, fc_input // 16)
        self.fc2 = nn.Linear(fc_input // 16, fc_input)
        self.fc = nn.Linear(2 * 512 * 7 * 7 + 1030 + 1080, num_classes)
        self.fc3 = nn.Linear(512 *7 * 7, 50)
        self.fc4 = nn.Linear(1030, 50)
        self.fc5 = nn.Linear(512 * 7 * 7, 50)
        self.fc6 = nn.Linear(1080, 50)
        self.fc7 = nn.Linear(200, num_classes)
        

    def forward(self, x1, x2, vector1, vector2):
        x1 = self.resnet18(x1)
        x1_fea = x1 
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x1 = x1.view(x1.size(0), x1.size(1), 1, 1)
        x1 = x1_fea * x1.expand_as(x1_fea)
        x1 = self.flatten(x1)
        x1 = self.fc3(x1)
        vector1 = self.fc4(vector1)
        x1 = torch.cat((x1, vector1), dim=1)

        x2 = self.resnet18(x2)
        x2_fea = x2 
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1(x2)
        x2 = F.relu(x2)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x2 = x2.view(x2.size(0), x2.size(1), 1, 1)
        x2 = x2_fea * x2.expand_as(x2_fea)
        x2 = self.flatten(x2)
        x2 = self.fc5(x2)
        vector2 = self.fc6(vector2)
        x2 = torch.cat((x2, vector2), dim=1)

        x = torch.cat((x1, x2), dim=1)
        x = self.flatten(x)
        x = self.fc7(x) 
        return x 