import torch 
import torch.nn as nn
import torchvision.models as models 
import torch.utils.model_zoo as model_zoo

from .drn_origin import drn_d_22
from .drn22 import drn_22_clinic


class ResnetLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, num_classes=2):
        super(ResnetLSTM, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        modules_18 = list(resnet18.children())[:-2]
        fc_input_18 = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules_18)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        resnet50 = models.resnet50(pretrained=True)
        modules_50 = list(resnet50.children())[:-2]
        fc_input_50 = resnet50.fc.in_features
        self.resnet50 = nn.Sequential(*modules_50)

        self.lstm = nn.LSTM(fc_input_18 + fc_input_50, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)


    def res18_fea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x 
    
    def res50_fea(self, x):
        x = self.resnet50(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x 
        

    def resfea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.squeeze(x, 1)
        return x 

    def forward(self, x_g, x_l, y_g, y_l, z_g, z_l):
        x_g = self.res50_fea(x_g)
        x_l = self.res18_fea(x_l)
        y_g = self.res50_fea(y_g)
        y_l = self.res18_fea(y_l)
        z_g = self.res50_fea(z_g)
        z_l = self.res18_fea(z_l)

        x = torch.cat((x_g, x_l), dim=1)
        y = torch.cat((y_g, y_l), dim=1)
        z = torch.cat((z_g, z_l), dim=1)
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 1)
        z = torch.unsqueeze(z, 1)

        cat_fea = torch.cat((x, y, z), dim=2)
        cat_fea = cat_fea.reshape(x.size(0), 3, -1)
        out, _ = self.lstm(cat_fea)
        out = out[:, -1, :]
        out = torch.squeeze(out, 1)
        out = self.fc(out)
        return out 
    

class ResnetLSTMClinic(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, num_classes=2):
        super(ResnetLSTMClinic, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        modules_18 = list(resnet18.children())[:-2]
        fc_input_18 = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules_18)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        resnet50 = models.resnet50(pretrained=True)
        modules_50 = list(resnet50.children())[:-2]
        fc_input_50 = resnet50.fc.in_features
        self.resnet50 = nn.Sequential(*modules_50)

        self.lstm = nn.LSTM(fc_input_18 + fc_input_50, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2 + 29, num_classes)


    def res18_fea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x 
    
    def res50_fea(self, x):
        x = self.resnet50(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x 
        

    def resfea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.squeeze(x, 1)
        return x 

    def forward(self, x_g, x_l, y_g, y_l, z_g, z_l, clinic):
        x_g = self.res50_fea(x_g)
        x_l = self.res18_fea(x_l)
        y_g = self.res50_fea(y_g)
        y_l = self.res18_fea(y_l)
        z_g = self.res50_fea(z_g)
        z_l = self.res18_fea(z_l)

        x = torch.cat((x_g, x_l), dim=1)
        y = torch.cat((y_g, y_l), dim=1)
        z = torch.cat((z_g, z_l), dim=1)
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 1)
        z = torch.unsqueeze(z, 1)

        cat_fea = torch.cat((x, y, z), dim=2)
        cat_fea = cat_fea.reshape(x.size(0), 3, -1)
        out, _ = self.lstm(cat_fea)
        out = out[:, -1, :]
        out = torch.squeeze(out, 1)
        out = torch.cat((out, clinic), dim=1)
        out = self.fc(out)
        return out 



class ResnetLSTMClinic2(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, num_classes=2):
        super(ResnetLSTMClinic2, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        modules_18 = list(resnet18.children())[:-2]
        fc_input_18 = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules_18)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        resnet50 = models.resnet50(pretrained=False)
        modules_50 = list(resnet50.children())[:-2]
        fc_input_50 = resnet50.fc.in_features
        self.resnet50 = nn.Sequential(*modules_50)

        self.lstm = nn.LSTM(fc_input_18 + fc_input_50, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 29)
        self.fc2 = nn.Linear(29 * 2, num_classes)


    def res18_fea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x 
    
    def res50_fea(self, x):
        x = self.resnet50(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x 
        

    def resfea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.squeeze(x, 1)
        return x 

    def forward(self, x_g, x_l, y_g, y_l, z_g, z_l, clinic):
        x_g = self.res50_fea(x_g)
        x_l = self.res18_fea(x_l)
        y_g = self.res50_fea(y_g)
        y_l = self.res18_fea(y_l)
        z_g = self.res50_fea(z_g)
        z_l = self.res18_fea(z_l)

        x = torch.cat((x_g, x_l), dim=1)
        y = torch.cat((y_g, y_l), dim=1)
        z = torch.cat((z_g, z_l), dim=1)
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 1)
        z = torch.unsqueeze(z, 1)

        cat_fea = torch.cat((x, y, z), dim=2)
        cat_fea = cat_fea.reshape(x.size(0), 3, -1)
        out, _ = self.lstm(cat_fea)
        out = out[:, -1, :]
        out = torch.squeeze(out, 1)
        out = self.fc1(out)
        out = torch.cat((out, clinic), dim=1)
        out = self.fc2(out)
        return out 



class Drn22LSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, num_classes=2):
        super(Drn22LSTM, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        modules_18 = list(resnet18.children())[:-2]
        fc_input_18 = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules_18)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        drn22 = drn_d_22(pretrained=False, num_classes=2)
        modules_drn22 = list(drn22.children())[:-1]
        self.drn22 = nn.Sequential(*modules_drn22)
        fc_input_22 = drn22.fc.in_channels

        self.lstm = nn.LSTM(fc_input_18 + fc_input_22, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)


    def res18_fea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x 
    
    def drn22_fea(self, x):
        x = self.drn22(x)
        x = x.view(x.size(0), -1)
        return x 
        

    def resfea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.squeeze(x, 1)
        return x 

    def forward(self, x_g, x_l, y_g, y_l, z_g, z_l):
        x_g = self.drn22_fea(x_g)
        x_l = self.res18_fea(x_l)
        y_g = self.drn22_fea(y_g)
        y_l = self.res18_fea(y_l)
        z_g = self.drn22_fea(z_g)
        z_l = self.res18_fea(z_l)

        x = torch.cat((x_g, x_l), dim=1)
        y = torch.cat((y_g, y_l), dim=1)
        z = torch.cat((z_g, z_l), dim=1)
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 1)
        z = torch.unsqueeze(z, 1)

        cat_fea = torch.cat((x, y, z), dim=2)
        cat_fea = cat_fea.reshape(x.size(0), 3, -1)
        out, _ = self.lstm(cat_fea)
        out = out[:, -1, :]
        out = torch.squeeze(out, 1)
        out = self.fc(out)
        return out 
    


class Drn22LSTMClinic(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, num_classes=2):
        super(Drn22LSTMClinic, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        modules_18 = list(resnet18.children())[:-2]
        fc_input_18 = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules_18)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        drn22 = drn_d_22(pretrained=False, num_classes=2)
        modules_drn22 = list(drn22.children())[:-1]
        self.drn22 = nn.Sequential(*modules_drn22)
        fc_input_22 = drn22.fc.in_channels

        self.lstm = nn.LSTM(fc_input_18 + fc_input_22, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2 + 29, num_classes)


    def res18_fea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x 
    
    def drn22_fea(self, x):
        x = self.drn22(x)
        x = x.view(x.size(0), -1)
        return x 
        

    def resfea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.squeeze(x, 1)
        return x 

    def forward(self, x_g, x_l, y_g, y_l, z_g, z_l, clinic):
        x_g = self.drn22_fea(x_g)
        x_l = self.res18_fea(x_l)
        y_g = self.drn22_fea(y_g)
        y_l = self.res18_fea(y_l)
        z_g = self.drn22_fea(z_g)
        z_l = self.res18_fea(z_l)

        x = torch.cat((x_g, x_l), dim=1)
        y = torch.cat((y_g, y_l), dim=1)
        z = torch.cat((z_g, z_l), dim=1)
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 1)
        z = torch.unsqueeze(z, 1)

        cat_fea = torch.cat((x, y, z), dim=2)
        cat_fea = cat_fea.reshape(x.size(0), 3, -1)
        out, _ = self.lstm(cat_fea)
        out = out[:, -1, :]
        out = torch.squeeze(out, 1)
        out = torch.cat((out, clinic), dim=1)
        out = self.fc(out)
        return out 
    



class Drn22LSTMClinic2(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, num_classes=2):
        super(Drn22LSTMClinic2, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        modules_18 = list(resnet18.children())[:-2]
        fc_input_18 = resnet18.fc.in_features
        self.resnet18 = nn.Sequential(*modules_18)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        drn22 = drn_d_22(pretrained=False, num_classes=2)
        modules_drn22 = list(drn22.children())[:-1]
        self.drn22 = nn.Sequential(*modules_drn22)
        fc_input_22 = drn22.fc.in_channels

        self.lstm = nn.LSTM(fc_input_18 + fc_input_22, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 29)
        self.fc2 = nn.Linear(29 * 2, num_classes)


    def res18_fea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x 
    
    def drn22_fea(self, x):
        x = self.drn22(x)
        x = x.view(x.size(0), -1)
        return x 
        

    def resfea(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.squeeze(x, 1)
        return x 

    def forward(self, x_g, x_l, y_g, y_l, z_g, z_l, clinic):
        x_g = self.drn22_fea(x_g)
        x_l = self.res18_fea(x_l)
        y_g = self.drn22_fea(y_g)
        y_l = self.res18_fea(y_l)
        z_g = self.drn22_fea(z_g)
        z_l = self.res18_fea(z_l)

        x = torch.cat((x_g, x_l), dim=1)
        y = torch.cat((y_g, y_l), dim=1)
        z = torch.cat((z_g, z_l), dim=1)
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 1)
        z = torch.unsqueeze(z, 1)

        cat_fea = torch.cat((x, y, z), dim=2)
        cat_fea = cat_fea.reshape(x.size(0), 3, -1)
        out, _ = self.lstm(cat_fea)
        out = out[:, -1, :]
        out = torch.squeeze(out, 1)
        out = self.fc1(out)
        out = torch.cat((out, clinic), dim=1)
        out = self.fc2(out)
        return out 
    