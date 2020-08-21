import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math 
import torch 

from .attention import PAM_Module, CAM_Module

BatchNorm = nn.BatchNorm2d

webroot = 'http://dl.yf.io/drn/'

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                        padding=padding, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, pool_size=28, arch='D', 
                 attention=False):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch
        self.attention = attention

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
                        bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_conv_layers(
            channels[0], layers[0], stride=1)
        self.layer2 = self._make_conv_layers(
            channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)

        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)

        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False)

        self.layer6 = self._make_layer(block, channels[5], layers[5],
                                       dilation=4, new_level=False)
        
        self.layer7 = None if layers[6] == 0 else \
            self._make_conv_layers(channels[6], layers[6], dilation=2)

        self.layer8 = None if layers[7] == 0 else \
            self._make_conv_layers(channels[7], layers[7], dilation=1)



        # attention 

        inter_channels_9 = channels[-1] // 4
        self.conv9a = nn.Sequential(nn.Conv2d(512, inter_channels_9, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels_9),
                                    nn.ReLU())
        self.sa9 = PAM_Module(inter_channels_9)
        self.conv91 = nn.Sequential(nn.Conv2d(inter_channels_9, inter_channels_9, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels_9),
                                    nn.ReLU())
        self.conv10 = nn.Sequential(nn.Conv2d(inter_channels_9, inter_channels_9, 1))

        self.conv9c = nn.Sequential(nn.Conv2d(512, inter_channels_9, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels_9),
                                    nn.ReLU())

        self.sc9 = CAM_Module(inter_channels_9)
        self.conv92 = nn.Sequential(nn.Conv2d(inter_channels_9, inter_channels_9, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels_9),
                                    nn.ReLU())
        self.conv11 = nn.Sequential(nn.Conv2d(inter_channels_9, inter_channels_9, 1))

        self.conv12 = nn.Sequential(nn.Conv2d(256, 256, 1))
        # self.conv12 = nn.Sequential(nn.Conv2d(128, 128, 1))




        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            if self.attention:
                self.fc = nn.Linear(inter_channels_9 * 2 + 29, num_classes)
            else:
                self.fc = nn.Linear(self.out_dim + 29, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def  _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        # the following if function is added
        if stride == 2 and dilation == 4:
            self.inplanes = 256
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x, clinic):
        batch_size = x.shape[0]
        y = list()

        x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)

        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)
        
        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)


        # modified to fit the attention module
        if self.attention:
            feat1 = self.conv9a(x)
            sa_feat = self.sa9(feat1)
            sa_conv = self.conv91(sa_feat)
            sa_output = self.conv10(sa_conv)

            feat2 = self.conv9c(x)
            sc_feat = self.sc9(feat2)
            sc_conv = self.conv92(sc_feat)
            sc_output = self.conv11(sc_conv)

            x = torch.cat([sa_output, sc_output], dim=1)
            x = self.conv12(x)
            # x = self.conv12(sa_output)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = torch.cat((x, clinic), dim=1)
            x = self.fc(x)
            
            return x
        else:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = torch.cat((x, clinic), dim=1)
            x = self.fc(x)

            return x



def drn_54_clinic(pretrained=False, attention=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['drn-d-54'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                        k in model_dict and k != 'fc.weight' and k != 'fc.bias'}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def drn_attention_clinic(pretrained=False, attention=True, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['drn-d-54'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                        k in model_dict and k != 'fc.weight' and k != 'fc.bias'}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model 
