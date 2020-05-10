import torch.nn as nn
import math, pdb
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter

### Download weights from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=bias)

def conv1x1(in_channel, out_channel, **kwargs):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, **kwargs)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNetFPN(nn.Module):

    def __init__(self, block, layers, use_bias, seq_len):
        self.inplanes = 64
        super(ResNetFPN, self).__init__()
        self.conv1 = nn.Conv2d(3*seq_len, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.conv6 = conv3x3(512 * block.expansion, 256, stride=2, padding=1, bias=use_bias)  # P6
        self.conv7 = conv3x3(256, 256, stride=2, padding=1, bias=use_bias)  # P7

        self.lateral_layer1 = conv1x1(512 * block.expansion, 256, bias=use_bias)
        self.lateral_layer2 = conv1x1(256 * block.expansion, 256, bias=use_bias)
        self.lateral_layer3 = conv1x1(128 * block.expansion, 256, bias=use_bias)
        
        self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1, bias=use_bias)  # P4
        self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1, bias=use_bias)  # P4
        self.corr_layer3 = conv3x3(256, 256, stride=1, padding=1, bias=use_bias)  # P3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=1)
                if hasattr(m.bias, 'data'):
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _upsample(self, x, y):
        _, _, h, w = y.size()
        x_upsampled = F.interpolate(x, [h, w], mode='nearest')

        return x_upsampled

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('x0 DATA ', torch.sum(x.data))
        x = self.layer1(x)
        # print('x DATA ', torch.sum(x.data))
        c3 = self.layer2(x)
        # print('c3 DATA ', torch.sum(c3.data))
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.lateral_layer1(c5)
        p5_upsampled = self._upsample(p5, c4)
        p5 = self.corr_layer1(p5)

        p4 = self.lateral_layer2(c4)
        p4 = p5_upsampled + p4
        p4_upsampled = self._upsample(p4, c3)
        p4 = self.corr_layer2(p4)

        p3 = self.lateral_layer3(c3)
        p3 = p4_upsampled + p3
        p3 = self.corr_layer3(p3)

        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        return p3, p4, p5, p6, p7

    def load_my_state_dict(self, state_dict, seq_len=1):

        ### Download weights from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        own_state = self.state_dict()
        # print(own_state.keys())
        for name, param in state_dict.items():
            # pdb.set_trace()
            if name in own_state.keys():
                # print(name)
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                if name == 'conv1.weight':
                    print(name, 'is being filled with {:d} seq_len\n'.format(seq_len))
                    param = param.repeat(1, seq_len, 1, 1)
                    param = param / float(seq_len)
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            else:
                print('NAME IS NOT IN OWN STATE::>' + name)

def resnetfpn(perms, name, use_bias, seq_len=1):
    num = int(name[6:])
    if num<50:
        return ResNetFPN(BasicBlock, perms, use_bias, seq_len)
    else:
        return ResNetFPN(Bottleneck, perms, use_bias, seq_len)
