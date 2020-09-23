import torch.nn as nn
import torch

class RCU(nn.Module):
    def __init__(self, planes, kernel_size=1, stride=1, bias=False):
        super(RCU, self).__init__()
        # Module for recurrent part
        self.conv_whh = nn.Conv3d(
            planes, planes, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, in_data):
        h = in_data[:, :, 0, :, :]
        h = h.unsqueeze(2)
        out = h
        for i in range(1, in_data.size(2)):
            h = self.conv_whh(h)
            h = (h + in_data[:, :, i, :, :].unsqueeze(2)) / 2.0
            out = torch.cat((out, h), 2)
        return out


class BottleneckRCN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, temp_kernal=1, downsample=None):
        super(BottleneckRCN, self).__init__()
        self.temp_kernal = temp_kernal
        temp_pad = 1
        if temp_kernal == 1:
            temp_pad = 0
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(
            1, 1, 1), padding=(0, 0, 0),  bias=False)
        if self.temp_kernal > 1:
            self.rcu = RCU(planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(
            1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.conv1(x)
        if self.temp_kernal > 1:
            out = self.rcu(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # print(out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
        # print(residual.shape)
        out += residual
        out = self.relu(out)

        return out


class CLSTM(nn.Module):
    def __init__(self, planes, kernel_size=1, stride=1, bias=True):
        super(CLSTM, self).__init__()
        self.hidden_dim = planes
        self.recurrent_conv = nn.Conv2d(
            planes*2, planes*4, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, in_data):
        in_shape = in_data.shape
        h_curr = torch.zeros(in_shape[0], in_shape[1], in_shape[3], in_shape[4], device=in_data.device)
        c_curr = torch.zeros(in_shape[0], in_shape[1], in_shape[3], in_shape[4], device=in_data.device)
        out = None
        for i in range(in_data.size(2)):
            x = torch.cat((in_data[:, :, i, :, :], h_curr), 1)
            cx = self.recurrent_conv(x)

            it, ft, ot, gt = torch.split(cx, self.hidden_dim, dim=1) 
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            ot = torch.sigmoid(ot)
            gt = torch.tanh(gt)

            c_curr = ft * c_curr + it * gt
            h_curr = ot * torch.tanh(c_curr)

            if out is None:
                out = h_curr.unsqueeze(2)
            else:
                out = torch.cat((out, h_curr.unsqueeze(2)), 2)

        return out


class BottleneckRCLSTM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, temp_kernal=1, downsample=None):
        super(BottleneckRCLSTM, self).__init__()
        self.temp_kernal = temp_kernal
        temp_pad = 1
        if temp_kernal == 1:
            temp_pad = 0
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(
            1, 1, 1), padding=(0, 0, 0),  bias=False)
        if self.temp_kernal > 1:
            self.clstm = CLSTM(planes, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(
            1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.conv1(x)
        if self.temp_kernal > 1:
            out1 = self.clstm(out)
            out = out1 + out
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # print(out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
        # print(residual.shape)
        out += residual
        out = self.relu(out)

        return out


class CGRU(nn.Module):
    def __init__(self, planes, kernel_size=1, stride=1, bias=True):
        super(CGRU, self).__init__()
        self.hidden_dim = planes
        self.recurrent_conv_gates = nn.Conv2d(
            planes*2, planes*2, kernel_size=kernel_size, stride=stride, bias=bias)
        self.recurrent_conv_out = nn.Conv2d(
            planes*2, planes, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, in_data):
        in_shape = in_data.shape
        # h_curr = torch.zeros(in_shape[0], in_shape[1], in_shape[3], in_shape[4], device=in_data.device)
        # h_curr = torch.zeros(in_shape[0], in_shape[1], in_shape[3], in_shape[4], device=in_data.device)
        h_curr = in_data[:,:,0,:,:]
        out = None
        for i in range(in_data.size(2)):
            xin = torch.cat((in_data[:, :, i, :, :], h_curr), 1)
            cx = self.recurrent_conv_gates(xin)

            update, reset = torch.split(cx, self.hidden_dim, dim=1) 
            update = torch.sigmoid(update)
            reset = torch.sigmoid(reset)
            
            x_out = torch.tanh(self.recurrent_conv_out(torch.cat([in_data[:, :, i, :, :], h_curr * reset], dim=1)))
            h_curr = h_curr * (1 - update) + x_out * update

            if out is None:
                out = h_curr.unsqueeze(2)
            else:
                out = torch.cat((out, h_curr.unsqueeze(2)), 2)

        return out


class BottleneckRCGRU(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, temp_kernal=1, downsample=None):
        super(BottleneckRCGRU, self).__init__()
        self.temp_kernal = temp_kernal
        temp_pad = 1
        if temp_kernal == 1:
            temp_pad = 0
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(
            1, 1, 1), padding=(0, 0, 0),  bias=False)
        if self.temp_kernal > 1:
            self.cgru = CGRU(planes, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(
            1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.conv1(x)
        if self.temp_kernal > 1:
            out1 = self.cgru(out)
            out = out1 + out
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # print(out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
        # print(residual.shape)
        out += residual
        out = self.relu(out)

        return out

class BottleneckI3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, temp_kernal=1, downsample=None):
        super(BottleneckI3D, self).__init__()
        # self.expansion = 4
        temp_pad = 1
        if temp_kernal == 1:
            temp_pad = 0
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(
            temp_kernal, 1, 1), padding=(temp_pad, 0, 0),  bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(
            1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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


class Bottleneck2PD(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, temp_kernal=1, downsample=None):
        super(Bottleneck2PD, self).__init__()
        self.temp_kernal = temp_kernal
        middle_filters = planes
        if self.temp_kernal>1:
            i = 1 * inplanes * planes * 1 * 1
            i /= inplanes * 1 * 1 + 1 * planes
            middle_filters = int(i)
        print('Middlefilters are 1', middle_filters)
        self.conv1 = nn.Conv3d(inplanes, middle_filters,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(middle_filters)
        if self.temp_kernal > 1:
            self.conv1_seprated = nn.Conv3d(
                middle_filters, planes, kernel_size=1, bias=False)
            self.bn1s = nn.BatchNorm3d(planes)

        if self.temp_kernal > 1:
            i = 3 * planes * planes * 3 * 3
            i /= planes * 3 * 3 + 3 * planes
            middle_filters = int(i)
        print('Middlefilters are 2', middle_filters)
        self.conv2 = nn.Conv3d(planes, middle_filters, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                               padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(middle_filters)
        
        if self.temp_kernal > 1:
            self.conv2_seprated = nn.Conv3d(middle_filters, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                                            padding=(1, 0, 0),
                                            bias=False)
            self.bn2s = nn.BatchNorm3d(planes)
        out_planes = planes * self.expansion
        middle_filters = out_planes
        if self.temp_kernal > 1:
            i = 1 * out_planes * planes * 1 * 1
            i /= planes * 1 * 1 + 1 * out_planes
            middle_filters = int(i)
        
        self.conv3 = nn.Conv3d(planes, middle_filters,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(middle_filters)

        if self.temp_kernal > 1:
            self.conv3_seprated = nn.Conv3d(
                middle_filters, out_planes, kernel_size=1, bias=False)
            self.bn3s = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.temp_kernal>1:
            out = self.conv1_seprated(out)
            out = self.bn1s(out)
            out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.temp_kernal > 1:
            out = self.conv2_seprated(out)
            out = self.bn2s(out)
            out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        

        if self.temp_kernal > 1:
            out = self.relu(out)
            out = self.conv3_seprated(out)
            out = self.bn3s(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckC2D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, temp_kernal=1, downsample=None):
        super(BottleneckC2D, self).__init__()
        # self.expansion = 4
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(
            1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # self.apply(c2_msra_fill)


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
