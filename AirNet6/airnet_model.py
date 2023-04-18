import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from collections import OrderedDict



def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, mean=0, std=math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # [N, C, T, V]
        return x[:, :, :x.shape[2]-self.chomp_size,:]

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2,padding_mode='replicate', stride=1,dilation_size=1):
        super(unit_tcn, self).__init__()

        pad = (kernel_size-1) * dilation_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),padding_mode=padding_mode, 
            stride=(stride, 1), dilation =(dilation_size,1))
        self.chomp = Chomp(pad)
        self.bn = nn.BatchNorm2d(out_channels)

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.chomp(x)
        x = self.bn(x)

        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels,  point_count, is_cuda):
        super(unit_gcn, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.B = nn.Parameter(torch.zeros(point_count, point_count) + 1e-6)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.A = Variable(torch.eye(point_count), requires_grad=False)
        self.is_cuda = is_cuda

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv, 1)

        

    def forward(self, x):
        N, C, T, V = x.size()

        f_in = x.contiguous().view(N, C * T, V)

        adj_mat = None
        if self.is_cuda:
            self.A = self.A.cuda(x.get_device())
            
        adj_mat = self.B[:,:] + self.A[:,:]
        adj_mat_min = torch.min(adj_mat)
        adj_mat_max = torch.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)

        D = Variable(torch.diag(torch.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = torch.sqrt(torch.inverse(D))

        adj_mat_norm_d12 = torch.matmul(torch.matmul(D_12, adj_mat), D_12)

        y = self.conv(torch.matmul(f_in, adj_mat_norm_d12).view(N, C, T, V))

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)    

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, point_count, is_cuda, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()

        # self.tcn1 = unit_tcn(in_channels, out_channels, stride=stride, dilation_size=1)
        self.gcn1 = unit_gcn(in_channels, out_channels, point_count,is_cuda)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride, dilation_size=1)
        self.tcn2 = unit_tcn(out_channels, out_channels, stride=stride, dilation_size=3)
        self.tcn3 = unit_tcn(out_channels, out_channels, stride=stride, dilation_size=8)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            # self.residual = unit_tcn(out_channels, out_channels, kernel_size=1, stride=stride)
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)


    def forward(self, x):
# Adding the output of the gcn layer to the input of the gcn layer.
        x = self.gcn1(x) + self.residual(x)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        x = self.relu(x)

        return x


class Model(nn.Module):
    def __init__(self, model_layers, input_timesteps, output_time_points,  point_count, channel_count, is_cuda):
        super(Model, self).__init__()
        self.input_timesteps = input_timesteps
        self.output_time_points = output_time_points
        self.is_cuda = is_cuda
        self.data_bn = nn.BatchNorm1d(channel_count * point_count) # channels (C) * vertices (V) ###

        self.trunk_layers = []
        layer_input = channel_count
        layer_outputs = [int(s) for s in model_layers.split(',')]
        for i in range(len(layer_outputs)-1):
            layer = TCN_GCN_unit(layer_input, layer_outputs[i],  point_count, is_cuda )
            self.add_module(f'trunk_{i}', layer)
            self.trunk_layers.append(layer)
            layer_input = layer_outputs[i]

        last_layer_output = layer_outputs[-1]
        self.l3s = []
        self.rds = []
        self.fcs = []
        for output_index in range(self.output_time_points):
            # l3 = TCN_GCN_unit(32, 64,  pointCount, isCuda)
            l3 = TCN_GCN_unit(layer_input, last_layer_output,  point_count, is_cuda)
            self.add_module(f'l3_{output_index}', l3)
            self.l3s.append( l3)
            # rd = unit_tcn(64, 4, kernel_size=1, stride=1)
            rd = unit_tcn(last_layer_output, 4, kernel_size=1, stride=1)
            self.add_module(f'conv_reduce_dim_{output_index}', rd)
            self.rds.append(rd)
            fc = nn.Linear(4*self.input_timesteps*point_count, point_count)
            self.add_module(f'fc_{output_index}', fc)
            nn.init.normal_(fc.weight, 0, math.sqrt(2. / point_count))
            self.fcs.append(fc)


        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)
        for trunk_layer in self.trunk_layers:
            x= trunk_layer(x)

        output = torch.empty((N,self.output_time_points, V)).to(x.get_device())
        if self.is_cuda:
            output = output.cuda(x.get_device())
        for output_index in range(self.output_time_points):
            o = x
            o = self.l3s[output_index](o)
            o = self.rds[output_index](o)
            o = o.view(o.size(0), -1)
            o = self.fcs[output_index](o)
            output[:, output_index, :]  = o
        return output

