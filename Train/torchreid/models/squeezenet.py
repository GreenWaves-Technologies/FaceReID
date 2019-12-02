from __future__ import absolute_import
from __future__ import division

import collections
from itertools import repeat
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torch.nn import functional as F
import torch.nn.init as init
import torchvision
import torch.utils.model_zoo as model_zoo

from torchreid.utils.quantization import round, roundnorm_reg, gap8_clip

__all__ = ['squeezenet1_0', 'squeezenet1_1', 'squeezenet1_0_fc512']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bits=16):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.bits = bits
        self.first_forward = False

    def forward(self, input):
        try: # quantized convolution
            # out = F.conv2d(input, self.weight, None, self.stride,
            #          self.padding, self.dilation, self.groups)
            # out = torch.clamp(out, -math.pow(2., 31), math.pow(2., 31) - 1)
            # out = gap8_clip(roundnorm_reg(out, self.norm), self.bits)
            # out += self.bias.view(1, -1, 1, 1).expand_as(out)
            self.weights = nn.ParameterList(
                [nn.Parameter(self.weight.data[:, i, :, :].unsqueeze_(1)) for i in range(self.weight.shape[1])])
            out = None
            for i in range(input.shape[1]):
                conv_res = F.conv2d(input[:, i, :, :].unsqueeze_(1), self.weights[i], None, self.stride,
                         self.padding, self.dilation, self.groups)
                #out = torch.clamp(out, -math.pow(2., 31), math.pow(2., 31) - 1)
                tmp = gap8_clip(roundnorm_reg(conv_res, self.norm), self.bits)
                if out is None:
                    out = tmp
                else:
                    out += tmp
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
            out = torch.clamp(out, -math.pow(2., self.bits - 1), math.pow(2., self.bits) - 1)
            return out
        except AttributeError:
            if not self.first_forward:
                self.first_forward = True
            else:
                print('Something went wrong!')
            return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv2d(input_channels, out_channels, quantized=False, bits=16, convbn=False, **kwargs):
    if quantized:
        return QuantizedConv2d(input_channels, out_channels, bits=bits, **kwargs)
    elif convbn:
        return nn.Sequential(nn.Conv2d(input_channels, out_channels, **kwargs),
                             nn.BatchNorm2d(out_channels))
    else:
        return nn.Conv2d(input_channels, out_channels, **kwargs)


def average_pooling(kernel_size, quantized=False, bits=16, **kwargs):
    if quantized:
        return GapAvgPool(kernel_size, out_bits=bits)
    else:
        return nn.AvgPool2d(kernel_size)
    

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, pool=False, ceil_mode=True, **kwargs):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = conv2d(inplanes, squeeze_planes, kernel_size=1, **kwargs)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1, **kwargs)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1, **kwargs)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.pool = pool

        if self.pool == 'max':
            self.maxpool1x1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode)
            self.maxpool3x3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        if self.pool == 'max':
            return torch.cat([
                self.maxpool1x1(self.expand1x1_activation(self.expand1x1(x))),
                self.maxpool3x3(self.expand3x3_activation(self.expand3x3(x)))
            ], 1)
        else:
            return torch.cat([
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x))
            ], 1)


class Identity(nn.Module):
    def forward(self, input):
        return input


class GapAvgPool(nn.Module): # Avg Pooling was tested only like Global Average Pooling
    def __init__(self, kernel_size, out_bits=16):
        super(GapAvgPool, self).__init__()
        self.kernel_size = kernel_size
        self.out_bits = out_bits

    def forward(self, input):
        input = F.avg_pool2d(input, self.kernel_size)
        input = torch.floor_(input * self.kernel_size * self.kernel_size + 0.1)
        pool_factor = math.pow(2, 16) // math.pow(self.kernel_size, 2)
        bound = math.pow(2.0, self.out_bits - 1)
        min_val = -bound
        max_val = bound - 1
        return torch.clamp(roundnorm_reg(input * pool_factor, self.out_bits), min_val, max_val)


class SqueezeNet(nn.Module):
    """
    SqueezeNet

    Reference:
    Iandola et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and< 0.5 MB model size. arXiv:1602.07360.
    """
    def __init__(self, num_classes, loss, version=1.0, fc_dims=None, dropout_p=None, grayscale=False, ceil_mode=True,
                 infer=False, normalize_embeddings=False, normalize_fc=False, **kwargs):
        super(SqueezeNet, self).__init__()
        self.loss = loss
        self.feature_dim = 512
        self.normalize_embeddings = normalize_embeddings
        self.normalize_fc = normalize_fc

        self.grayscale = grayscale
        if grayscale:
            self.conv1 = conv2d(1, 3, kernel_size=1, **kwargs)

        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))

        if version == 1.0:
            self.features = nn.Sequential(
                conv2d(3, 96, kernel_size=7, stride=2, **kwargs),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(96, 16, 64, 64, **kwargs),
                Fire(128, 16, 64, 64, **kwargs),
                Fire(128, 32, 128, 128, pool=True, ceil_mode=ceil_mode, **kwargs),
                Identity(), #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode), # Pooling is inside of Fire module
                Fire(256, 32, 128, 128, **kwargs),               # Identity blocks are needed for backward compatibility
                Fire(256, 48, 192, 192, **kwargs),               # with previously trained models
                Fire(384, 48, 192, 192, **kwargs),
                Fire(384, 64, 256, 256, pool=True, ceil_mode=ceil_mode, **kwargs),
                Identity(), #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(512, 64, 256, 256, **kwargs),
            )
        else:
            self.features = nn.Sequential(
                conv2d(3, 64, kernel_size=3, stride=2, **kwargs),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(64, 16, 64, 64, **kwargs),
                Fire(128, 16, 64, 64, pool='max', ceil_mode=ceil_mode, **kwargs),
                Identity(), #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(128, 32, 128, 128, **kwargs),
                Fire(256, 32, 128, 128, pool='max', ceil_mode=ceil_mode, **kwargs),
                Identity(), #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(256, 48, 192, 192, **kwargs),
                Fire(384, 48, 192, 192, **kwargs),
                Fire(384, 64, 256, 256, **kwargs),
                Fire(512, 64, 256, 256, **kwargs),
            )

        self.global_avgpool = average_pooling(7, **kwargs)
        self.fc = self._construct_fc_layer(fc_dims, 512, dropout_p)
        if not infer:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        
        assert isinstance(fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))
        
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        self.feature_dim = fc_dims[-1]
        
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.grayscale:
            x = self.conv1(x)

        f = self.features(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.normalize_embeddings:
            v = F.normalize(v)

        if self.normalize_fc and hasattr(self, 'classifier'):
            self.classifier.weight.data = F.normalize(self.classifier.weight.data)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url, map_location=None)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url))


def squeezenet1_0(num_classes, loss, pretrained=True, **kwargs):
    model = SqueezeNet(
        num_classes, loss,
        version=1.0,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_0'])
    return model


def squeezenet1_0_fc512(num_classes, loss, pretrained=True, **kwargs):
    model = SqueezeNet(
        num_classes, loss,
        version=1.0,
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_0'])
    return model


def squeezenet1_1(num_classes, loss, pretrained=True, **kwargs):
    model = SqueezeNet(
        num_classes, loss,
        version=1.1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_1'])
    return model