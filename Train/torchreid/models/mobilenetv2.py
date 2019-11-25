import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import sys
from torch.nn import functional as F

from torchreid.utils.load_weights import load_weights


###V2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

model_urls = {
    # top1 = 71.8
    'imagenet': 'https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2-36f4e720.pth'
}


__all__ = ['mobilenetv2']


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, loss, width_mult=1., expand_ratio=6):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # expand_coeff, channels, num_blocks, stride
            [1,  16, 1, 1],
            [expand_ratio,  24, 2, 2],
            [expand_ratio,  32, 3, 2],
            [expand_ratio,  64, 4, 2],
            [expand_ratio,  96, 3, 1],
            [expand_ratio, 160, 3, 2],
            [expand_ratio, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for expand_coeff, channels, num_blocks, stride in self.cfgs:
            output_channel = int(channels * width_mult)
            layers.append(block(input_channel, output_channel, stride, expand_coeff))
            input_channel = output_channel
            for i in range(1, num_blocks):
                layers.append(block(input_channel, output_channel, 1, expand_coeff))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        #self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        #self.dropout = nn.Dropout(.2)
        self.classifier = nn.Linear(output_channel, num_classes)
        self.loss = loss

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        #x = self.avgpool(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        #x = self.dropout(x)
        #x = self.classifier(x)

        if not self.training:
            return x

        y = self.classifier(x)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
        return x


def mobilenetv2(num_classes, loss, pretrained='imagenet', **kwargs):
    model = MobileNetV2(num_classes, loss, **kwargs)
    if pretrained == 'imagenet':
        model_url = model_urls['imagenet']
        load_weights(model, model_url)
    return model