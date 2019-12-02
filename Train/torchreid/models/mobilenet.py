import torch.nn as nn
import torch.nn.functional as F

from torchreid.utils.load_weights import load_weights


pretrained_settings = {
    'mobilenet': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/mobilenet_feature.pth'
        }
    }
}


def conv_bn(inp,oup,stride):
    return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride, dilation=1, padding=1):
    return nn.Sequential(
            nn.Conv2d(inp,inp, kernel_size=3, stride=stride,groups = inp, dilation=dilation, padding=padding, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
    )


class BasicMobileNet(nn.Module):
    def __init__(self, num_classes, loss={'xent'}):
        super(BasicMobileNet, self).__init__()
        layers = []
        self.base = nn.ModuleList(layers)
        self.classifier = None
        self.loss = loss

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)

        if not self.training:
            return x

        y = self.classifier(x)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class MobileNet(BasicMobileNet):
    def __init__(self, num_classes, loss={'xent'}):
        super(MobileNet, self).__init__(num_classes, loss)
        layers = []
        layers += [conv_bn(3, 32, 2)]
        layers += [conv_dw(32, 64, 1)]
        layers += [conv_dw(64, 128, 2)]
        layers += [conv_dw(128, 128, 1)]
        layers += [conv_dw(128, 256, 2)]
        layers += [conv_dw(256, 256, 1)]
        layers += [conv_dw(256, 512, 2)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 512, 1)]
        layers += [conv_dw(512, 1024, 2)]
        layers += [conv_dw(1024, 1024, 1)]

        self.base = nn.ModuleList(layers)
        self.classifier = nn.Linear(1024, num_classes)  # shouldn't be hardcoded!
        self.loss = loss


class ThinMobileNet(BasicMobileNet):
    def __init__(self, num_classes, loss={'xent'}):
        super(ThinMobileNet, self).__init__(num_classes, loss)
        layers = []
        layers += [conv_bn(3, 32, 2)]
        layers += [conv_dw(32, 64, 1)]
        layers += [conv_dw(64, 128, 2)]
        layers += [conv_dw(128, 128, 1)]
        layers += [conv_dw(128, 256, 2)]
        layers += [conv_dw(256, 256, 1)]
        layers += [conv_dw(256, 384, 2)]
        layers += [conv_dw(384, 384, 1)]
        layers += [conv_dw(384, 384, 1)]
        layers += [conv_dw(384, 384, 1)]
        layers += [conv_dw(384, 384, 1)]
        layers += [conv_dw(384, 384, 1)]
        layers += [conv_dw(384, 384, 2)]
        layers += [conv_dw(384, 384, 1)]

        self.base = nn.ModuleList(layers)
        self.classifier = nn.Linear(384, num_classes)  # shouldn't be hardcoded!
        self.loss = loss


class DilatedThinMobileNet(BasicMobileNet):
    def __init__(self, num_classes, loss={'xent'}):
        super(DilatedThinMobileNet, self).__init__(num_classes, loss)
        layers = []
        layers += [conv_bn(3, 32, 2)]
        layers += [conv_dw(32, 64, 1)]
        layers += [conv_dw(64, 128, 2)]
        layers += [conv_dw(128, 128, 1)]
        layers += [conv_dw(128, 256, 2)]
        layers += [conv_dw(256, 256, 1)]
        layers += [conv_dw(256, 256, 1, dilation=2, padding=2)]
        layers += [conv_dw(256, 256, 1, dilation=2, padding=2)]
        layers += [conv_dw(256, 256, 1, dilation=2, padding=2)]
        layers += [conv_dw(256, 256, 1, dilation=2, padding=2)]
        layers += [conv_dw(256, 256, 1, dilation=2, padding=2)]
        layers += [conv_dw(256, 256, 1, dilation=2, padding=2)]
        layers += [conv_dw(256, 512, 2, dilation=2, padding=2)]
        layers += [conv_dw(512, 512, 1, dilation=2, padding=2)]

        self.base = nn.ModuleList(layers)
        self.classifier = nn.Linear(512, num_classes)  # shouldn't be hardcoded!
        self.loss = loss


def mobilenet(num_classes, loss, pretrained='imagenet', **kwargs):
    model = MobileNet(num_classes, loss, **kwargs)
    if pretrained == 'imagenet':
        model_url = pretrained_settings['mobilenet']['imagenet']['url']
        load_weights(model, model_url)
    return model


def thin_mobilenet(num_classes, loss, pretrained='imagenet', **kwargs):
    model = ThinMobileNet(num_classes, loss, **kwargs)
    if pretrained == 'imagenet':
        model_url = pretrained_settings['mobilenet']['imagenet']['url']
        load_weights(model, model_url)
    return model


def dilated_thin_mobilenet(num_classes, loss, pretrained='imagenet', **kwargs):
    model = DilatedThinMobileNet(num_classes, loss, **kwargs)
    if pretrained == 'imagenet':
        model_url = pretrained_settings['mobilenet']['imagenet']['url']
        load_weights(model, model_url)
    return model