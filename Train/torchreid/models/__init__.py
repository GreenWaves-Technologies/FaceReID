from __future__ import absolute_import

from .resnet import *
from .resnetmid import *
from .resnext import *
from .senet import *
from .densenet import *
from .inceptionresnetv2 import *
from .inceptionv4 import *
from .xception import *

from .nasnet import *
from .mobilenetv2 import *
from .shufflenet import *
from .shufflenetv2 import *
from .squeezenet import *
from .mobilenet import *
from .mnasnet import *
from .espnet import *

from .mudeep import *
from .hacnn import *
from .pcb import *
from .mlfn import *
from .proxyless.model_zoo import *


__model_factory = {
    # image classification models
    'resnet50': resnet50,
    'resnet50_fc512': resnet50_fc512,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext50_32x4d_fc512': resnext50_32x4d_fc512,
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'densenet121': densenet121,
    'densenet121_fc512': densenet121_fc512,
    'inceptionresnetv2': InceptionResNetV2,
    'inceptionv4': inceptionv4,
    'xception': xception,
    # lightweight models
    'nasnsetmobile': nasnetamobile,
    'mobilenetv2': mobilenetv2,
    'shufflenet': shufflenet,
    'shufflenetv2': shufflenetv2,
    'shufflenetv2small': shufflenetv2small,
    'proxyless_cpu': proxyless_cpu,
    'proxyless_gpu': proxyless_gpu,
    'proxyless_mobile': proxyless_mobile,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_0_fc512': squeezenet1_0_fc512,
    'squeezenet1_1': squeezenet1_1,
    'mobilenetv1': mobilenet,
    'thinmobilenetv1': thin_mobilenet,
    'dilatedthinmobilenet': dilated_thin_mobilenet,
    'mnasnet': mnasnet,
    'espnet': espnet,
    # reid-specific models
    'mudeep': MuDeep,
    'resnet50mid': resnet50mid,
    'hacnn': HACNN,
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'mlfn': mlfn,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)