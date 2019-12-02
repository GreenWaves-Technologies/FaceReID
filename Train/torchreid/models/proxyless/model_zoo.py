import os
import json

import torch

from torchreid.models.proxyless.utils import download_url, load_url
from .nas_modules import ProxylessNASNets
from torchreid.utils.load_weights import load_weights


def proxyless_base(num_classes, loss, pretrained=True, net_config=None, net_weight=None):
    assert net_config is not None, "Please input a network config"
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))
    net = ProxylessNASNets.build_from_config(net_config_json, num_classes, loss)

    if 'bn' in net_config_json:
        net.set_bn_param(bn_momentum=net_config_json['bn']['momentum'], bn_eps=net_config_json['bn']['eps'])
    else:
        net.set_bn_param(bn_momentum=0.1, bn_eps=1e-3)

    if pretrained:
        assert net_weight is not None, "Please specify network weights"
        init_path = download_url(net_weight)
        load_weights(net, init_path)

    return net


from functools import partial

def proxyless_cpu(num_classes, loss, pretrained, **kwargs):
    return proxyless_base(num_classes, loss,
                          pretrained=pretrained,
                          net_config="http://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.config",
                          net_weight="http://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.pth")


def proxyless_gpu(num_classes, loss, pretrained, **kwargs):
    return proxyless_base(num_classes, loss,
                          pretrained=pretrained,
                          net_config="http://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.config",
                          net_weight="http://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.pth")


def proxyless_mobile(num_classes, loss, pretrained, **kwargs):
    return proxyless_base(num_classes, loss,
                          pretrained=pretrained,
                          net_config="http://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.config",
                          net_weight="http://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.pth")