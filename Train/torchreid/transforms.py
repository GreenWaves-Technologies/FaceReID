from __future__ import absolute_import
from __future__ import division

from torchvision.transforms import *
import torch

from PIL import Image
import random
import time
import numpy as np

from torchreid.utils.quantization import int_bits
from torchreid.utils.quantization import integerize

class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


class GrayScale(object):
    def __call__(self, img):
        return img.convert('L')


class ToTensorNoNorm(object):
    def __call__(self, pic):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class Quantize(object):
    def __init__(self, bits, int_bits=2):
        self.bits = bits
        self.int_bits = int_bits
        self.float_bits = self.bits - self.int_bits - 1
        self.counter = 0

    def __call__(self, tensor):
        if self.counter >= 100:
            return integerize(tensor, self.float_bits, self.bits)
        else:
            self.counter += 1
            return tensor


class ImShow(object):
    def __call__(self, img):
        img.show()
        time.sleep(5)
        return img


def build_transforms(height, width, is_train, grayscale=False, no_normalize=False, quantization=False, bits=16,
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], **kwargs):
    """Build transforms

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - is_train (bool): train or test phase.
    """
    
    # use imagenet mean and std as default

    grayscale_mean = [0.449]
    grayscale_std = [0.225]

    normalize = Normalize(mean=mean, std=std)

    transforms = []

    if is_train:
        transforms += [Random2DTranslation(height, width)]
        transforms += [RandomHorizontalFlip()]
    else:
        transforms += [Resize((height, width))]

    if grayscale:
        transforms += [GrayScale()]
    #transforms += [ImShow()]
    if no_normalize:
        transforms += [ToTensorNoNorm()]
    else:
        transforms += [ToTensor()]
        transforms += [normalize]

        if quantization:
            transforms += [Quantize(bits)]

    transforms = Compose(transforms)

    return transforms