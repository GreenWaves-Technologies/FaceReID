from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ring_loss import RingLoss
from .sphere_losses import ArcFace, CosFace, SphereFace, AMSoftmaxLoss
from .lifted_loss import LiftedLoss
from .distill_loss import DistillLoss

from torchreid.utils.load_weights import load_weights
from torchreid import models
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.iotools import check_isfile


class Criterion:
    def __init__(self, name, calc_loss, weight):
        self.name = name
        self.calc_loss = calc_loss
        self.weight = weight
        self.train_stats = AverageMeter()
        self.test_stats = AverageMeter()


def choose_losses(args, data_manager, model, use_gpu):
    criteria = []

    if args.xent_loss == 'am':
        criteria.append(Criterion('am', AMSoftmaxLoss(), args.weight_xent))
    elif args.xent_loss is not None:
        criteria.append(Criterion('xent', CrossEntropyLoss(num_classes=data_manager.num_train_pids, use_gpu=use_gpu,
                                                           label_smooth=args.label_smooth), args.weight_xent))

    if args.euclid_loss:
        if args.euclid_loss == 'triplet':
            loss = TripletLoss(margin=args.margin)
        elif args.euclid_loss == 'lifted':
            loss = LiftedLoss(margin=args.margin)
        else:
            raise KeyError('Unknown euclidean loss: {}'.format(args.euclid_loss))
        criteria.append(Criterion('euclid', loss, args.weight_euclid))

    if args.face_loss:
        input_channels = 3
        if args.grayscale:
            input_channels = 1
        features = model(torch.zeros(1, input_channels, args.height, args.width))
        feature_vector_size = features.shape[1]
        if args.face_loss == 'arc':
            loss = ArcFace(feature_vector_size, data_manager.num_train_pids)
        elif args.face_loss == 'cos':
            loss = CosFace(feature_vector_size, data_manager.num_train_pids)
        elif args.face_loss == 'sphere':
            loss = SphereFace(feature_vector_size, data_manager.num_train_pids)
        elif args.face_loss == 'am':
            loss = AMSoftmaxLoss()
        else:
            raise KeyError('Unknown face loss: {}'.format(args.face_loss))
        if use_gpu:
            loss = loss.cuda()
        criteria.append(Criterion('face', loss, args.weight_face))

    if args.teacher_arch:
        print("Initializing teacher model: {}".format(args.teacher_arch))
        teacher_model = models.init_model(name=args.teacher_arch, num_classes=data_manager.num_train_pids, loss={'xent', 'htri'}, pretrained=False)
        if not args.load_teacher_weights or not check_isfile(args.load_teacher_weights):
            print('Teacher model checkpoint wasn\'t provided!')
            return
        load_weights(teacher_model, args.load_teacher_weights)
        print("Loaded pretrained weights from '{}' for teacher model".format(args.load_teacher_weights))
        if use_gpu:
            teacher_model = nn.DataParallel(teacher_model).cuda()
        criteria.append(Criterion('distill', DistillLoss(teacher_model), args.weight_distill))

    return criteria


def DeepSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
