from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torchreid.losses.cross_entropy_loss import CrossEntropyLoss


class FaceLoss(nn.Module):
    def __init__(self, out_features, use_gpu, label_smooth):
        super(FaceLoss, self).__init__()
        self.cross_entropy = CrossEntropyLoss(num_classes=out_features, use_gpu=use_gpu, label_smooth=label_smooth)

    def forward(self, input, **kwargs):
        pass


class ArcFace(FaceLoss):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.40, easy_margin=False, use_gpu=True, label_smooth=True):
        super(ArcFace, self).__init__(out_features, use_gpu, label_smooth)
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        #self.cross_entropy = CrossEntropyLoss(num_classes=out_features, use_gpu=use_gpu, label_smooth=args.label_smooth)

    def forward(self, input, **kwargs):
        label = kwargs['targets']
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        loss = self.cross_entropy(output, **kwargs)
        return loss


class CosFace(FaceLoss):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40, use_gpu=True, label_smooth=True):
        super(CosFace, self).__init__(out_features, use_gpu, label_smooth)
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, **kwargs):
        label = kwargs['targets']
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        loss = self.cross_entropy(output, **kwargs)
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereFace(FaceLoss):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4, use_gpu=True, label_smooth=True):
        super(SphereFace, self).__init__(out_features, use_gpu, label_smooth)
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, **kwargs):
        label = kwargs['targets']
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        loss = self.cross_entropy(output, **kwargs)
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'


def focal_loss(input_values, gamma):
    #print(input_values)
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class AMSoftmaxLoss(nn.Module):
    """Computes the AM-Softmax loss with cos or arc margin"""
    margin_types = ['cos', 'arc']

    def __init__(self, margin_type='cos', gamma=0., m=0.35, s=30, t=1.):
        super(AMSoftmaxLoss, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t

    def forward(self, cos_theta, **kwargs):
        target = kwargs['targets']
        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m #cos(theta+m)
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.:
            return F.cross_entropy(self.s*output, target)

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s*output, target)

        return focal_loss(F.cross_entropy(self.s*output, target, reduction='none'), self.gamma)

