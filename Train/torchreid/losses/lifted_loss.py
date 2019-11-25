import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class LiftedLoss(nn.Module):
    def __init__(self, margin=1):
        super(LiftedLoss, self).__init__()
        self.margin = margin

    def forward(self, score, **kwargs):
        """
          Lifted loss, per "Deep Metric Learning via Lifted Structured Feature Embedding" by Song et al
          Implemented in `pytorch`
        """
        target = kwargs['targets']
        loss = 0
        counter = 0
        bsz = score.size(0)
        mag = (score ** 2).sum(1).expand(bsz, bsz)
        sim = score.mm(score.transpose(0, 1))
        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = torch.nn.functional.relu(dist).sqrt()
        target = target.cpu()
        target_set = np.unique(target.numpy()).tolist()
        mask_other_targets = {t : target != t for t in target_set}
        for i in range(bsz):
            t_i_value = target[i].item()
            mask_same_targets = target == t_i_value
            mask_same_targets[i] = 0
            l_nj = torch.sum(torch.exp((self.margin - dist[mask_same_targets][:, mask_other_targets[t_i_value]])), dim=1)
            l_ni = (self.margin - dist[i][mask_other_targets[t_i_value]]).exp().sum()
            l_ni = torch.unsqueeze(l_ni, l_ni.dim())
            l_ni = l_ni.expand(l_nj.shape)
            l_n = torch.log(l_ni + l_nj)
            l_p = dist[i, mask_same_targets]
            loss += torch.sum(torch.nn.functional.relu(l_n + l_p) ** 2)
            counter += torch.nonzero(mask_same_targets).nelement()
        if counter == 0:
            print('Panic!')
            return 0
        return loss / (2 * counter) / 2
