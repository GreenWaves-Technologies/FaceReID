import torch
import torch.nn as nn


class DistillLoss(nn.Module):
    def __init__(self, teacher_model):
        super(DistillLoss, self).__init__()
        self.teacher = teacher_model
        self.criterion = nn.L1Loss()

    def forward(self, features, **kwargs):
        imgs = kwargs['imgs']
        with torch.no_grad():
            teacher_score = self.teacher(imgs)[1]  # teacher model is in the train mode, not eval
        return self.criterion(features, teacher_score)

