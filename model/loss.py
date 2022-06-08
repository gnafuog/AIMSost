import torch
from torch.nn import Module, CrossEntropyLoss


class Loss(Module):

    def __init__(self, weight=None):
        super(Loss, self).__init__()
        self.stage_loss = CrossEntropyLoss(weight).to(torch.float32)

    def forward(self, y_pred, y_true):
        stage_loss = self.stage_loss(y_pred, y_true.long())
        return stage_loss
