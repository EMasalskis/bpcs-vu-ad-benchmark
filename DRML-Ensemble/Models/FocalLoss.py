import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, use_sigmoid=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        r"""
        Focal loss
        :param pred: shape=(B,  HW)
        :param label: shape=(B, HW)
        """
        if self.use_sigmoid:
            pred = self.sigmoid(pred)
        pred = pred.view(-1)
        label = target.view(-1)
        pos = torch.nonzero(label > 0).squeeze(1)
        pos_num = max(pos.numel(), 1.0)
        mask = ~(label == -1)
        pred = pred[mask]
        label = label[mask]
        focal_weight = self.alpha * (label - pred).abs().pow(self.gamma) * (label > 0.0).float() + (
                    1 - self.alpha) * pred.abs().pow(self.gamma) * (label <= 0.0).float()
        loss = F.binary_cross_entropy(pred, label.float(), reduction='none') * focal_weight
        return loss.sum() / pos_num

if __name__ == '__main__':
    a = torch.randn(200)
    a = nn.Sigmoid()(a)
    t = torch.ones(200)
    loss = FocalLoss()
    print(loss(a,t))