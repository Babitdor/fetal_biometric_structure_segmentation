from torch import optim, nn
import torch


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Apply sigmoid to logits
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice_loss = 1 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        bce_loss = self.bce(pred, target)

        return bce_loss + dice_loss
