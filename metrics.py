import torch
import torch.nn.functional as F

def iou(y_true, y_pred):
    smooth = 1e-15
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score.item()

def dice_coef(y_true, y_pred, smooth=1e-15):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    intersection = (y_true * y_pred).sum()
    dice_score = (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
    return dice_score.item()

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
