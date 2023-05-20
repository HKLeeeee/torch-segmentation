import torch.nn.functional as F
import torch
import torch.nn as nn
from monai.losses.dice import DiceLoss, DiceCELoss

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse(output, target):
    loss = torch.nn.MSELoss()
    return loss(output, target)

def bce_logitloss(output, target):
    loss = nn.BCEWithLogitsLoss()    
    # return loss(output.squeeze(), target.squeeze())
    return loss(output, target)

def nmae(output, target):
    error = output - target
    absolute_error = torch.absolute(error)
    return torch.mean(absolute_error) / torch.mean(target)

def cross_entropy(output, target):
    print(output.shape, target.shape)
    loss = torch.nn.CrossEntropyLoss()
    target = target.squeeze().long()
    return loss(output, target)

def dice_loss(output, target):
    loss = DiceLoss(to_onehot_y=True)
    target = target.long()
    return loss(output, target)

def dice_ce_loss(output, target):
    loss = DiceCELoss(to_onehot_y=True)
    target = target.long()
    return loss(output, target)
