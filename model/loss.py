import torch.nn.functional as F
import torch
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    return torch.nn.CrossEntropyLoss(output, target)

def mse(output, target):
    return torch.nn.MSELoss(output, target)

def bce_logitloss(output, target):
    loss = nn.BCEWithLogitsLoss()    
    return loss(output.squeeze(), target.squeeze())

def nmae(output, target):
    error = output - target
    absolute_error = torch.absolute(error)
    return torch.mean(absolute_error) / torch.mean(target)
