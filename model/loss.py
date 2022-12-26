import torch.nn.functional as F
import torch
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    loss = torch.nn.CrossEntropyLoss()
    return loss(output, target)

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
