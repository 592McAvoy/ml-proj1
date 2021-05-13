import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

