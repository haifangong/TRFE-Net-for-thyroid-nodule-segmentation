import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, pred, target):
        num = target.size(0)
        probs = F.sigmoid(pred)
        m1 = probs.view(num, -1)
        m2 = target.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


def soft_dice(pred, target, size_average=True, batch_average=True):
    loss_f = SoftDiceLoss()
    return loss_f(pred, target)


def load_pretrain_model(net, weights):
    net_keys = list(net.state_dict().keys())
    weights_keys = list(weights.keys())
    # assert(len(net_keys) <= len(weights_keys))
    i = 0
    j = 0
    while i < len(net_keys) and j < len(weights_keys):
        name_i = net_keys[i]
        name_j = weights_keys[j]
        if net.state_dict()[name_i].shape == weights[name_j].shape:
            net.state_dict()[name_i].copy_(weights[name_j].cpu())
            i += 1
            j += 1
        else:
            i += 1
    # print i, len(net_keys), j, len(weights_keys)
    return net


def load_pretrain_model_fast(net, weights):
    net_keys = net.state_dict().keys()
    weights_keys = weights.keys()
    # assert(len(net_keys) <= len(weights_keys))
    i = 0
    j = 0
    while i < len(net_keys) and j < len(weights_keys):
        name_i = net_keys[i]
        name_j = weights_keys[j]
        if net.state_dict()[name_i].shape == weights[name_j].shape:
            net.state_dict()[name_i].copy_(weights[name_j].cpu())
            i += 1
            j += 1
        else:
            break
            i += 1
    # print i, len(net_keys), j, len(weights_keys)
    return net
