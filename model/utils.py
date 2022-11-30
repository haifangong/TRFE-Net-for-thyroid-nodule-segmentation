import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch import Tensor
from torch import cosh, log
# from pydensecrf import densecrf

class BoundaryLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss

def boundary_loss(pred, target):
    loss_f = SoftDiceLoss()
    return loss_f(pred, target)

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, pred, target):
        num = target.size(0)
        probs = torch.sigmoid(pred)
        m1 = probs.view(num, -1)
        m2 = target.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score

def soft_dice(pred, target):
    loss_f = SoftDiceLoss()
    return loss_f(pred, target)

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

def densecrf(sigm_val):
    sigm_val = np.squeeze(sigm_val)
    d = densecrf.DenseCRF2D(224, 224, 2)
    U = np.expand_dims(-np.log(sigm_val), axis=0)
    U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0)
    unary = np.concatenate((U_, U), axis=0)
    unary = unary.reshape((2, -1))
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
    Q = d.inference(5)
    pred_raw_dcrf = np.argmax(Q, axis=0).reshape((224, 224)).astype(np.float32)
    predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])


def soft_mse(pred, target):
    loss_f = LogCoshLoss()
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


# from PraNet official implementation
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
