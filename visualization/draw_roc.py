import argparse
import os
from itertools import cycle

import matplotlib.axes
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloaders import tn3k, tg3k, tatn, ddti
from dataloaders import custom_transforms as trforms

# Model includes
from model.mtnet import MTNet
from model.trfe import TRFENet
from model.trfe1 import TRFENet1
from model.trfe2 import TRFENet2
from model.trfeplus import TRFEPLUS

from torchvision.models.segmentation.segmentation import deeplabv3_resnet50
from model.deeplab_v3_plus import Deeplabv3plus
from model.fcn import FCN8s
from model.segnet import SegNet
from model.unet import Unet
from model.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.cpfnet import CPFNet
from model.sgunet import SGUNet

def init_model(model_name, num_classes):
    if 'deeplab' in model_name:
        if 'resnet101' in model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=num_classes, os=16,
                                backbone_type='resnet101')
        elif 'resnet50' in model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=num_classes, os=16,
                                backbone_type='resnet50')
        elif 'resnet34' in model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=num_classes, os=16,
                                backbone_type='resnet34')
        elif 'v3' in model_name:
            net = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
        else:
            raise NotImplementedError
    elif 'unet' == model_name:
        net = Unet(in_ch=3, out_ch=1)
    elif 'trfe' in model_name:
        if model_name == 'trfe1':
            net = TRFENet1(in_ch=3, out_ch=1)
        elif model_name == 'trfe2':
            net = TRFENet2(in_ch=3, out_ch=1)
        elif model_name == 'trfe':
            net = TRFENet(in_ch=3, out_ch=1)
        elif 'trfeplus' in model_name:
            net = TRFEPLUS(in_ch=3, out_ch=1)
        batch_size = 8
    elif 'mtnet' in model_name:
        net = MTNet(in_ch=3, out_ch=1)
        batch_size = 8
    elif 'segnet' in model_name:
        net = SegNet(input_channels=3, output_channels=1)
    elif 'fcn' in model_name:
        net = FCN8s(1)
    elif 'ViT' in model_name:
        config_vit = CONFIGS_ViT_seg[model_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3  # 这里的n_skip含义不明,R50为3,别的用0
        if model_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(224 / 16), int(224 / 16))
        net = ViT_seg(config_vit, img_size=224, num_classes=config_vit.n_classes)
    elif 'cpfnet' in model_name:
        net = CPFNet(num_classes)
    elif 'sgunet' == model_name:
        net = SGUNet(n_classes=num_classes)
    else:
        raise NotImplementedError
    return net


def forward(model_name, net, inputs):
    if 'trfe' in model_name or model_name == 'mtnet':
        if 'trfesw' in model_name:
            outputs, _, _ = net.forward(inputs)
        else:
            outputs, _ = net.forward(inputs)
    elif 'cpfnet' in model_name:
        main_out = net(inputs)
        outputs = main_out[:,0,:,:].view(1, 1, 224, 224)
    elif 'v3' in model_name:
        outputs = net(inputs)['out']
    else:
        outputs = net.forward(inputs)
    return outputs
    


def get_model_name(model_name: str):
    """规范化模型名称，用于plt的label"""
    print(model_name)
    if 'deeplab' in model_name:
        standard_model_name = 'Deeplabv3+'
    elif 'fcn' in model_name:
        standard_model_name = "FCN"
    elif 'unet' == model_name or 'unet_origin' == model_name:
        standard_model_name = "Unet"
    elif 'trfe' in model_name:
        standard_model_name = "TRFE"
        if model_name == 'trfeplus':
            standard_model_name = 'TRFE+'
    elif 'sgunet' in model_name:
        standard_model_name = "SGUNet"
    elif 'mtnet' in model_name:
        standard_model_name = "MTNet"
    elif 'segnet' in model_name:
        standard_model_name = 'SegNet'
    elif 'ViT' in model_name:
        standard_model_name = "Trans-Unet"
    elif 'Pranet' in model_name:
        standard_model_name = "PraNet"
    elif 'CPFNet' in model_name or 'cpfnet' in model_name:
        standard_model_name = "CPF-Net"
    else:
        raise NotImplementedError
    return standard_model_name


def get_pred_paths(model_name: str, dataset: str, fold: int):
    root = f'./results/test-{dataset}/{model_name}/fold{fold}'
    return [os.path.join(root, x) for x in os.listdir(root) if 's' in x]


def read_img(path, resize=None):
    img = Image.open(path).convert('L')
    if resize:
        img = img.resize(resize)
    img = np.array(img)
    return img


def get_gt_dict(dataset: str):
    gt_dict = {}
    if dataset == 'TN3K':
        data_dir = "/home/liguanbin/TRFE-Net-for-thyroid-nodule-segmentation-main/data/tn3k/test-mask/"
    elif dataset == 'DDTI':
        data_dir = "/home/liguanbin/TRFE-Net-for-thyroid-nodule-segmentation-main/data/DDTI/mask/"
    else:
        raise NotImplementedError
    img_paths = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    for idx, path in enumerate(img_paths):
        gt_dict[os.path.basename(img_paths[idx])] = read_img(path, resize=(224, 224))
    return gt_dict


def calculate_roc(model_name: str, dataset: str, save_dir: str):
    y_list = []
    score_list = []
    # pred_paths = get_pred_paths(model_name, dataset, fold)
    testloader = DataLoader(datasets[dataset], batch_size=1, shuffle=False, num_workers=0)
    gt_dict = get_gt_dict(dataset)
    net = init_model(model_name, num_classes=1)
    for fold in range(5):
        model_path = f"/home/liguanbin/TRFE-Net-for-thyroid-nodule-segmentation-main/run/{model_name}/fold{fold}/{model_name}_best.pth"
        if model_name == 'trfe':
            model_path = f"/home/liguanbin/TRFE-Net-for-thyroid-nodule-segmentation-main/run_old/trfe/fold{fold}/trfe_best.pth"
        # elif model_name == 'trfesw':
        #     model_path = f"/home/liguanbin/TRFE-Net-for-thyroid-nodule-segmentation-main/run/trfesw_ori/fold{fold}/trfesw_best.pth"
        net.load_state_dict(torch.load(model_path))
        net.cuda()
        net.eval()
        # for sample_batched in tqdm(test_loader):
        #     inputs, labels = sample_batched['image'], sample_batched['label']
        #     inputs = Variable(inputs, requires_grad=False)
        #     labels = Variable(labels)
        #     labels = labels.cuda()
        #     inputs = inputs.cuda()
        #     # print(inputs.size())
        #     # exit(0)
        #     # score = read_img(path, resize=(224, 224)) / 255
        #     # score =  (score - np.min(score) + 1e-8) / (np.max(score) - np.min(score) + 1e-8)
        #     output = forward(model_name, net, inputs)
        #     score = output[0][0].detach().cpu().numpy()
        #     score_list.append(score)
        #     gt = gt_dict[sample_batched.get('label_name')[0]] / 255
        #     # print(gt.shape, gt.max(), gt.min())
        #     y_list.append(gt)
        for sample_batched in tqdm(testloader):
                inputs, labels, label_name, size = sample_batched['image'], sample_batched['label'], sample_batched.get(
                    'label_name'), sample_batched['size']
                # print(inputs.size())
                # exit(0)
                inputs = Variable(inputs, requires_grad=False)
                labels = Variable(labels)
                labels = labels.cuda()
                inputs = inputs.cuda()
                if 'trfe' in model_name or 'mtnet' in model_name:                
                    if 'trfesw' in model_name:
                        nodule_pred, gland_pred, _ = net.forward(inputs)
                    else:
                        nodule_pred, gland_pred = net.forward(inputs)
                    gland_pred = torch.sigmoid(gland_pred)
                    # gland_pred = np.round(torch.sigmoid(gland_pred).cpu().data.numpy())
                elif 'v3' in model_name:
                    nodule_pred = net(inputs)['out']
                elif 'cpfnet' in model_name:
                    nodule_pred = net(inputs)
                else:
                    nodule_pred = net.forward(inputs)
                prob_pred = torch.sigmoid(nodule_pred)
                score_list.append(prob_pred.detach().cpu().numpy())
                gt = gt_dict[sample_batched.get('label_name')[0]] / 255
                # print(gt.shape, gt.max(), gt.min())
                y_list.append(gt)
    y = np.concatenate(y_list).flatten().round()
    score = np.concatenate(score_list).flatten()
    fpr, tpr, threshold = roc_curve(y, score)
    # print(y.dtype)
    # df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold})
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # df.to_excel(f'{save_dir}/{model_name}.xlsx')
    # print(f'saved at :{save_dir}/{model_name}.xlsx')
    return fpr, tpr, threshold


def plot_roc(fpr, tpr, ax: matplotlib.axes.Axes, model_name, color):
    roc_auc = auc(fpr, tpr)
    if model_name == 'trfe':
        ax.plot(fpr, tpr, label=f'{get_model_name(model_name)}(AUC=%0.4f)' % roc_auc, color='red')
    elif model_name == 'trfesw':
        ax.plot(fpr, tpr, label=f'{get_model_name(model_name)}(AUC=%0.4f)' % roc_auc, color='lime')
    else:
        ax.plot(fpr, tpr, label=f'{get_model_name(model_name)}(AUC=%0.4f)' % roc_auc, color=color)
    print(f"{model_name}'s roc curve has been plottd")


def draw_roc_func(dataset, from_scratch=False):
    root = "run/"
    models_name = ['unet', 'sgunet', 'trfe', 'fcn', 'segnet', 'deeplab-resnet50', 'cpfnet', 'R50-ViT-B_16', 'trfesw-refine']
    color_list = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'gold', 'cyan', 'lime']
    # models_name = ['trfesw-refine']
    # color_list = ['lime']
    # marker = ["o", "v", "^", "<", ">", "D", 'd']
    # marker = cycle(marker)
    fig, ax = plt.subplots()
    ax.set_xlim([0.0, 1.0])
    if dataset == 'TN3K':
        ax.set_ylim([0.75, 1.0])
    elif dataset == 'DDTI':
        ax.set_ylim([0.5, 1.0])
    ax.set_xlabel('1-Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title(f'{dataset}')
    for i, model_name in enumerate(models_name):
        # if os.path.exists(f'./results/ruc/{dataset}/fold{fold}/{model_name}.xlsx') and not from_scratch:
        #     df = pd.read_excel(f'./results/ruc/{dataset}/fold{fold}/{model_name}.xlsx')
        #     # fpr, tpr, threshold = df['fpr'], df['tpr'], df['threshold']
        #     fpr = df['fpr']
        #     tpr = df['tpr']
        # else:
        #     print(f'./results/ruc/fold{fold}/{model_name}.xlsx not found and calculate_roc from scratch.')
        fpr, tpr, threshold = calculate_roc(model_name, dataset, f'no use')
        # plot_roc(fpr, tpr, ax, net.__class__.__name__, next(marker))
        # print(np.array(fpr))
        # fpr = np.mean(np.array(fpr), axis=0)
        # tpr = np.mean(np.array(tpr), axis=0)
        plot_roc(fpr, tpr, ax, model_name, color_list[i])
    ax.legend(loc="lower right")
    # plt.show()
    if not os.path.exists(f'./results/ruc/{dataset}/'):
        os.makedirs(f'./results/ruc/{dataset}/')
    fig.savefig(f'./results/ruc/{dataset}/{dataset}.pdf')
    print(f'figure saved at: ./results/ruc/{dataset}/{dataset}.pdf')


if __name__ == '__main__':
    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(224, 224)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])
    # datasets=['TN3K', 'DDTI']
    datasets = {
        'TN3K': tn3k.TN3K(mode='test', transform=composed_transforms_ts, return_size=True),
        'DDTI': ddti.DDTI(transform=composed_transforms_ts, return_size=True)
    }
    for dataset in datasets.keys():
        draw_roc_func(dataset, from_scratch=True)
    # net = TRFESW(in_ch=3, out_ch=1)
    # net.load_state_dict(torch.load("/home/liguanbin/TRFE-Net-for-thyroid-nodule-segmentation-main/run/trfesw_old/fold1/trfesw_best.pth"))
    # print(net)
