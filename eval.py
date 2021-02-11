import argparse
import os
import time

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders import custom_transforms as trforms

# Dataloaders includes
from dataloaders import tn3k
from dataloaders import utils

# Custom includes
from model.unet import Unet
from model.segnet import SegNet
from model.fcn import FCN8s
from model.mtnet import MTNet
from model.trfe import TRFENet
from model.trfe1 import TRFENet1
from model.trfe2 import TRFENet2
from model.deeplab_v3_plus import Deeplabv3plus


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-model_name', type=str,
                        default='trfe')  # unet, mtnet, segnet, deeplab-resnet50, fcn, trfe, trfe1, trfe2
    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-output_stride', type=int, default=16)
    parser.add_argument('-load_path', type=str, default='./run/run_1/trfe_best.pth')
    parser.add_argument('-save_dir', type=str, default='./results')
    parser.add_argument('-test_dataset', type=str, default='TN3K')
    parser.add_argument('-test_fold', type=str, default='/test')
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if 'deeplab' in args.model_name:
        if 'resnet101' in args.model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=args.num_classes, os=args.output_stride,
                                backbone_type='resnet101')
        elif 'resnet50' in args.model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=args.num_classes, os=args.output_stride,
                                backbone_type='resnet50')
        elif 'resnet34' in args.model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=args.num_classes, os=args.output_stride,
                                backbone_type='resnet34')
    elif 'unet' in args.model_name:
        net = Unet(in_ch=3, out_ch=1)
    elif 'trfe' in args.model_name:
        if args.model_name == 'trfe':
            net = TRFENet(in_ch=3, out_ch=1)
        elif args.model_name == 'trfe1':
            net = TRFENet1(in_ch=3, out_ch=1)
        elif args.model_name == 'trfe2':
            net = TRFENet2(in_ch=3, out_ch=1)
    elif 'mtnet' in args.model_name:
        net = MTNet(in_ch=3, out_ch=1)
    elif 'segnet' in args.model_name:
        net = SegNet(input_channels=3, output_channels=1)
    elif 'fcn' in args.model_name:
        net = FCN8s(1)
    else:
        raise NotImplementedError
    net.load_state_dict(torch.load(args.load_path))
    net.cuda()

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    if args.test_dataset == 'TN3K':
        test_data = tn3k.TN3K(mode='test', transform=composed_transforms_ts, return_size=True)

    save_dir = args.save_dir + args.test_fold + '-' + args.test_dataset + os.sep + args.model_name + os.sep
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    num_iter_ts = len(testloader)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    net.cuda()
    net.eval()
    start_time = time.time()
    with torch.no_grad():
        total_iou = 0
        for sample_batched in tqdm(testloader):
            inputs, labels, label_name, size = sample_batched['image'], sample_batched['label'], sample_batched[
                'label_name'], sample_batched['size']
            inputs = Variable(inputs, requires_grad=False)
            labels = Variable(labels)
            labels = labels.cuda()
            inputs = inputs.cuda()
            if 'trfe' in args.model_name or 'mtnet' in args.model_name:
                outputs, _ = net.forward(inputs)
            else:
                outputs = net.forward(inputs)
            prob_pred = torch.sigmoid(outputs)
            iou = utils.get_iou(prob_pred, labels)
            total_iou += iou

            shape = (size[0, 0], size[0, 1])
            prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data
            save_data = prob_pred[0]
            save_png = save_data[0].numpy()
            save_png = np.round(save_png)
            save_png = save_png * 255
            save_png = save_png.astype(np.uint8)
            save_path = save_dir + label_name[0]
            if not os.path.exists(save_path[:save_path.rfind('/')]):
                os.makedirs(save_path[:save_path.rfind('/')])
            cv2.imwrite(save_dir + label_name[0], save_png)

    print(args.model_name + ' iou:' + str(total_iou / len(testloader)))
    duration = time.time() - start_time
    print("-- %s contain %d images, cost time: %.4f s, speed: %.4f s." % (
        args.test_dataset, num_iter_ts, duration, duration / num_iter_ts))
    print("------------------------------------------------------------------")


if __name__ == '__main__':
    args = get_arguments()
    main(args)
