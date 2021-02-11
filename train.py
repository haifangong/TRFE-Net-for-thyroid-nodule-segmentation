import argparse
import glob
import os
import random
import socket
import time
from datetime import datetime

import numpy as np

# PyTorch includes
import torch
import torch.optim as optim

# Tensorboard include
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

# Dataloaders includes
from dataloaders import tn3k, tg3k, tatn
from dataloaders import custom_transforms as trforms
from dataloaders import utils

# Model includes
from model.deeplab_v3_plus import Deeplabv3plus
from model.fcn import FCN8s
from model.mtnet import MTNet
from model.segnet import SegNet
from model.trfe import TRFENet
from model.trfe1 import TRFENet1
from model.trfe2 import TRFENet2
from model.unet import Unet

# Loss function includes
from model.utils import soft_dice


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    ## Model settings
    parser.add_argument('-model_name', type=str,
                        default='unet')  # unet, trfe, trfe1, trfe2, mtnet, segnet, deeplab-resnet50, fcn
    parser.add_argument('-criterion', type=str, default='Dice')
    parser.add_argument('-pretrain', type=str, default='None')  # THYROID

    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-output_stride', type=int, default=16)

    ## Train settings
    parser.add_argument('-dataset', type=str, default='TN3K')  # TN3K, TG3K, TATN
    parser.add_argument('-fold', type=str, default='0')
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-nepochs', type=int, default=60)
    parser.add_argument('-resume_epoch', type=int, default=0)

    ## Optimizer settings
    parser.add_argument('-naver_grad', type=str, default=1)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-update_lr_every', type=int, default=10)
    parser.add_argument('-weight_decay', type=float, default=5e-4)

    ## Visualization settings
    parser.add_argument('-save_every', type=int, default=10)
    parser.add_argument('-log_every', type=int, default=40)
    parser.add_argument('-load_path', type=str, default='')
    parser.add_argument('-run_id', type=int, default=-1)
    parser.add_argument('-use_eval', type=int, default=1)
    parser.add_argument('-use_test', type=int, default=1)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1234)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if args.resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    if args.run_id >= 0:
        run_id = args.run_id

    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
    log_dir = os.path.join(save_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    batch_size = args.batch_size

    if 'deeplab' in args.model_name:
        if 'resnet101' in args.model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=args.num_classes, os=args.output_stride, backbone_type='resnet101')
        elif 'resnet50' in args.model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=args.num_classes, os=args.output_stride, backbone_type='resnet50')
        elif 'resnet34' in args.model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=args.num_classes, os=args.output_stride, backbone_type='resnet34')
        else:
            raise NotImplementedError
    elif 'unet' in args.model_name:
        net = Unet(in_ch=3, out_ch=1)
    elif 'trfe' in args.model_name:
        if args.model_name == 'trfe1':
            net = TRFENet1(in_ch=3, out_ch=1)
        elif args.model_name == 'trfe2':
            net = TRFENet2(in_ch=3, out_ch=1)
        elif args.model_name == 'trfe':
            net = TRFENet(in_ch=3, out_ch=1)
        batch_size = 4
    elif 'mtnet' in args.model_name:
        net = MTNet(in_ch=3, out_ch=1)
        batch_size = 4
    elif 'segnet' in args.model_name:
        net = SegNet(input_channels=3, output_channels=1)
    elif 'fcn' in args.model_name:
        net = FCN8s(1)
    else:
        raise NotImplementedError

    if args.resume_epoch == 0:
        print('Training ' + args.model_name + ' from scratch...')
    else:
        load_path = os.path.join(save_dir, args.model_name + '_epoch-' + str(args.resume_epoch) + '.pth')
        print('Initializing weights from: {}...'.format(load_path))
        net.load_state_dict(torch.load(load_path))

    if args.pretrain == 'THYROID':
        net.load_state_dict(torch.load('./pre_train/thyroid-pretrain.pth', map_location=lambda storage, loc: storage))
        print('loading pretrain model......')

    torch.cuda.set_device(device=0)
    net.cuda()

    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum
    )

    if args.criterion == 'Dice':
        criterion = soft_dice
    else:
        raise NotImplementedError

    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.RandomHorizontalFlip(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    if args.dataset == 'TN3K':
        train_data = tn3k.TN3K(mode='train', transform=composed_transforms_tr, fold=args.fold)
        val_data = tn3k.TN3K(mode='val', transform=composed_transforms_ts, fold=args.fold)
    elif args.dataset == 'TG3K':
        train_data = tg3k.TG3K(mode='train', transform=composed_transforms_tr)
        val_data = tg3k.TG3K(mode='val', transform=composed_transforms_ts)
    elif args.dataset == 'TATN':
        train_data = tatn.TATN(mode='train', transform=composed_transforms_tr, fold=args.fold)
        val_data = tatn.TATN(mode='val', transform=composed_transforms_ts, fold=args.fold)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

    num_iter_tr = len(trainloader)
    num_iter_ts = len(testloader)
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = args.resume_epoch * len(train_data)
    print('nitrs: %d num_iter_tr: %d' % (nitrs, num_iter_tr))
    print('nsamples: %d tot_num_samples: %d' % (nsamples, len(train_data)))

    aveGrad = 0
    global_step = 0
    recent_losses = []
    start_t = time.time()

    best_f, cur_f = 0.0, 0.0
    for epoch in range(args.resume_epoch, args.nepochs):
        net.train()
        epoch_losses = []
        for ii, sample_batched in enumerate(trainloader):
            if 'trfe' in args.model_name or args.model_name == 'mtnet':
                nodules, glands = sample_batched
                inputs_n, labels_n = nodules['image'].cuda(), nodules['label'].cuda()
                inputs_g, labels_g = glands['image'].cuda(), glands['label'].cuda()
                inputs = torch.cat([inputs_n[0].unsqueeze(0), inputs_g[0].unsqueeze(0)], dim=0)

                for i in range(1, inputs_n.size()[0]):
                    inputs = torch.cat([inputs, inputs_n[i].unsqueeze(0)], dim=0)
                    inputs = torch.cat([inputs, inputs_g[i].unsqueeze(0)], dim=0)

                global_step += inputs.data.shape[0]
                nodule, thyroid = net.forward(inputs)
                loss = 0
                for i in range(inputs.size()[0]):
                    if i % 2 == 0:
                        loss += criterion(nodule[i], labels_n[int(i / 2)], size_average=False, batch_average=True)
                    else:
                        loss += 0.5 * criterion(thyroid[i], labels_g[int((i-1) / 2)], size_average=False, batch_average=True)

            else:
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                global_step += inputs.data.shape[0]

                outputs = net.forward(inputs)
                loss = criterion(outputs, labels, size_average=False, batch_average=True)

            trainloss = loss.item()
            epoch_losses.append(trainloss)
            if len(recent_losses) < args.log_every:
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss

            # Backward the averaged gradient
            loss.backward()
            aveGrad += 1
            nitrs += 1
            nsamples += args.batch_size

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % args.naver_grad == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            if nitrs % args.log_every == 0:
                meanloss = sum(recent_losses) / len(recent_losses)
                print('epoch: %d ii: %d trainloss: %.2f timecost:%.2f secs' % (
                    epoch, ii, meanloss, time.time() - start_t))
                writer.add_scalar('data/trainloss', meanloss, nsamples)

        meanloss = sum(epoch_losses) / len(epoch_losses)
        print('epoch: %d meanloss: %.2f' % (epoch, meanloss))
        writer.add_scalar('data/epochloss', meanloss, nsamples)

        if args.use_test == 1:
            prec_lists = []
            recall_lists = []
            sum_testloss = 0.0
            total_mae = 0.0
            cnt = 0
            count = 0
            iou = 0
            if args.use_eval == 1:
                net.eval()
            for ii, sample_batched in enumerate(testloader):
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                with torch.no_grad():
                    if 'trfe' in args.model_name or args.model_name == 'mtnet':
                        outputs, _ = net.forward(inputs)
                    else:
                        outputs = net.forward(inputs)

                loss = criterion(outputs, labels, size_average=False, batch_average=True)
                sum_testloss += loss.item()

                predictions = torch.sigmoid(outputs)

                iou += utils.get_iou(predictions, labels)
                count += 1

                total_mae += utils.get_mae(predictions, labels) * predictions.size(0)
                prec_list, recall_list = utils.get_prec_recall(predictions, labels)
                prec_lists.extend(prec_list)
                recall_lists.extend(recall_list)
                cnt += predictions.size(0)

                if ii % num_iter_ts == num_iter_ts - 1:
                    mmae = total_mae / cnt
                    mean_testloss = sum_testloss / num_iter_ts
                    mean_prec = sum(prec_lists) / len(prec_lists)
                    mean_recall = sum(recall_lists) / len(recall_lists)
                    fbeta = 1.3 * mean_prec * mean_recall / (0.3 * mean_prec + mean_recall)
                    iou = iou / count

                    print('Validation:')
                    print('epoch: %d, numImages: %d testloss: %.2f mmae: %.4f fbeta: %.4f iou: %.4f' % (
                        epoch, cnt, mean_testloss, mmae, fbeta, iou))
                    writer.add_scalar('data/validloss', mean_testloss, nsamples)
                    writer.add_scalar('data/validmae', mmae, nsamples)
                    writer.add_scalar('data/validfbeta', fbeta, nsamples)
                    writer.add_scalar('data/validiou', iou, epoch)

                    cur_f = iou
                    if cur_f > best_f:
                        save_path = os.path.join(save_dir, args.model_name + '_best' + '.pth')
                        torch.save(net.state_dict(), save_path)
                        print("Save model at {}\n".format(save_path))
                        best_f = cur_f

        if epoch % args.save_every == args.save_every - 1:
            save_path = os.path.join(save_dir, args.model_name + '_epoch-' + str(epoch) + '.pth')
            torch.save(net.state_dict(), save_path)
            print("Save model at {}\n".format(save_path))


if __name__ == "__main__":
    args = get_arguments()
    main(args)
