import torch.utils.data as data
import PIL.Image as Image
import os
import json
import numpy as np
import torch
import random


def make_dataset(root, seed, name):
    imgs = []
    img_labels = {}
    img_names = os.listdir(root +'/'+ name+'-image/')
    img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))
    for i in seed:
        img_name = img_names[i]
        img = os.path.join(root +'/'+ name+ '-image/', img_name)
        mask = os.path.join(root +'/'+ name+ '-mask/', img_name)
        imgs.append((img, mask))
    return imgs


def make_testset(root):
    imgs = []
    img_labels = {}
    img_names = os.listdir(root +'tn3k/test-image/')
    img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))

    for img_name in imgnames:
        img = os.path.join(root +'tn3k/'+ name+ '-image/', img_name)
        mask = os.path.join(root +'tn3k/'+ name+ '-mask/', img_name)
        imgs.append((img, mask))
    return imgs


class TATN(data.Dataset):
    def __init__(self, mode, transform=None, return_size=False, fold=0):
        self.mode = mode
        root = './data/'
        nodule_trainval = json.load(open(root + 'tn3k/tn3k-trainval-fold' + str(fold) + '.json', 'r'))  # seeds for k-fold cross validation
        thyroid_trainval = json.load(open(root + 'tg3k/tg3k-trainval.json', 'r'))  # seeds for k-fold cross validation

        nodule_imgs = []
        gland_imgs = []
        marks = []

        if mode == 'train':
            nodule_pathes = make_dataset(root, nodule_trainval['train'], 'tn3k/trainval')
            gland_pathes = make_dataset(root, thyroid_trainval['train'], 'tg3k/thyroid')
            length = min(len(nodule_pathes), len(gland_pathes))
            for i in range(length):
                nodule_imgs.append(nodule_pathes[i])
                gland_imgs.append(gland_pathes[i])

        elif mode == 'val':
            nodule_imgs = make_dataset(root, nodule_trainval['val'], 'tn3k/trainval')
        
        elif mode == 'test':
            nodule_imgs = make_dataset(root)
        
        self.marks = marks
        self.nodule_imgs = nodule_imgs
        self.gland_imgs = gland_imgs
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, item):
        if self.mode == 'train':
            image_path, label_path = self.nodule_imgs[item]
            label_name = os.path.basename(label_path)

            assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
            assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

            image = Image.open(image_path).convert('RGB')
            label = np.array(Image.open(label_path).convert('L'))
            label = label / label.max()
            label = Image.fromarray(label.astype(np.uint8))

            w, h = image.size
            size = (h, w)

            nodule = {'image': image, 'label': label}
            nodule = self.transform(nodule)

            if self.return_size:
                nodule['size'] = torch.tensor(size)

            label_name = os.path.basename(label_path)
            nodule['label_name'] = label_name
            nodule['mark'] = 1

            # Gland
            image_path, label_path = self.gland_imgs[item]
            assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
            assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

            image = Image.open(image_path).convert('RGB')
            label = np.array(Image.open(label_path).convert('L'))
            label = label / label.max()
            label = Image.fromarray(label.astype(np.uint8))

            w, h = image.size
            size = (h, w)

            gland = {'image': image, 'label': label}
            gland = self.transform(gland)

            if self.return_size:
                gland['size'] = torch.tensor(size)

            label_name = os.path.basename(label_path)
            gland['label_name'] = label_name
            gland['mark'] = 0

            return nodule, gland

        else:
            image_path, label_path = self.nodule_imgs[item]
            assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
            assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

            image = Image.open(image_path).convert('RGB')
            label = np.array(Image.open(label_path).convert('L'))
            label = label / label.max()
            label = Image.fromarray(label.astype(np.uint8))

            w, h = image.size
            size = (h, w)

            nodule = {'image': image, 'label': label}
            nodule = self.transform(nodule)

            if self.return_size:
                nodule['size'] = torch.tensor(size)

            label_name = os.path.basename(label_path)
            nodule['label_name'] = label_name
            nodule['mark'] = 1

            return nodule

    def __len__(self):
        return len(self.nodule_imgs)
