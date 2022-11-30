import torch.utils.data as data
import PIL.Image as Image
import os
import json
import numpy as np
import torch


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
    img_names = os.listdir(root +'/test-image/')
    img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))

    for img_name in img_names:
        img = os.path.join(root +'/test-image/', img_name)
        mask = os.path.join(root +'/test-mask/', img_name)
        imgs.append((img, mask))
    return imgs


class TN3K(data.Dataset):
    def __init__(self, mode, transform=None, return_size=False, fold=0):
        self.mode = mode
        # FIXME:for test, 记得改回原来的 ./data/tn3k
        root = './data/tn3k/'
        trainval = json.load(open(root + 'tn3k-trainval-fold'+str(fold)+'.json', 'r'))
        if mode == 'train':
            imgs = make_dataset(root, trainval['train'], 'trainval')
        elif mode == 'val':
            imgs = make_dataset(root, trainval['val'], 'trainval')
        elif mode == 'test':
            imgs = make_testset(root)

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, item):
        if True:
            image_path, label_path = self.imgs[item]
            assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
            assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

            image = Image.open(image_path).convert('RGB')
            label = np.array(Image.open(label_path).convert('L'))
            label = label / label.max()

            label = Image.fromarray(label.astype(np.uint8))
            w, h = image.size
            size = (h, w)

            sample = {'image': image, 'label': label}

            if self.transform:
                sample = self.transform(sample)
            if self.return_size:
                sample['size'] = torch.tensor(size)

            label_name = os.path.basename(label_path)
            sample['label_name'] = label_name

            return sample
        # else:
        #     image_path = self.imgs[item]
        #     image = Image.open(image_path).convert('RGB')
        #     label = Image.open(image_path)
        #     sample = {'image': image, 'label': label}
        #     w, h = image.size
        #     size = (h, w)
        #
        #     if self.transform:
        #         sample = self.transform(sample)
        #     if self.return_size:
        #         sample['size'] = torch.tensor(size)
        #
        #     label_name = os.path.basename(image_path)
        #     sample['label_name'] = label_name
        #
        #     return sample

    def __len__(self):
        return len(self.imgs)
