import torch.utils.data as data
import PIL.Image as Image
import os
import json
import numpy as np
import torch


def make_dataset(root, seed):
    imgs = []
    img_labels = {}

    img_names = os.listdir(root + 'thyroid-image/')
    img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))

    for i in seed:
        img_name = img_names[i]
        img = os.path.join(root + 'thyroid-image/', img_name)
        mask = os.path.join(root + 'thyroid-mask/', img_name)
        imgs.append((img, mask, 0))
    return imgs


class TG3K(data.Dataset):
    def __init__(self, mode, transform=None, return_size=False):
        self.mode = mode
        root = './data/tg3k/'
        trainval = json.load(open(root + 'tg3k-trainval.json', 'r')) 
        if mode == 'train':
            imgs = make_dataset(root, trainval['train'])
        elif mode == 'val':
            imgs = make_dataset(root, trainval['val'])

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, item):
        if True:
            image_path, label_path, label = self.imgs[item]
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

    def __len__(self):
        return len(self.imgs)
