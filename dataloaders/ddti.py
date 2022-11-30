import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class DDTI(data.Dataset):

    def __init__(self, root='./data/DDTI', transform=None, return_size=False):
        img_names = os.listdir(root+'/image')
        img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))
        self.root = root
        self.img_names = img_names
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, index: int):
        img_name = self.img_names[index]
        image_path = os.path.join(self.root + '/image', img_name)
        label_path = os.path.join(self.root + '/mask', img_name)
        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

        image = Image.open(image_path).convert('RGB')
        label = np.array(Image.open(label_path).convert('L'))
        label = label / label.max()
        label = Image.fromarray(label.astype(np.uint8))

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            w, h = image.size
            size = (h, w)
            sample['size'] = torch.tensor(size)

        label_name = os.path.basename(label_path)
        sample['label_name'] = label_name

        return sample

    def __len__(self) -> int:
        return len(self.img_names)
