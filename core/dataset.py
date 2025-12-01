import torch

from core import field
from torch.utils.data import Dataset
import copy


class ColorAugDataset(Dataset):
    def __init__(self, dataset, geo_transform, color_transform, common_transform):
        self.dataset = dataset
        self.color_transform = color_transform
        self.geo_transform = geo_transform
        self.common_transform = common_transform

    def __getitem__(self, idx):
        try:
            img12, x, y = self.dataset[idx]
        except:
            x, y = self.dataset[idx]

        blob = self.geo_transform(**dict(image=x, mask=y[field.MASK1]))
        img = blob['image']
        mask = blob['mask']

        # x, mask -> tensor
        blob = self.common_transform(image=img, mask=mask)
        org_img = blob['image']
        mask = blob['mask']
        y[field.MASK1] = mask

        # x -> color_trans_x -> tensor
        if self.color_transform:
            color_trans_x = self.color_transform(**dict(image=img))['image']
            blob = self.common_transform(image=color_trans_x)
            color_trans_x = blob['image']
            y[field.COLOR_TRANS_X] = color_trans_x

        try:
            return img12,org_img,y
        except:
            return org_img, y

    def __len__(self):
        return len(self.dataset)
