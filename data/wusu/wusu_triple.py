import os
import glob
import random

from torch.utils.data import Dataset
import ever as er
from skimage.io import imread
import numpy as np
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, OneOf, Normalize
from torch.utils.data import ConcatDataset
from core import field
from torch.utils.data import SequentialSampler, RandomSampler
from core.dataset import ColorAugDataset


def group_files(root, patterns):
    fps_list = [sorted(glob.glob(os.path.join(root, p))) for p in patterns]
    return fps_list


class WUSU(Dataset):
    def __init__(self, source_dir, target_dir, transforms=None):
        imgs = glob.glob(os.path.join(source_dir, "HS/imgs/HS15_*.tif"))
        self.im_source = imgs.copy()
        self.im_source += [fp.replace("HS15", "HS16") for fp in imgs]
        self.im_source += [fp.replace("HS15", "HS18") for fp in imgs]
        imgs = glob.glob(os.path.join(source_dir, "JA/imgs/JA15_*.tif"))
        self.im_source += imgs
        self.im_source += [fp.replace("JA15", "JA16") for fp in imgs]
        self.im_source += [fp.replace("JA15", "JA18") for fp in imgs]

        self.gt_source = [fp.replace("imgs", "class") for fp in self.im_source]
        assert len(self.im_source) == len(self.gt_source)

        imgs = glob.glob(os.path.join(target_dir, "HS/imgs/HS15_*.tif"))
        self.im1_target = imgs.copy()
        self.im2_target = [fp.replace("HS15", "HS16") for fp in imgs]
        self.im1_target += [fp.replace("HS15", "HS16") for fp in imgs]
        self.im2_target += [fp.replace("HS15", "HS18") for fp in imgs]
        imgs = glob.glob(os.path.join(target_dir, "JA/imgs/JA15_*.tif"))
        self.im1_target += imgs
        self.im2_target += [fp.replace("JA15", "JA16") for fp in imgs]
        self.im1_target += [fp.replace("JA15", "JA16") for fp in imgs]
        self.im2_target += [fp.replace("JA15", "JA18") for fp in imgs]

        self.gt1_target = [fp.replace("imgs", "class") for fp in self.im1_target]
        self.gt2_target = [fp.replace("imgs", "class") for fp in self.im2_target]

        self.transforms = transforms

    def __getitem__(self, idx):
        # source
        idx_source = random.randint(0,len(self.im_source)-1)
        img = imread(self.im_source[idx_source])[:,:,:3]
        label = imread(self.gt_source[idx_source]).astype(np.int32) - 1
        y = dict()
        y[field.MASK1] = label

        # target
        im1 = imread(self.im1_target[idx])[:,:,:3]
        im2 = imread(self.im2_target[idx])[:,:,:3]
        # -0, 0-5
        label1 = imread(self.gt1_target[idx]).astype(np.int32) - 1
        label2 = imread(self.gt2_target[idx]).astype(np.int32) - 1
        labelc = np.where(label1 == label2, 0, 1)
        labelc = np.where(label1 == -1, -1, labelc)
        labelc = np.where(label2 == -1, -1, labelc)

        img12 = np.concatenate([im1, im2], axis=2)
        masks = [label1, label2, labelc]
        if self.transforms:
            blob = self.transforms(**dict(image=img12,masks=masks))
            img12 = blob['image']

        return img12, img, y

    def __len__(self):
        return len(self.im1_target)


@er.registry.DATALOADER.register()
class WUSULoader_triple(er.ERDataLoader):
    def __init__(self, config):
        super(WUSULoader_triple, self).__init__(config)

    @property
    def dataloader_params(self):
        dataset = WUSU(self.config.source_dir,
                         self.config.target_dir,
                         self.config.transforms_target)

        dataset = ColorAugDataset(dataset,
                                  geo_transform=self.config.geo_transforms,
                                  color_transform=self.config.color_transforms,
                                  common_transform=self.config.common_transforms)

        # sampler = SequentialSampler(dataset)
        if self.config.training:
            try:
                sampler = er.data.StepDistributedSampler(dataset)
                print(1)
            except:
                sampler = RandomSampler(dataset)
        else:
            SequentialSampler(dataset)

        return dict(dataset=dataset,
                    batch_size=self.config.batch_size,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=True)

    def set_default_config(self):
        self.config.update(dict(
            root='',
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], 0.75),
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
                er.preprocess.albu.ToTensor()
            ]),
            cv=dict(k=5, i=0),
            training=True,
            batch_size=4,
            num_workers=4,
        ))
