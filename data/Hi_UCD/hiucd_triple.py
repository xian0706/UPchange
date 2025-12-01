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


class HiUCD(Dataset):
    def __init__(self, source_dir, target_dir, transforms=None):
        self.im1_source, self.im2_source, self.gt_source = group_files(
            source_dir, ['image/2017/9/*.png', 'image/2018/9/*.png', 'mask_merge/2017_2018/9/*.png']
        )
        self.im_source = self.im1_source + self.im2_source
        self.gt_source = self.gt_source + self.gt_source
        assert len(self.im_source) == len(self.gt_source)

        try:
            self.im1_target, self.im2_target, self.gt_target = group_files(
                target_dir, ['image/2017/9/*.png', 'image/2018/9/*.png', 'mask_merge/2017_2018/9/*.png']
            )
            assert len(self.im1_target) == len(self.im2_target)
            assert len(self.im1_target) == len(self.gt_target)
        except:
            self.im1_target, self.im2_target, self.gt_target = group_files(
                target_dir, ['image/2018/9/*.png', 'image/2019/9/*.png', 'mask_merge/2018_2019/9/*.png']
            )
            assert len(self.im1_target) == len(self.im2_target)
            assert len(self.im1_target) == len(self.gt_target)

        self.transforms = transforms

    def __getitem__(self, idx):
        # source
        idx_source = random.randint(0,len(self.im_source)-1)
        img = imread(self.im_source[idx_source])
        label = imread(self.gt_source[idx_source]).astype(np.int32) - 1
        if idx_source<=len(self.im1_target):
            label = label[:,:,0]
        else:
            label = label[:,:,1]
        y = dict()
        y[field.MASK1] = label

        # target
        im1 = imread(self.im1_target[idx])
        im2 = imread(self.im2_target[idx])
        # -0, 0-5
        label = imread(self.gt_target[idx]).astype(np.int32) - 1
        label1 = label[:, :, 0]
        label2 = label[:, :, 1]
        labelc = label[:, :, 2]

        img12 = np.concatenate([im1, im2], axis=2)
        masks = [label1, label2, labelc]
        if self.transforms:
            blob = self.transforms(**dict(image=img12,masks=masks))
            img12 = blob['image']

        return img12, img, y

    def __len__(self):
        return len(self.im1_target)


@er.registry.DATALOADER.register()
class HiUCDLoader_triple(er.ERDataLoader):
    def __init__(self, config):
        super(HiUCDLoader_triple, self).__init__(config)

    @property
    def dataloader_params(self):
        dataset = HiUCD(self.config.source_dir,
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
