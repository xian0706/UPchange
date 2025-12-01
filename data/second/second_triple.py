import os
import glob
from torch.utils.data import Dataset
import ever as er
from skimage.io import imread
import numpy as np
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, OneOf, Normalize
from torch.utils.data import ConcatDataset
from core import field
from torch.utils.data import SequentialSampler,RandomSampler
from core.dataset import ColorAugDataset
import random


def group_files(root, patterns):
    fps_list = [sorted(glob.glob(os.path.join(root, p))) for p in patterns]
    return fps_list


class Second(Dataset):
    def __init__(self, source_dir, target_dir, transforms=None):
        if any([isinstance(source_dir, tuple),
                isinstance(source_dir, list)]):
            im_source_list = []
            gt_source_list = []
            for im_dir in source_dir:
                im_source = glob.glob(os.path.join(im_dir, '*.png'))
                gt_source = [fp.replace('im', 'label') for fp in im_source]
                im_source_list.append(im_source)
                gt_source_list.append(gt_source)
            self.im_source = ConcatDataset(im_source_list)
            self.gt_source = ConcatDataset(gt_source_list)
        else:
            self.im_source = glob.glob(os.path.join(source_dir, '*.png'))
            self.gt_source = [fp.replace('im', 'label') for fp in self.im_source]

        self.im1_target, self.im2_target, self.gt1_target, self.gt2_target = group_files(
            target_dir, ['im1/*.png', 'im2/*.png', 'label1/*.png', 'label2/*.png']
        )
        assert len(self.im1_target) == len(self.im2_target)
        assert len(self.im1_target) == len(self.gt1_target)
        assert len(self.im2_target) == len(self.gt2_target)
        assert len(self.im_source) == len(self.gt_source)
        self.transforms = transforms

    def __getitem__(self, idx):
        # source
        idx_source = random.randint(0, len(self.im_source) - 1)
        im = imread(self.im_source[idx_source])
        gt = imread(self.gt_source[idx_source]).astype(np.int32) - 1
        y = dict()
        y[field.MASK1] = gt

        # target
        im1 = imread(self.im1_target[idx])
        im2 = imread(self.im2_target[idx])
        # -0, 0-5
        label1 = imread(self.gt1_target[idx]).astype(np.int32) - 1
        label2 = imread(self.gt2_target[idx]).astype(np.int32) - 1

        img12 = np.concatenate([im1, im2], axis=2)
        masks = [label1, label2, (label1 >= 0).astype(label1.dtype)]
        if self.transforms:
            blob = self.transforms(**dict(image=img12,masks=masks))
            img12 = blob['image']

        return img12, im, y

    def __len__(self):
        return len(self.im1_target)


@er.registry.DATALOADER.register()
class SecondLoader_triple(er.ERDataLoader):
    def __init__(self, config):
        super(SecondLoader_triple, self).__init__(config)

    @property
    def dataloader_params(self):
        dataset = Second(self.config.source_dir,
                         self.config.target_dir,
                         self.config.transforms_target)

        dataset = ColorAugDataset(dataset,
                                  geo_transform=self.config.geo_transforms,
                                  color_transform=self.config.color_transforms,
                                  common_transform=self.config.common_transforms)

        if self.config.training:
            try:
                sampler = er.data.StepDistributedSampler(dataset)
                print(1)
            except:
                sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        # sampler = SequentialSampler(dataset)
        # sampler = er.data.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(dataset)

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
