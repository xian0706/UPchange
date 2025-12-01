import os
import glob
from torch.utils.data import Dataset
import ever as er
from skimage.io import imread
import numpy as np
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, OneOf, Normalize
from torch.utils.data import SequentialSampler, RandomSampler
from core import field


def group_files(root, patterns):
    fps_list = [sorted(glob.glob(os.path.join(root, p))) for p in patterns]
    return fps_list


class HiUCD(Dataset):
    def __init__(self, root, transforms=None,target_label=None):
        self.gt_fps =  sorted(glob.glob(os.path.join(target_label,"*.png")))
        try:
            self.im1_fps, self.im2_fps = group_files(
              root, ['image/2017/9/*.png', 'image/2018/9/*.png'])
            assert len(self.im1_fps) == len(self.gt_fps)
        except:
            self.im1_fps, self.im2_fps = group_files(
              root, ['image/2018/9/*.png', 'image/2019/9/*.png'])
            assert len(self.im1_fps) == len(self.gt_fps)
        self.transforms = transforms

    def __getitem__(self, idx):
        im1 = imread(self.im1_fps[idx])
        im2 = imread(self.im2_fps[idx])

        label = imread(self.gt_fps[idx]).astype(np.int32) - 1
        label1 = label[:, :, 0]
        label2 = label[:, :, 1]
        labelc = label[:, :, 2]

        img12 = np.concatenate([im1, im2], axis=2)
        masks = [label1, label2, labelc]
        if self.transforms:
            blob = self.transforms(**dict(image=img12, masks=masks))
            img12 = blob['image']
            masks = blob['masks']

        y = dict()
        y[field.MASK1] = masks[0]
        y[field.VMASK2] = masks[1]

        return img12, y

    def __len__(self):
        return len(self.im1_fps)


@er.registry.DATALOADER.register()
class HiUCDLoader_ST(er.ERDataLoader):
    def __init__(self, config,pesudo_path):
        self.label_path = pesudo_path
        super(HiUCDLoader_ST, self).__init__(config)


    @property
    def dataloader_params(self):
        dataset = HiUCD(self.config.image_dir,
                        self.config.transforms,
                        self.label_path)

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
