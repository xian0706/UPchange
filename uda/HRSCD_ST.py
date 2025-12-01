import os
import glob
from torch.utils.data import Dataset
import ever as er
from skimage.io import imread
import numpy as np
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, OneOf, Normalize
from torch.utils.data import SequentialSampler,RandomSampler
import tifffile as tf
import cv2
from core import field


def group_files(root, patterns):
    fps_list = [sorted(glob.glob(os.path.join(root, p))) for p in patterns]
    return fps_list


class HRSCD(Dataset):
    def __init__(self, root, transforms=None,target_label=None):

        self.im1_fps = glob.glob(os.path.join(root, "image/2006/D14/*.png"))
        self.im2_fps = glob.glob(os.path.join(root, "image/2012/D14/*.png"))
        self.gt1_cls = []
        self.gt2_cls = []
        for img in self.im2_fps:
            im = os.path.basename(img)
            self.gt1_cls.append(os.path.join(target_label, 'label1', '%s' % im.replace("tif","png")))
            self.gt1_cls.append(os.path.join(target_label, 'label2', '%s' % im.replace("tif","png")))

        assert len(self.im1_fps) == len(self.im2_fps)
        self.transforms = transforms

    def __getitem__(self, idx):
        im1 = imread(self.im1_fps[idx])[:,:,:3]
        im2 = imread(self.im2_fps[idx])[:,:,:3]

        label1 = imread(self.gt1_cls[idx]).astype(np.int32) - 1
        label2 = imread(self.gt2_cls[idx]).astype(np.int32) - 1
        labelc = np.where(label1 == label2, 0, 1)
        labelc = np.where(label1 == -1, -1, labelc)
        labelc = np.where(label2 == -1, -1, labelc)

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
class HRSCDLoader_ST(er.ERDataLoader):
    def __init__(self, config,pesudo_path):
        self.label_path = pesudo_path
        super(HRSCDLoader_ST, self).__init__(config)

    @property
    def dataloader_params(self):
        dataset = HRSCD(self.config.root,
                         self.config.transforms,
                        self.label_path)

        if self.config.training:
            try:
                sampler = er.data.StepDistributedSampler(dataset)
                print("step")
            except:
                sampler = RandomSampler(dataset)
                print("random")
        else:
            sampler = SequentialSampler(dataset)
            print("seq")

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
