import os
import glob
from torch.utils.data import Dataset
import ever as er
from skimage.io import imread
import numpy as np
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, OneOf, Normalize
from torch.utils.data import SequentialSampler
import tifffile as tf
import cv2


def group_files(root, patterns):
    fps_list = [sorted(glob.glob(os.path.join(root, p))) for p in patterns]
    return fps_list


class WUSU(Dataset):
    def __init__(self, root, transforms=None):
        imgs = glob.glob(os.path.join(root, "HS/imgs/HS15_*.tif"))
        self.im1_fps = imgs.copy()
        self.im2_fps = [fp.replace("HS15","HS16") for fp in imgs]
        self.im1_fps += [fp.replace("HS15","HS16") for fp in imgs]
        self.im2_fps += [fp.replace("HS15","HS18") for fp in imgs]
        imgs = glob.glob(os.path.join(root, "JA/imgs/JA15_*.tif"))
        self.im1_fps += imgs
        self.im2_fps += [fp.replace("JA15","JA16") for fp in imgs]
        self.im1_fps += [fp.replace("JA15","JA16") for fp in imgs]
        self.im2_fps += [fp.replace("JA15","JA18") for fp in imgs]

        self.gt1_cls = [fp.replace("imgs","class") for fp in self.im1_fps]
        self.gt2_cls = [fp.replace("imgs","class") for fp in self.im2_fps]
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

        masks.append(os.path.basename(self.im1_fps[idx]))
        masks.append(os.path.basename(self.im2_fps[idx]))

        return img12, masks

    def __len__(self):
        return len(self.im1_fps)


@er.registry.DATALOADER.register()
class WUSULoader(er.ERDataLoader):
    def __init__(self, config):
        super(WUSULoader, self).__init__(config)

    @property
    def dataloader_params(self):
        dataset = WUSU(self.config.root,
                         self.config.transforms)

        sampler = SequentialSampler(dataset)
        #sampler = er.data.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(dataset)

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
