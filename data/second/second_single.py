import glob
import os
import copy
import ever as er
import numpy as np
from albumentations import Compose, Normalize
from skimage.io import imread
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler, SubsetRandomSampler,RandomSampler
from core import field
from core.dataset import ColorAugDataset


class Second_single(Dataset):
    def __init__(self, img_dir, transforms=None):

        self.image_fps = glob.glob(os.path.join(img_dir, '*.png'))
        self.gt_fps = [fp.replace('im', 'label') for fp in self.image_fps]
        self.transforms = transforms

    def __getitem__(self, idx):
        img = imread(self.image_fps[idx])
        gt = imread(self.gt_fps[idx]).astype(np.int32) - 1

        if self.transforms:
            for transform in self.transforms:
                blob = transform(**dict(image=img, mask=gt))
                img = blob['image']
                gt = blob['mask']

        y = dict()
        y[field.MASK1] = gt
        y['image_filename'] = os.path.basename(self.image_fps[idx])
        return img, y

    def __len__(self):
        return len(self.image_fps)

@er.registry.DATALOADER.register()
class SecondLoader_single(er.ERDataLoader):
    def __init__(self, config):
        super(SecondLoader_single, self).__init__(config)

    @property
    def dataloader_params(self):
        if self.config.training:
            #transform = [self.config.color_transforms]
            transform = None
        else:
            transform = self.config.common_transforms

        if any([isinstance(self.config.image_dir, tuple),
                isinstance(self.config.image_dir, list)]):
            dataset_list = []
            for im_dir in self.config.image_dir:
                dataset_list.append(Second_single(im_dir,
                                                transform))

            dataset = ConcatDataset(dataset_list)

        else:
            dataset = Second_single(self.config.image_dir,
                                  transform)

        if self.config.training:
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

        return dict(dataset=dataset,
                    batch_size=self.config.batch_size,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    timeout=0,
                    worker_init_fn=None)

    def set_default_config(self):
        self.config.update(dict(
            root_dir='',
            transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225), max_pixel_value=255),
                er.preprocess.albu.ToTensor(),
            ]),
            subsample_ratio=1.0,
            batch_size=1,
            num_workers=0,
            training=False
        ))