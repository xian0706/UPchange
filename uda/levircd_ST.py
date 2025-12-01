import glob
import os

import ever as er
import numpy as np
from albumentations import Compose, Normalize
from skimage.io import imread
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler, SubsetRandomSampler,RandomSampler
from core import field


class LEVIRCD(Dataset):
    def __init__(self, root_dir, transforms=None,target_label=None):
        self.root_dir = root_dir
        self.A_image_fps = glob.glob(os.path.join(root_dir, 'A', '*.png'))
        self.B_image_fps = [fp.replace('A', 'B') for fp in self.A_image_fps]
        phase = os.path.basename(root_dir)
        self.gt_fps = sorted(glob.glob(os.path.join(target_label,phase,"label","*.png")))
        self.transforms = transforms

    def __getitem__(self, idx):
        im1 = imread(self.A_image_fps[idx])
        im2 = imread(self.B_image_fps[idx])

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
        return len(self.A_image_fps)


@er.registry.DATALOADER.register()
class LEVIRCDLoader_ST(er.ERDataLoader):
    def __init__(self, config,pesudo_path):
        self.label_path = pesudo_path
        super(LEVIRCDLoader_ST, self).__init__(config)

    @property
    def dataloader_params(self):
        if any([isinstance(self.config.root_dir, tuple),
                isinstance(self.config.root_dir, list)]):
            dataset_list = []
            for im_dir in self.config.root_dir:
                dataset_list.append(LEVIRCD(im_dir,
                                            self.config.transforms,
                                            self.label_path))

            dataset = ConcatDataset(dataset_list)

        else:
            dataset = LEVIRCD(self.config.root_dir,
                              self.config.transforms,
                              self.label_path)

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
