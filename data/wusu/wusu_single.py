import os
import glob
from torch.utils.data import Dataset
import ever as er
from skimage.io import imread
import numpy as np
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, OneOf, Normalize
from core import field
from core.dataset import ColorAugDataset
from torch.utils.data import SequentialSampler, ConcatDataset, RandomSampler


def group_files(root, patterns):
    fps_list = [sorted(glob.glob(os.path.join(root, p))) for p in patterns]
    return fps_list


class WUSU(Dataset):
    def __init__(self, root, transforms=None):
        imgs = glob.glob(os.path.join(root, "HS/imgs/HS15_*.tif"))
        self.im_fps = imgs.copy()
        self.im_fps += [fp.replace("HS15", "HS16") for fp in imgs]
        self.im_fps += [fp.replace("HS15", "HS18") for fp in imgs]
        imgs = glob.glob(os.path.join(root, "JA/imgs/JA15_*.tif"))
        self.im_fps += imgs
        self.im_fps += [fp.replace("JA15", "JA16") for fp in imgs]
        self.im_fps += [fp.replace("JA15", "JA18") for fp in imgs]

        self.gt_fps = [fp.replace("imgs", "class") for fp in self.im_fps]
        assert len(self.im_fps) == len(self.gt_fps)
        self.transforms = transforms

    def __getitem__(self, idx):
        img = imread(self.im_fps[idx])[:,:,:3]

        # -0, 0-5
        label = imread(self.gt_fps[idx]).astype(np.int32) - 1

        if self.transforms:
            blob = self.transforms(**dict(image=img, masks=label))
            img = blob['image']

        y = dict()
        y[field.MASK1] = label
        y['image_filename'] = os.path.basename(self.im_fps[idx])

        return img, y

    def __len__(self):
        return len(self.im_fps)


@er.registry.DATALOADER.register()
class WUSULoader_single(er.ERDataLoader):
    def __init__(self, config):
        super(WUSULoader_single, self).__init__(config)

    @property
    def dataloader_params(self):
        if self.config.training:
            transform = None
        else:
            transform = self.config.common_transforms

        if any([isinstance(self.config.image_dir, tuple),
                isinstance(self.config.image_dir, list)]):
            dataset_list = []
            for im_dir in self.config.image_dir:
                dataset_list.append(WUSU(im_dir,transform))

            dataset = ConcatDataset(dataset_list)
        else:
            dataset = WUSU(self.config.image_dir,transform)

        if self.config.training:
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
                    pin_memory=True,
                    drop_last=False,
                    timeout=0,
                    worker_init_fn=None)

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
            cv=dict(k=10, i=0),
            training=True,
            batch_size=4,
            num_workers=4,
        ))
