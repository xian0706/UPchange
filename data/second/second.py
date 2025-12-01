import os
import glob
from torch.utils.data import Dataset
import ever as er
from skimage.io import imread
import numpy as np
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, OneOf, Normalize
from torch.utils.data import SequentialSampler, SubsetRandomSampler,ConcatDataset,RandomSampler

def group_files(root, patterns):
    fps_list = [sorted(glob.glob(os.path.join(root, p))) for p in patterns]
    return fps_list


class Second(Dataset):
    def __init__(self, root, transforms=None):
        self.im1_fps, self.im2_fps, self.gt1_fps, self.gt2_fps = group_files(
            root, ['im1/*.png', 'im2/*.png', 'label1/*.png', 'label2/*.png']
        )
        assert len(self.im1_fps) == len(self.im2_fps)
        assert len(self.im1_fps) == len(self.gt1_fps)
        assert len(self.im2_fps) == len(self.gt2_fps)
        self.transforms = transforms

    def __getitem__(self, idx):
        im1 = imread(self.im1_fps[idx])
        im2 = imread(self.im2_fps[idx])
        # -0, 0-5
        label1 = imread(self.gt1_fps[idx]).astype(np.int32) - 1
        label2 = imread(self.gt2_fps[idx]).astype(np.int32) - 1

        img12 = np.concatenate([im1, im2], axis=2)
        masks = [label1, label2, (label1 >= 0).astype(label1.dtype)]

        if self.transforms:
            blob = self.transforms(**dict(image=img12, masks=masks))
            img12 = blob['image']
            masks = blob['masks']

        masks.append(os.path.basename(self.im1_fps[idx]))

        return img12, masks

    def __len__(self):
        return len(self.im1_fps)


@er.registry.DATALOADER.register()
class SecondLoader(er.ERDataLoader):
    def __init__(self, config):
        super(SecondLoader, self).__init__(config)

    @property
    def dataloader_params(self):
        dataset = Second(self.config.root,
                         self.config.transforms)
        """CV = er.data.CrossValSamplerGenerator(dataset, distributed=self.config.distributed, seed=2333)
                sampler_pairs = CV.k_fold(self.config.cv.k)
                train_sampler, val_sampler = sampler_pairs[self.config.cv.i]
                if self.config.training:
                    sampler = train_sampler
                else:
                    sampler = SequentialSampler(dataset)"""

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
