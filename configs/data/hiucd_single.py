from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, OneOf, Normalize, RandomCrop,RandomBrightnessContrast
import ever as er
from data.aug import RandomBrightnessContrastv2
try:
    from ever.api.preprocess.albu import RandomDiscreteScale
except:
    from ever.preprocess.albu import RandomDiscreteScale

data = dict(
    train=dict(
        type='HiUCDLoader_single',
        params=dict(
            image_dir='/data1/yjx23/dataset/Hi-UCD_mini/train',
            geo_transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], p=0.75),
                er.preprocess.albu.RandomDiscreteScale([1.25, 1.5, 1.75, 2.0], p=0.5),
                RandomCrop(256, 256, True),
            ]),
            color_transforms=RandomBrightnessContrast(p=0.75),
            common_transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225), max_pixel_value=255),
                er.preprocess.albu.ToTensor(),
            ]),
            training=True,
            batch_size=16,
            num_workers=4,
        ),
    ),
    test=dict(
        type='HiUCDLoader',
        params=dict(
            root='/data1/yjx23/dataset/Hi-UCD_mini/test',
            transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
                er.preprocess.albu.ToTensor()
            ]),
            training=False,
            batch_size=1,
            num_workers=0,
        ),
    ),
    pseudo=dict(
        type='HiUCDLoader',
        params=dict(
            root='/data1/yjx23/dataset/Hi-UCD_mini/test',
            transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
                er.preprocess.albu.ToTensor()
            ]),
            training=False,
            batch_size=8,
            num_workers=0,
        ),
    ),
    target=dict(
        type='HiUCDLoader',
        params=dict(
            root='/data1/yjx23/dataset/Hi-UCD_mini/test',
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], 0.75),
                RandomDiscreteScale((1.25, 1.5, 1.75, 2.0), p=0.5),
                RandomCrop(256, 256, True),
                RandomBrightnessContrastv2(p=0.5),
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
                er.preprocess.albu.ToTensor()
            ]),
            training=True,
            batch_size=16,
            num_workers=4,
        ),
    ),
)
optimizer = dict(
    type='sgd',
    params=dict(
        momentum=0.9,
        weight_decay=0.0001
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)
learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.03,
        power=0.9,
        max_iters=40000,
    )
)
train = dict(
    forward_times=1,
    num_iters=40000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=False,
    apex_sync_bn=True,
    sync_bn=False,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=5000,
    eval_interval_epoch=80,
)
test = dict(
)