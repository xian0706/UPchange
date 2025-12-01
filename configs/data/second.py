from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, OneOf, Normalize, RandomCrop
import ever as er
from data.aug import RandomBrightnessContrastv2
from ever.api.preprocess.albu import RandomDiscreteScale

data = dict(
    train=dict(
        type='SecondLoader',
        params=dict(
            root='C:\doctor\dataset\second/train',
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], 0.75),
                RandomDiscreteScale((1.25, 1.5, 1.75, 2.0), p=0.5),
                RandomCrop(256, 256),
                RandomBrightnessContrastv2(p=0.5),
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
                er.preprocess.albu.ToTensor()
            ]),
            cv=dict(k=5, i=0),
            training=True,
            batch_size=1,
            num_workers=4,
            distributed=False,
        ),
    ),
    test=dict(
        type='SecondLoader',
        params=dict(
            root='C:\doctor\dataset\second/train',
            transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
                er.preprocess.albu.ToTensor()
            ]),
            cv=dict(k=5, i=0),
            training=False,
            batch_size=1,
            num_workers=0,
            distributed=False,
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
        max_iters=15000,
    )
)
train = dict(
    forward_times=1,
    num_iters=15000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=False,
    apex_sync_bn=True,
    sync_bn=False,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=1000,
    eval_interval_epoch=80,
)
test = dict(
)
