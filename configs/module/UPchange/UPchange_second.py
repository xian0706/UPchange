from configs.data import second_single as std_config

config = dict(
    model=dict(
        type='Changeuda',
        params=dict(
            encoder=dict(name='efficientnet-b0',weights='imagenet'),
            temporal_transformer=dict(
                type='TemporalCat',
            ),
            semantic_decoder=dict(
                name='Unet',
                decoder=dict(
                    decoder_channels=(256, 128, 64, 32, 16),
                ),
                classifier=dict(
                    in_channels=16,
                    out_channels=6,
                    scale_factor=1,
                    kernel_size=1,
                )
            ),
            change_decoder=dict(
                name='Unet',
                decoder=dict(
                    decoder_channels=(256, 128, 64, 32, 16),
                ),
                classifier=dict(
                    in_channels=16,
                    out_channels=36,
                    scale_factor=1,
                    kernel_size=1,
                ),
            ),
            loss=dict(
                ignore_index=-1,
                ce=dict(),
                #tver=dict(alpha=0.5, beta=0.5),
                #con=dict(beta=2.0,weight=0),
                #uda=dict(minent=dict(weight=0.5))
            ),
            use_constract = True,
            use_trans = False,
            use_semantic = True,
            use_feature = False  
        ),
    ),
    data=std_config.data,
    optimizer=std_config.optimizer,
    learning_rate=std_config.learning_rate,
    train=std_config.train,
    test=std_config.test,
    warmup_step=10000,
    preheat_step=10000,
    eval_every=5000,
    generate_psedo_every=10000,
    #eval_per_epoch=True
)
