#fp16 = dict(loss_scale=512.)
_base_ = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[0.2, 0.5, 1.0, 2.0, 5.0])),
    roi_head=dict(
        bbox_head=dict(
            num_classes=6,
            reg_decoded_bbox=True,
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))))

dataset_type = 'CocoDataset'
data_root = '/swdata/df/cz_data/data/tile_round1_train_20201231/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(2048, 1500), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1500),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'cam3_train.json',
        img_prefix=data_root + 'CAM3/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'cam3_test.json',
        img_prefix=data_root + 'CAM3/test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'cam3_test.json',
        img_prefix=data_root + 'CAM3/test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
work_dir = './df_use/cam3'
