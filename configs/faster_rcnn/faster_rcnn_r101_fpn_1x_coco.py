_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2)),
    pretrained='torchvision://resnet101',
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

runner = dict(type='EpochBasedRunner', max_epochs=50)
dataset_type = 'CocoDataset'
classes = ('pfm-1', 'ksf-casing')
data = dict(
    train=dict(
        type=dataset_type,
        classes=classes,
        img_prefix='landmine/train/',
        ann_file='landmine/train/coco_annotation.json'),
    val=dict(
        type=dataset_type,
        classes=classes,
        img_prefix='landmine/val/',
        ann_file='landmine/val/coco_annotation.json'),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_prefix='landmine/test/',
        ann_file='landmine/test/coco_annotation.json'))

# load from pre-trained model
load_from = 'checkpoints/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
