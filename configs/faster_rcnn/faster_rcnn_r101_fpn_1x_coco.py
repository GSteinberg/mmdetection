_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2)),
    pretrained='torchvision://resnet101',
    backbone=dict(depth=101))

dataset_type = 'COCODataset'
classes = ('pfm-1', 'ksf-casing')
data = dict(
    train=dict(
        img_prefix='landmine/train/',
        classes=classes,
        ann_file='landmine/train/coco_annotation.json'),
    val=dict(
        img_prefix='landmine/train/',
        classes=classes,
        ann_file='landmine/train/coco_annotation.json'),
    test=dict(
        img_prefix='landmine/train/',
        classes=classes,
        ann_file='landmine/train/coco_annotation.json'))

# load from pre-trained model
load_from = 'checkpoints/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'