_base_ = './retinanet_r50_caffe_fpn_mstrain_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 23])
runner = dict(max_epochs=24)
