_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/trash_detection_aug.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

log_config = dict(
            interval=150,
            hooks=[
                dict(type='MlflowLoggerHook', exp_name='faster_rcnn_r50_fpn_1x_trash_aug_tta'),
                # dict(type='TextLoggerHook')
            ])