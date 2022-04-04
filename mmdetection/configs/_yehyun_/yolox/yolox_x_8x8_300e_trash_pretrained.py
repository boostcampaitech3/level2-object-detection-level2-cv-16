_base_ = './yolox_s_8x8_300e_trash.py'

pretrained = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth'

# model settings
model = dict(
    backbone=dict(
        deepen_factor=1.33, widen_factor=1.25,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320))

# log_config = dict(interval=50)
log_config = dict(
            interval=150,
            hooks=[
                dict(type='MlflowLoggerHook', exp_name='yolox_x_trash_pretrained'),
                dict(type='TextLoggerHook')
            ])