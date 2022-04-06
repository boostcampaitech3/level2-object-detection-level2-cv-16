_base_ = [
    '../universenet/models/universenet101_2008d.py',
    '../_base_/datasets/trash_detection_aug.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu=16)

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=512.)

log_config = dict(
            interval=50,
            hooks=[
                dict(type='MlflowLoggerHook', exp_name='universenet101_2008d_fp16_4x4_mstrain_480_960_20e_trash'),
                dict(type='TextLoggerHook')
            ])