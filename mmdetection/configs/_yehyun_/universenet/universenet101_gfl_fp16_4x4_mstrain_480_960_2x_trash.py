_base_ = [
    '../_base_/models/universenet101_gfl.py',
    '../_base_/datasets/trash_detection.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=512.)

log_config = dict(
            interval=300,
            hooks=[
                dict(type='MlflowLoggerHook', exp_name='universenet101_gfl_fp16_4x4_mstrain_480_960_2x_trash'),
                dict(type='TextLoggerHook')
            ])