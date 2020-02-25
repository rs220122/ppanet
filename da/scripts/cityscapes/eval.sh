# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-30T05:30:32.834Z
# Description:
#
# ===============================================

TRAIN_LOGDIR='da/logs/cityscapes/2020-01-17_extra'

python eval.py \
       --output_stride=16 \
       --crop_size=1025,2049 \
       --batch_size=1 \
       --model_variant=resnet_v1_101_beta \
       --backbone_atrous_rates=1 \
       --backbone_atrous_rates=2 \
       --backbone_atrous_rates=4 \
       --decoder_output_stride=4 \
       --module_order=da \
       --eval_logdir=${TRAIN_LOGDIR} \
       --checkpoint_dir=${TRAIN_LOGDIR} \
       --dataset_name=cityscapes \
       --dataset_dir=dataset/cityscapes/tfrecord \
       --split_name=val_fine \
       --add_flipped_images \
       --eval_scales=0.75 \
       --eval_scales=1.0 \
       --eval_scales=1.25
