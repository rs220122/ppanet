# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-28T07:58:56.053Z
# Description:
#
# ===============================================

TRAIN_LOGDIR='aspp/logs/cityscapes/2020-01-09_extra'

python vis.py \
       --output_stride=8 \
       --batch_size=1 \
       --crop_size=1025,2049 \
       --model_variant=resnet_v1_101_beta \
       --backbone_atrous_rates=1 \
       --backbone_atrous_rates=2 \
       --backbone_atrous_rates=4 \
       --decoder_output_stride=4 \
       --atrous_rates=12 \
       --atrous_rates=24 \
       --atrous_rates=36 \
       --module_order=aspp \
       --vis_logdir=${TRAIN_LOGDIR} \
       --checkpoint_dir=${TRAIN_LOGDIR} \
       --split_name=val_fine \
       --dataset_name=cityscapes \
       --dataset_dir=dataset/cityscapes/tfrecord \
       --save_labels \
       --add_flipped_images \
       --eval_scales=0.75 \
       --eval_scales=1.0 \
       --eval_scales=1.25
