# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-28T07:58:56.053Z
# Description:
#
# ===============================================

TRAIN_LOGDIR='ppa/logs/cityscapes/2020-01-19_extra'

python vis.py \
       --output_stride=16 \
       --batch_size=1 \
       --crop_size=1025,2049 \
       --model_variant=resnet_v1_101_beta \
       --backbone_atrous_rates=1 \
       --backbone_atrous_rates=2 \
       --backbone_atrous_rates=4 \
       --decoder_output_stride=4 \
       --ppm_rates=1 \
       --ppm_rates=2 \
       --ppm_rates=3 \
       --ppm_rates=6 \
       --ppm_pooling_type=avg \
       --atrous_rates=8 \
       --atrous_rates=16 \
       --atrous_rates=24 \
       --module_order=ppa \
       --vis_logdir=${TRAIN_LOGDIR} \
       --checkpoint_dir=${TRAIN_LOGDIR} \
       --split_name=val_fine \
       --dataset_name=cityscapes \
       --dataset_dir=dataset/cityscapes/tfrecord \
       --add_flipped_images \
       --eval_scales=0.75 \
       --eval_scales=1.0 \
       --eval_scales=1.25

       # --save_labels \
