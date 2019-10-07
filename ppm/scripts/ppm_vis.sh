# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-28T07:58:56.053Z
# Description:
#
# ===============================================

TRAIN_LOGDIR='logs/ppm/log-2019-09-03'

python vis.py \
       --output_stride=8 \
       --batch_size=1 \
       --ppm_rates=1 \
       --ppm_rates=2 \
       --ppm_rates=3 \
       --ppm_rates=6 \
       --ppm_pooling_type=avg \
       --crop_size=360,480 \
       --model_variant=resnet_v1_50_beta \
       --backbone_atrous_rates=1 \
       --backbone_atrous_rates=2 \
       --backbone_atrous_rates=4 \
       --decoder_output_stride=4 \
       --vis_logdir=${TRAIN_LOGDIR} \
       --checkpoint_dir=${TRAIN_LOGDIR} \
       --split_name=val \
       --dataset_name=camvid \
       --dataset_dir=dataset/CamVid/tfrecord
