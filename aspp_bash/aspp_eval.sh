# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-30T05:30:32.834Z
# Description:
#
# ===============================================

TRAIN_LOGDIR='logs/aspp/log-2019-09-09'

python eval.py \
       --output_stride=8 \
       --crop_size=360,480 \
       --batch_size=1 \
       --model_variant=resnet_v1_50_beta \
       --backbone_atrous_rates=1 \
       --backbone_atrous_rates=2 \
       --backbone_atrous_rates=4 \
       --decoder_output_stride=4 \
       --atrous_rates=6 \
       --atrous_rates=12 \
       --atrous_rates=18 \
       --eval_logdir=${TRAIN_LOGDIR} \
       --checkpoint_dir=${TRAIN_LOGDIR} \
       --dataset_name=camvid \
       --dataset_dir=dataset/CamVid/tfrecord \
       --split_name=val
