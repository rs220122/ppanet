# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-28T07:58:56.053Z
# Description:
#
# ===============================================

DATASET=pascal
MODEL_VARIANT=resnet_v1_101_beta
TRAIN_LOGDIR=s_ppa/logs/${DATASET}/2019-12-05_train
DATASET_DIR=./dataset/VOCdevkit/tfrecord

python vis.py \
       --output_stride=8 \
       --batch_size=1 \
       --crop_size=512,512 \
       --model_variant=${MODEL_VARIANT} \
       --backbone_atrous_rates=1 \
       --backbone_atrous_rates=2 \
       --backbone_atrous_rates=4 \
       --decoder_output_stride=4 \
       --ppm_rates=1 \
       --ppm_rates=2 \
       --ppm_rates=3 \
       --ppm_rates=6 \
       --ppm_pooling_type=avg \
       --atrous_rates=12 \
       --atrous_rates=24 \
       --atrous_rates=36 \
       --module_order=ppm,aspp,pam \
       --checkpoint_dir=${TRAIN_LOGDIR} \
       --vis_logdir=${TRAIN_LOGDIR} \
       --split_name=val \
       --dataset_name=${DATASET} \
       --dataset_dir=${DATASET_DIR} \
       --save_labels 
       # --add_flipped_images \
       # --eval_scales=0.5 \
       # --eval_scales=0.75 \
       # --eval_scales=1.0 \
       # --eval_scales=1.25 \
       # --eval_scales=1.5
