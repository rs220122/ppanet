# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-03T14:06:05.435Z
# Description:
#
# ===============================================
NOW_DATE=`date '+%F'`
TRAIN_LOGDIR='logs/ppm/log-'${NOW_DATE}
mkdir -p ${TRAIN_LOGDIR}
nohup python train.py \
                --output_stride=16 \
                --ppm_rates=1 \
                --ppm_rates=2 \
                --ppm_rates=3 \
                --ppm_rates=6 \
                --crop_size=512,512 \
                --ppm_pooling_type=avg \
                --batch_size=8             \
                --model_variant=resnet_v1_50_beta \
                --tf_initial_checkpoint=./backbone_ckpt/resnet_v1_50/model.ckpt \
                --save_summaries_images \
                --backbone_atrous_rates=1 \
                --backbone_atrous_rates=2 \
                --backbone_atrous_rates=4 \
                --base_learning_rate=0.01 \
                --weight_decay=0.0001 \
                --train_logdir=${TRAIN_LOGDIR} \
                --decoder_output_stride=4     \
                --train_steps=10000 > ${TRAIN_LOGDIR}'/out.log'  &
less +F ${TRAIN_LOGDIR}'/out.log'
