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
cp ppm_train.sh ${TRAIN_LOGDIR}'/ppm_train.sh'
nohup python train.py \
                --output_stride=8 \
                --ppm_rates=1 \
                --ppm_rates=2 \
                --ppm_rates=3 \
                --ppm_rates=6 \
                --crop_size=360,480 \
                --min_scale_factor=0.5 \
                --max_scale_factor=2.0 \
                --scale_factor_step_size=0.25 \
                --ppm_pooling_type=avg \
                --batch_size=8 \
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
                --train_steps=90000 > ${TRAIN_LOGDIR}'/out.log'  &
less +F ${TRAIN_LOGDIR}'/out.log'
