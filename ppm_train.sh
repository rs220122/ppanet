# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-03T14:06:05.435Z
# Description:
#
# ===============================================
TRAIN_LOGDIR='ppm_out_stride8'
mkdir ${TRAIN_LOGDIR}
nohup python train.py --output_stride=8 \
                --ppm_rates=1 \
                --ppm_rates=2 \
                --ppm_rates=3 \
                --ppm_rates=6 \
                --crop_size=512,512 \
                --ppm_pooling_type=average \
                --batch_size=4             \
                --model_variant=resnet_v1_50_beta \
                --tf_initial_checkpoint=./backbone_ckpt/resnet_v1_50/model.ckpt \
                --save_summaries_images \
                --backbone_atrous_rate=1 \
                --backbone_atrous_rate=2 \
                --backbone_atrous_rate=4 \
                --base_learning_rate=0.01 \
                --weight_decay=0.0001 \
                --train_logdir=${TRAIN_LOGDIR} \
                --train_steps=100000 > ${TRAIN_LOGDIR}'/out.log'  &
