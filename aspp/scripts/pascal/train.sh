# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-03T14:06:05.435Z
# Description:
#
# ===============================================
NOW_DATE=`date '+%F'`
TRAIN_LOGDIR='aspp/logs/pascal/log-'${NOW_DATE}
mkdir -p ${TRAIN_LOGDIR}
cp aspp/scripts/train.sh ${TRAIN_LOGDIR}'/train_copy.sh'
nohup python train.py \
                --output_stride=8 \
                --crop_size=512,512 \
                --batch_size=4 \
                --model_variant=resnet_v1_50_beta \
                --backbone_atrous_rates=1 \
                --backbone_atrous_rates=2 \
                --backbone_atrous_rates=4 \
                --base_learning_rate=0.01 \
                --weight_decay=0.0001 \
                --decoder_output_stride=4 \
                --atrous_rates=6 \
                --atrous_rates=12 \
                --atrous_rates=18 \
                --module_order=aspp \
                --tf_initial_checkpoint=./backbone_ckpt/resnet_v1_50/model.ckpt \
                --save_summaries_images \
                --dataset_name=pascal \
                --dataset_dir=./dataset/VOCdevkit/tfrecord \
                --train_logdir=${TRAIN_LOGDIR} \
                --train_steps=90000  > ${TRAIN_LOGDIR}'/out.log' &
less +F ${TRAIN_LOGDIR}'/out.log'
