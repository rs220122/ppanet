# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-03T14:06:05.435Z
# Description:
#
# ===============================================

NOW_DATE=`date '+%F'`
DATASET=cityscapes
MODEL_VARIANT=resnet_v1_101_beta
TRAIN_LOGDIR=da/logs/${DATASET}/${NOW_DATE}
DATASET_DIR='./dataset/cityscapes/tfrecord'

# training model on cityscapes while fite-tuning batch-norm params.
mkdir -p ${TRAIN_LOGDIR}
cp da/scripts/${DATASET}/train.sh ${TRAIN_LOGDIR}'/train_copy.sh'
python train.py \
                --output_stride=16 \
                --crop_size=768,768 \
                --batch_size=4 \
                --model_variant=${MODEL_VARIANT} \
                --backbone_atrous_rates=1 \
                --backbone_atrous_rates=2 \
                --backbone_atrous_rates=4 \
                --base_learning_rate=0.007 \
                --weight_decay=0.0001 \
                --decoder_output_stride=4 \
                --module_order=da \
                --tf_initial_checkpoint=./backbone_ckpt/resnet_v1_101_beta/model.ckpt \
                --dataset_name=${DATASET} \
                --dataset_dir=${DATASET_DIR} \
                --split_name=train_fine \
                --max_scale_factor=2.0 \
                --train_logdir=${TRAIN_LOGDIR} \
                --fine_tune_batch_norm=True \
                --train_steps=90000  > ${TRAIN_LOGDIR}'/out.log'

python train.py \
                --output_stride=16 \
                --crop_size=768,768 \
                --batch_size=4 \
                --model_variant=${MODEL_VARIANT} \
                --backbone_atrous_rates=1 \
                --backbone_atrous_rates=2 \
                --backbone_atrous_rates=4 \
                --base_learning_rate=0.0001 \
                --weight_decay=0.0001 \
                --decoder_output_stride=4 \
                --module_order=da \
                --tf_initial_checkpoint=${TRAIN_LOGDIR}/model.ckpt-90000 \
                --dataset_name=${DATASET} \
                --dataset_dir=${DATASET_DIR} \
                --split_name=train_fine \
                --max_scale_factor=2.0 \
                --train_logdir=${TRAIN_LOGDIR}_extra \
                --fine_tune_batch_norm=False \
                --train_steps=180000  > ${TRAIN_LOGDIR}'/out.log'
