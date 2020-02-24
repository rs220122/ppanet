# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-08-03T14:06:05.435Z
# Description:
#
# ===============================================

NOW_DATE=`date '+%F'`
DATASET=pascal
MODEL_VARIANT=resnet_v1_101_beta
COCO_PRETRAIN_CKPT=aspp/logs/coco_thing/2019-11-12/model.ckpt-150000
TRAIN_LOGDIR_AUG=aspp/logs/${DATASET}/2019-11-16_trainaug
DATASET_DIR='./dataset/VOCdevkit/tfrecord'

# training model on pascal  voc trainaug set.
mkdir -p ${TRAIN_LOGDIR_AUG}
cp aspp/scripts/pascal/train.sh ${TRAIN_LOGDIR_AUG}'/train_copy.sh'
python train.py \
                --output_stride=16 \
                --crop_size=512,512 \
                --batch_size=8 \
                --model_variant=${MODEL_VARIANT} \
                --backbone_atrous_rates=1 \
                --backbone_atrous_rates=2 \
                --backbone_atrous_rates=4 \
                --base_learning_rate=0.001 \
                --weight_decay=0.0001 \
                --decoder_output_stride=4 \
                --atrous_rates=6 \
                --atrous_rates=12 \
                --atrous_rates=18 \
                --module_order=aspp \
                --tf_initial_checkpoint=${COCO_PRETRAIN_CKPT} \
                --dataset_name=${DATASET} \
                --dataset_dir=${DATASET_DIR} \
                --split_name=trainaug \
                --max_scale_factor=2.0 \
                --train_logdir=${TRAIN_LOGDIR_AUG} \
                --fine_tune_batch_norm=False \
                --train_steps=60000  > ${TRAIN_LOGDIR_AUG}'/out.log'

# training model on pascal voc train set.
TRAIN_LOGDIR=aspp/logs/${DATASET}/${NOW_DATE}_train

mkdir -p ${TRAIN_LOGDIR}
python train.py \
		--output_stride=16 \
		--crop_size=512,512 \
		--batch_size=8 \
		--model_variant=${MODEL_VARIANT} \
		--backbone_atrous_rates=1 \
		--backbone_atrous_rates=2 \
		--backbone_atrous_rates=4 \
		--base_learning_rate=0.0001 \
		--weight_decay=0.0001 \
		--decoder_output_stride=4 \
		--atrous_rates=6 \
		--atrous_rates=12 \
		--atrous_rates=18 \
		--module_order=aspp \
		--tf_initial_checkpoint=${TRAIN_LOGDIR_AUG}'/model.ckpt-60000' \
		--dataset_name=${DATASET} \
		--dataset_dir=${DATASET_DIR} \
		--split_name=train \
		--max_scale_factor=2.0 \
		--train_logdir=${TRAIN_LOGDIR} \
		--fine_tune_batch_norm=False \
		--train_steps=60000 > ${TRAIN_LOGDIR}'/out.log'
