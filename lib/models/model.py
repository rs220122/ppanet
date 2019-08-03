# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-07-25T06:03:15.492Z
# Description:
#
# ===============================================

"""

"""
# packages
import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib import slim

# user packages
from lib.models.backbone import feature_extractor

FLAGS = tf.app.flags.FLAGS

# DEFINE FLAGS
# tf.app.flags.DEFINE_string('dataset_dir',
#                            os.path.join('dataset', 'CamVid', 'tfrecord'),
#                            'tfrecord directory')

def build_model(images,
                num_classes,
                model_variant,
                output_stride=8,
                fine_tune_batch_norm=True,
                weight_decay=0.0001,
                backbone_atrous_rate=None,
                is_training=True,
                ppm_rates=[1, 2, 3, 6],
                ppm_pooling_type='average'):

    if backbone_atrous_rate is not None:
        if 'beta' not in model_variant:
            raise ValueError('{} is not correct.'.format(model_variant))

    features, end_points = feature_extractor.extract_features(
                           images=images,
                           output_stride=output_stride,
                           model_variant=model_variant,
                           weight_decay=weight_decay,
                           is_training=is_training,
                           fine_tune_batch_norm=fine_tune_batch_norm,
                           preprocess_images=True,
                           multi_grid=backbone_atrous_rate,
                           preprocessed_images_dtype=images.dtype,
                           )

    print('build_model')

    for key, val in end_points.items():
        print('{}: {}'.format(key, val.shape))

    print('features: {}, shape: {}'.format(features.name, features.get_shape().as_list()))

    if ppm_rates:

        # ppm_features = _build_PPM(is_training=is_training,
        #                           fine_tune_batch_norm=fine_tune_batch_norm,
        #                           ppm_rates)
        batch_norm_params = {
            'is_training': is_training and fine_tune_batch_norm,
            'decay': 0.9996,
            'epsilon': 1e-5,
            'scale': True
        }
        features_list = [features]
        _, base_height, base_width, _ = features.get_shape().as_list()
        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                padding='SAME',
                stride=1):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                for rate in ppm_rates:
                    kernel_height = int(base_height / rate)
                    kernel_width  = int(base_width  / rate)
                    pool_feature = slim.avg_pool2d(features, [kernel_height, kernel_width], stride=[kernel_height, kernel_width])

                    pool_feature = slim.conv2d()
    return features, end_points
