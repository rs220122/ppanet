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
import sys

# user packages
from lib.models.backbone import feature_extractor
from lib.models.backbone.feature_extractor import DECODER_END_POINTS

# module list.
# ppm  => pyramid pooling module.
# aspp => atrous spatial pyramid pooling.
# sa   => self-attention.
# ppa  => pyramid pooling attention. my aproach.
_MODULE_LIST = ['ppm', 'aspp', 'sa', 'ppa']

def _build_ppm(features, ppm_rates, ppm_pooling_type, depth=256):

    features_list = [features]
    _, base_height, base_width, _ = features.get_shape().as_list()

    if ppm_pooling_type == 'avg':
        pooling_method = slim.avg_pool2d
    elif ppm_pooling_type == 'max':
        pooling_method = slim.max_pool2d
    else:
        raise ValueError('pooling type {} is not supported. now supporting avg or max.'.format(ppm_pooling_type))

    with tf.variable_scope('ppm', [features]):
        for rate in ppm_rates:
            kernel_height = int(base_height / rate)
            kernel_width  = int(base_width  / rate)
            pool_feature = pooling_method(features, [kernel_height, kernel_width], stride=[kernel_height, kernel_width])
            pool_feature = slim.conv2d(pool_feature, depth, 1, scope='kernel_{}'.format(rate))
            print('============= rate {} ==============='.format(rate))
            print('pool_feature.shape: {}'.format(pool_feature.get_shape().as_list()))
            resized_feature = tf.image.resize_bilinear(pool_feature, [base_height, base_width], align_corners=True)
            print('resized_feature.shape: {}'.format(resized_feature.get_shape().as_list()))
            features_list.append(resized_feature)
        ppm_features = tf.concat(features_list, axis=3)
        ppm_features = slim.conv2d(ppm_features, 512, 3, scope='conv_concat')

    return ppm_features


def _build_aspp(features, atrous_rates, depth=256):
    with tf.variable_scope('aspp', [features]):
        aspp_features_list = [features]
        for rate in atrous_rates:
            print('============= atrous rate {} ==============='.format(rate))
            aspp_features = slim.conv2d(
                features, depth, 3, rate=rate, scope='aspp_rate{}'.format(rate))
            print('aspp_feature.shape: {}'.format(aspp_features.get_shape().as_list()))
            aspp_features_list.append(aspp_features)
        features = tf.concat(aspp_features_list, axis=3)
        features = slim.conv2d(features, 512, 3, scope='conv_concat')
    return features


def _build_self_attention(features1, features2=None):

    with tf.variable_scope('self_attention', [features1]):
        original_channel = features1.get_shape().as_list()[3]
        features_A = slim.conv2d(features1, original_channel//8, 1, scope='attention_A')
        if features2 is not None:
            features_B = slim.conv2d(features2, original_channel//8, 1, scope='attention_B')
        else:
            features_B = slim.conv2d(features1, original_channel//8, 1, scope='attention_B')
        features_C = slim.conv2d(features1, original_channel//8, 1, scope='attention_C')
        f_shape = tf.shape(features_A)

        reshaped_features_A = tf.reshape(features_A, [f_shape[0], f_shape[1]*f_shape[2], f_shape[3]])
        reshaped_features_B = tf.reshape(features_B, [f_shape[0], f_shape[1]*f_shape[2], f_shape[3]])
        reshaped_features_C = tf.reshape(features_C, [f_shape[0], f_shape[1]*f_shape[2], f_shape[3]])

        attention_map = tf.matmul(reshaped_features_A, tf.transpose(reshaped_features_B, [0, 2, 1]))
        attention_map = tf.nn.softmax(attention_map, axis=2)

        attention_features = tf.matmul(attention_map, reshaped_features_C)
        attention_features = tf.reshape(attention_features, f_shape)
        attention_features = slim.conv2d(attention_features, original_channel, 1)

        alpha_initializer = tf.constant_initializer([0])
        alpha = tf.get_variable('attention_alpha', shape=[], dtype=tf.float32, initializer=alpha_initializer)
        tf.summary.scalar('attention_alpha', alpha)

        attention_features = alpha * attention_features + features1

    return attention_features


def _build_ppa(features, ppm_rates, ppm_pooling_type, atrous_rates, depth=256):

    with tf.variable_scope('pyramid_pooling_attention', [features]):

        ppm_features = _build_ppm(features, ppm_rates, ppm_pooling_type, depth=depth//2)
        aspp_features = _build_aspp(features, atrous_rates, depth=depth//2)

        ppa_features = _build_self_attention(ppm_features, aspp_features)

    return ppa_features


def _build_decoder(features, end_points, model_variant, decoder_output_stride):

    with tf.variable_scope('decoder', [features]):
        feature_names = feature_extractor.networks_to_feature_maps[model_variant][
            DECODER_END_POINTS][decoder_output_stride]

        for feature_name in feature_names:
            feature_name = feature_extractor.name_scope[model_variant] + '/' + feature_name
            features_from_backbone = end_points[feature_name]
            _, decoder_height, decoder_width, _ = features_from_backbone.get_shape().as_list()
            decoder_features_list = [tf.image.resize_bilinear(features, [decoder_height, decoder_width], align_corners=True)]
            decoder_features_list.append(
                slim.conv2d(features_from_backbone, 48, 1))
    return tf.concat(decoder_features_list, axis=3)


def build_model(images,
                num_classes,
                model_variant,
                output_stride=8,
                fine_tune_batch_norm=True,
                weight_decay=0.0001,
                backbone_atrous_rates=None,
                is_training=True,
                ppm_rates=None,
                ppm_pooling_type='avg',
                decoder_output_stride=None,
                atrous_rates=None,
                self_attention_flag=False,
                module_order=None,
                ppa_flag=False):

    # backbone_atrous_rates requires model_variant=='resnet_beta'.
    if backbone_atrous_rates is not None:
        if 'beta' not in model_variant:
            raise ValueError("You set 'backbone_atrous_rates'." +
                             "'model_variant' is not 'resnet_beta'. you set {}".format(model_variant))
    _, images_height, images_width, _ = images.get_shape().as_list()
    backbone_features, end_points = feature_extractor.extract_features(
                           images=images,
                           output_stride=output_stride,
                           model_variant=model_variant,
                           weight_decay=weight_decay,
                           is_training=is_training,
                           fine_tune_batch_norm=fine_tune_batch_norm,
                           preprocess_images=True,
                           multi_grid=backbone_atrous_rates,
                           preprocessed_images_dtype=images.dtype,
                           )

    print("build {} model.".format(model_variant))

    for key, val in end_points.items():
        print('{}: {}'.format(key, val.shape))

    print('features: {}, shape: {}'.format(backbone_features.name, backbone_features.get_shape().as_list()))

    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True
    }

    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            depth = 256
            features = slim.conv2d(backbone_features, 512, 1, scope='dimension_reduction')

            module_list = []
            # if two or three modules are specified.
            if (((atrous_rates is not None) and (ppm_rates is not None)) or
               ((ppm_rates is not None) and self_attention_flag) or
               (self_attention_flag and (atrous_rates is not None))):
                if module_order is not None:
                    for module in module_order:
                        if module in _MODULE_LIST:
                            module_list.append(module)
                        else:
                            raise ValueError('{} module is not implemented.' +
                                             'now implemented modules is {}'.format(module, _MODULE_LIST))
                elif ppa_flag:
                    module_list.append('ppa')
                else:
                    print('ppa_flag: {}'.format(ppa_flag))
                    raise ValueError('There are two modules you specify. you should specify module_order in FLAG.')

            # if only one module are specified.
            else:
                if ppm_rates is not None:
                    module_list.append('ppm')
                elif atrous_rates is not None:
                    module_list.append('aspp')
                elif self_attention_flag:
                    module_list.append('sa')
                else:
                    print('NO EXTRA MODULE')


            for i, module in enumerate(module_list):
                tf.logging.info('{}-th module: {}'.format(i+1, module))
                if module == 'ppm':
                    features = _build_ppm(features, ppm_rates, ppm_pooling_type, depth=depth)
                elif module == 'aspp':
                    features = _build_aspp(features, atrous_rates, depth=depth)
                elif module == 'sa':
                    features = _build_self_attention(features)
                elif module == 'ppa':
                    features = _build_ppa(features, ppm_rates, ppm_pooling_type, atrous_rates, depth=depth)


            # if (ppm_rates is not None) and (atrous_rates is not None) and (self_attention_flag):
            #     raise ValueError('Both ppm and aspp are set.' +
            #                      'You must take away either ppm or aspp.')

            # if ppm_rates is not None:
            #     # perform pyramid pooling module proposed by PSPNet.
            #     features = _build_ppm(features, ppm_rates, ppm_pooling_type, depth=depth)
            #
            # if atrous_rates is not None:
            #     features = _build_aspp(features, atrous_rates, depth=depth)
            #
            # if self_attention_flag:
            #     features = _build_self_attention(features)

            if decoder_output_stride is not None:
                features = _build_decoder(features, end_points, model_variant, decoder_output_stride)

            features = slim.conv2d(features, 256, 3, scope='conv1_before_logits')
            features = slim.conv2d(features, 256, 3, scope='conv2_before_logits')
            features = slim.conv2d(features, num_classes, 1, scope='conv_logits')
    sys.stdout.flush()
    return features


def predict_labels(images,
                   num_classes,
                   model_variant,
                   output_stride,
                   backbone_atrous_rates,
                   ppm_rates,
                   ppm_pooling_type,
                   decoder_output_stride,
                   atrous_rates,
                   self_attention_flag,
                   ppa_flag):

    logits = build_model(images,
                         num_classes=num_classes,
                         model_variant=model_variant,
                         output_stride=output_stride,
                         backbone_atrous_rates=backbone_atrous_rates,
                         ppm_rates=ppm_rates,
                         ppm_pooling_type=ppm_pooling_type,
                         decoder_output_stride=decoder_output_stride,
                         atrous_rates=atrous_rates,
                         is_training=False,
                         fine_tune_batch_norm=False,
                         self_attention_flag=self_attention_flag,
                         ppa_flag=ppa_flag)


    logits = tf.image.resize_bilinear(logits, tf.shape(images)[1:3], align_corners=True)

    predictions = tf.argmax(logits, 3)

    return predictions
