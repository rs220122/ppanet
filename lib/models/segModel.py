# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-09-18T06:57:02.770Z
# Description:
#
# ===============================================

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
# ppm => pyramid pooling module .
# aspp => atrous spatial pyramid pooling.
# sa => self-attention
# ppa => pyramid pooling attention. my approach.
_MODULE_LIST = ['ppm', 'aspp', 'sa', 'ppa']

class SegModel(object):

    def __init__(self, num_classes, model_variant,
                 output_stride=8,
                 fine_tune_batch_norm=True,
                 weight_decay=0.0001,
                 backbone_atrous_rates=None,
                 is_training=True,
                 ppm_rates=None,
                 ppm_pooling_type='avg',
                 atrous_rates=None,
                 self_attention_flag=False,
                 module_order=None,
                 ppa_flag=False,
                 decoder_output_stride=None):

        if 'beta' in model_variant:
            if backbone_atrous_rates is None:
                raise ValueError("You must set 'backone_atrous_rates'." +
                                 "because model_variant is 'resnet_beta'.")
        else:
            if backbone_atrous_rates is not None:
                raise ValueError("Now you set 'backbone_atrous_rates'." +
                                 "but model_variant is not 'resnet_beta'.")

        self.num_classes = num_classes
        self.model_variant=model_variant
        self.output_stride=output_stride
        self.fine_tune_batch_norm=fine_tune_batch_norm
        self.weight_decay=weight_decay
        self.backbone_atrous_rates=backbone_atrous_rates
        self.is_training=is_training

        # for pyramid pooling module
        self.ppm_rates=ppm_rates
        self.ppm_pooling_type=ppm_pooling_type
        # for atrous spatial pyramid pooling
        self.atrous_rates=atrous_rates
        # for self-attention
        self.self_attention_flag=self_attention_flag
        # for pyramid pooling attention
        self.ppa_flag=ppa_flag
        # for decoder
        self.decoder_output_stride=decoder_output_stride

        self._module_list = []
        for module in module_order:
            if module in _MODULE_LIST:
                self._module_list.append(module)
            else:
                raise ValueError('{} module is not implemented.'.format(module) +
                                 'now implemented modules is {}'.format(_MODULE_LIST))


    def build(self, images):

        _, image_height, image_width, _ = images.get_shape().as_list()
        backbone_features, end_points = feature_extractor.extract_features(
                                images=images,
                                output_stride=self.output_stride,
                                model_variant=self.model_variant,
                                weight_decay=self.weight_decay,
                                is_training=self.is_training,
                                fine_tune_batch_norm=self.fine_tune_batch_norm,
                                preprocess_images=True,
                                multi_grid=self.backbone_atrous_rates,
                                preprocessed_images_dtype=images.dtype,)

        tf.logging.info('build %s model.', self.model_variant)

        for key, val in end_points.items():
            tf.logging.debug('{}: {}'.format(key, val.shape))

        tf.logging.debug('backbone_feature name: {}'.format(backbone_features.name))
        tf.logging.debug('backbone_feature shape: {}'.format(backbone_features.get_shape()))

        batch_norm_params = {
            'is_training': self.is_training and self.fine_tune_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True
        }

        with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(self.weight_decay),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding='SAME',
            stride=1,):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                depth=256
                features = slim.conv2d(backbone_features, 512, 1, scope='dimension_reduction')

                for i, module in enumerate(self._module_list, start=1):
                    tf.logging.info('{}-th module: {}'.format(i, module))
                    if module == 'ppm':
                        if self.ppm_rates is None:
                            raise ValueError('ppm_rates is None. But module_order set ppm.')
                        features = self._build_ppm(features, depth=depth)
                    elif module == 'aspp':
                        if self.atrous_rates is None:
                            raise ValueError('atrous_rates is None. But module_order set aspp.')
                        featuers = self._build_aspp(features, depth=depth)
                    elif module == 'sa':
                        if not self.self_attention_flag:
                            raise ValueError('self_attention_flag is False. But module_order set self_attention.')
                        features = self._build_self_attention(features)
                    elif module == 'ppa':
                        if not self.ppa_flag:
                            raise ValueError('ppa_flag is False. But module_order set ppa.')
                        features = self._build_ppa(features, depth=depth)

                if self.decoder_output_stride is not None:
                    tf.logging.info('Using decoder')
                    features = self._build_decoder(features, end_points)

                features = slim.conv2d(features, 256, 3, scope='conv1_before_logits')
                features = slim.conv2d(features, 256, 3, scope='conv2_before_logits')
                features = slim.conv2d(features, self.num_classes, 1, scope='conv_logits')

        return features


    def _build_ppm(self, features, depth):
        """
        """
        features_list = [features]
        _, base_height, base_width, _ = features.get_shape().as_list()

        if self.ppm_pooling_type == 'avg':
            pooling_method = slim.avg_pool2d
        elif self.ppm_pooling_type == 'max':
            pooling_method = slim.max_pool2d
        else:
            raise ValueError('pooling type {} is not supported. now supporting avg or max.'.format(self.ppm_pooling_type))

        with tf.variable_scope('ppm', [features]):
            for rate in self.ppm_rates:
                kernel_height = int(base_height / rate)
                kernel_width  = int(base_width  / rate)
                pool_feature = pooling_method(features, [kernel_height, kernel_width], stride=[kernel_height, kernel_width])
                pool_feature = slim.conv2d(pool_feature, depth, 1, scope='kernel_{}'.format(rate))

                tf.logging.info('============= rate {} ==============='.format(rate))
                tf.logging.info("pool_feature's shape: {}".format(pool_feature.get_shape()))
                resized_feature = tf.image.resize_bilinear(pool_feature, [base_height, base_width], align_corners=True)
                features_list.append(resized_feature)
            ppm_features = tf.concat(features_list, axis=3)
            ppm_features = slim.conv2d(ppm_features, 512, 1, scope='conv_concat')
        return ppm_features


    def _build_aspp(self, features, depth):
        """
        """
        with tf.variable_scope('aspp', [features]):
            aspp_features_list = [features]
            for rate in self.atrous_rates:
                tf.logging.info('============= atrous rate {} ==============='.format(rate))
                aspp_features = slim.conv2d(
                    features, depth, 3, rate=rate, scope='aspp_rate{}'.format(rate))
                tf.logging.info("aspp_feature's shape: {}".format(aspp_features.get_shape()))
                aspp_features_list.append(aspp_features)
            features = tf.concat(aspp_features_list, axis=3)
            features = slim.conv2d(features, 512, 1, scope='conv_concat')
        return features


    def _build_self_attention(self, features1, features2=None):
        """
        """
        with tf.variable_scope('self_attention', [features1, features2]):
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
            if self.is_training:
                tf.summary.scalar('attention_alpha', alpha)


            attention_features = alpha * attention_features + features1

        return attention_features


    def _build_ppa(self, features, end_points, depth=256):
        """
        """
        with tf.variable_scope('pyramid_pooling_attention', [features]):
            ppm_features = self._build_ppm(features, depth=depth)
            aspp_featuers = self._build_aspp(features, depth=depth)

            ppa_features = self._build_self_attention(ppm_features, aspp_features)
        return ppa_features


    def _build_decoder(self, features, end_points):
        """
        """
        with tf.variable_scope('decoder', [features]):
            feature_names = feature_extractor.networks_to_feature_maps[self.model_variant][
                DECODER_END_POINTS][self.decoder_output_stride]

            for feature_name in feature_names:
                feature_name = feature_extractor.name_scope[self.model_variant] + '/' + feature_name
                features_from_backbone = end_points[feature_name]
                _, decoder_height, decoder_width, _ = features_from_backbone.get_shape().as_list()
                decoder_features_list = [tf.image.resize_bilinear(features, [decoder_height, decoder_width], align_corners=True)]
                decoder_features_list.append(
                    slim.conv2d(features_from_backbone, 48, 1))
        return tf.concat(decoder_features_list, axis=3)


    def predict_labels(self, images):

        logits = self.build(images)

        logits = tf.image.resize_bilinear(logits, tf.shape(images)[1:3], align_corners=True)

        predictions = tf.argmax(logits, 3)

        return predictions
