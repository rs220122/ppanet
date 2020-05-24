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
# ppm  => pyramid pooling module .
# aspp => atrous spatial pyramid pooling.
# da   => dual attention module. (composed of pam and cam.)
# pam  => position attention module.
# cam  => channel attention module.
# ppa  => pyramid pooling attention. This is my approach.
_MODULE_LIST = ['ppm', 'aspp', 'da', 'pam', 'cam', 'ppa']

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
                 module_order=None,
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
        # for decoder
        self.decoder_output_stride=decoder_output_stride

        self.pam_flag = False
        self.cam_flag = False

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

        # for debug
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
                features = slim.conv2d(backbone_features, 256, 1, scope='dimension_reduction')

                for i, module in enumerate(self._module_list, start=1):
                    tf.logging.info('{}-th module: {}'.format(i, module))
                    if module == 'ppm':
                        if self.ppm_rates is None:
                            raise ValueError('ppm_rates is None. But you set ppm to module_order.')
                        features = self._build_ppm(features, depth=depth)
                    elif module == 'aspp':
                        if self.atrous_rates is None:
                            raise ValueError('atrous_rates is None. But you set aspp to module_order.')
                        featuers = self._build_aspp(features, depth=depth)
                    elif module == 'da':
                        features = self._build_da(features)
                    elif module == 'pam':
                        features = self._build_position_attention(features)
                    elif module == 'cam':
                        features = self._build_channel_attention(features)
                    elif module == 'ppa':
                        features = self._build_ppa(features, depth=depth)

                features = slim.dropout(
                    features,
                    keep_prob=0.9,
                    is_training=self.is_training,
                    scope='f_map_dropout')

                if self.decoder_output_stride is not None:
                    tf.logging.info('Using decoder')
                    features = self._build_decoder(features, end_points)

                features = slim.conv2d(features, 256, 3, scope='conv1_before_logits')
                features = slim.conv2d(features, 256, 3, scope='conv2_before_logits')
                features = slim.conv2d(features, self.num_classes, 1, scope='conv_logits')

        return features


    def _build_ppm(self, features, depth):
        """Build the pyramid pooling module.

        Args:
            features: features.
            depth: depth for convolution layer.
        Return:
            ppm features.
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
            for i, rate in enumerate(self.ppm_rates):
                kernel_height = int(base_height / rate)
                kernel_width  = int(base_width  / rate)
                pool_feature = pooling_method(features, [kernel_height, kernel_width], stride=[kernel_height, kernel_width])
                pool_feature = slim.conv2d(pool_feature, depth, 1, scope='ppm_{}'.format(i))

                tf.logging.info('============= rate {} ==============='.format(rate))
                tf.logging.info("pool_feature's shape: {}".format(pool_feature.get_shape()))
                resized_feature = tf.image.resize_bilinear(pool_feature, [base_height, base_width], align_corners=True)
                features_list.append(resized_feature)
            ppm_features = tf.concat(features_list, axis=3)
            ppm_features = slim.conv2d(ppm_features, 256, 1, scope='conv_concat')
        return ppm_features


    def _build_aspp(self, features, depth, gap=False):
        """Build the atrous spatial pyramid pooling module.

        Args:
            features: features.
            depth: depth for convolution layer.
            gap: use global pooling layer. In the paper, True.
        Return:
            aspp features.
        """
        with tf.variable_scope('aspp', [features]):
            aspp_features_list = [features]
            if gap:
                _, f_h, f_w, _ = features.get_shape().as_list()
                image_level_features = slim.max_pool2d(features, [f_h, f_w], stride=[f_h, f_w])
                aspp_features_list.append(tf.image.resize_bilinear(image_level_features), align_corners=True)
            for i, rate in enumerate(self.atrous_rates):
                tf.logging.info('============= atrous rate {} ==============='.format(rate))
                if self.output_stride == 8:
                    aspp_features = slim.conv2d(
                        features, depth, 3, rate=rate, scope='aspp_rate{}'.format(rate//2))
                else:
                    aspp_features = slim.conv2d(
                        features, depth, 3, rate=rate, scope='aspp_rate{}'.format(rate))

                # aspp_features = slim.conv2d(
                #       features, depth, 3, rate=rate, scope='aspp_{}'.format(i))
                tf.logging.info("aspp_feature's shape: {}".format(aspp_features.get_shape()))
                aspp_features_list.append(aspp_features)
            features = tf.concat(aspp_features_list, axis=3)
            features = slim.conv2d(features, 256, 1, scope='conv_concat')
        return features


    def _build_position_attention(self, features1, features2=None):
        """Build the position attention module.

        Args:
            features1: features1. In the Dual Attention Network paper, this is defined as B.
            features2: features2. In the paper, this is defined as C.
        Returns:
            position attention features.
        """
        self.pam_flag=True

        with tf.variable_scope('position_attention', [features1, features2]):
            original_channel = features1.get_shape().as_list()[3]

            features_A = slim.conv2d(features1, original_channel//8, 1, scope='attention_A')
            if features2 is not None:
                features_B = slim.conv2d(features2, original_channel//8, 1, scope='attention_B')
            else:
                features_B = slim.conv2d(features1, original_channel//8, 1, scope='attention_B')
            features_C = slim.conv2d(features1, original_channel//8, 1, scope='attention_C')
            f_shape = tf.shape(features_A)
            self._f_shape = f_shape

            reshaped_features_A = tf.reshape(features_A, [f_shape[0], f_shape[1]*f_shape[2], f_shape[3]])
            reshaped_features_B = tf.reshape(features_B, [f_shape[0], f_shape[1]*f_shape[2], f_shape[3]])
            reshaped_features_C = tf.reshape(features_C, [f_shape[0], f_shape[1]*f_shape[2], f_shape[3]])

            attention_map = tf.matmul(reshaped_features_A, tf.transpose(reshaped_features_B, [0, 2, 1]))
            self._attention_map = tf.nn.softmax(attention_map, axis=2)

            attention_features = tf.matmul(self._attention_map, reshaped_features_C)
            attention_features = tf.reshape(attention_features, f_shape)
            attention_features = slim.conv2d(attention_features, original_channel, 1)

            alpha_initializer = tf.constant_initializer([0])
            alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32, initializer=alpha_initializer)
            if self.is_training:
                tf.summary.scalar('pam_alpha', alpha)

            attention_features = alpha * attention_features + features1

        return attention_features


    def _build_channel_attention(self, features1):
        """Build the channel attention module.

        Args:
            features:
        Returns:
            cam features.
        """
        self.cam_flag = True

        with tf.variable_scope('channel_attention', [features]):
            f_shape = tf.shape(features)

            reshaped_features = tf.reshape(features, [f_shape[0], f_shape[1]*f_shape[2], f_shape[3]])

            channel_attention_map = tf.matmul(tf.transpose(reshaped_features, [0, 2, 1]), reshaped_features)
            self._ca_map = tf.nn.softmax(channel_attention_map, axis=1)

            cam_features = tf.matmul(reshaped_features, self._ca_map)
            cam_features = tf.reshape(cam_features, f_shape)

            alpha_initializer = tf.constant_initializer([0])
            alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32, initializer=alpha_initializer)
            if self.is_training:
                tf.summary.scalar('cam_alpha', alpha)
            cam_features = alpha * cam_features + features

        return cam_features


    def _build_da(self, features):
        """Build dual attention module.

        Args:
            features: feature map. (BHWC format)
        Returns:
            dual attention features.
        """
        # position attention
        tf.logging.info('building position attention module ...')
        pam_features = self._build_position_attention(features)
        # channel attention
        tf.logging.info('building channel attention module ...')
        cam_features = self._build_channel_attention(features)
        da_features = pam_features + cam_features
        return da_features




    def _build_ppa(self, features, depth=256):
        """
        """
        with tf.variable_scope('pyramid_pooling_attention', [features]):
            ppm_features = self._build_ppm(features, depth=depth)
            aspp_features = self._build_aspp(features, depth=depth)

            ppa_features = self._build_position_attention(ppm_features, aspp_features)
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


    def summary_attention_map(self, input, labels):
        """ Create tensorboard summary for attention map.

        Args:

        Returns:

        """
        temp_input = tf.identity(input[0], name='Image_for_Attention_Map')
        crop_size = tf.shape(temp_input)
        temp_attention_map = self._attention_map[0]
        # feature map shape (attention map shape)
        f_height, f_width = self._f_shape[1], self._f_shape[2]

        # ３つのピクセルをサンプリング
        SUMPLING_NUM = 3
        sumpling_pixels = tf.random.uniform([SUMPLING_NUM],
                                            maxval=f_height*f_width,
                                            dtype=tf.int32)


        # 画像のピクセルにマーカーをつける
        temp_attention_map = tf.gather(temp_attention_map, sumpling_pixels)
        temp_attention_map = tf.reshape(temp_attention_map,
                                        [SUMPLING_NUM, f_height, f_width])

        heights_offset = (sumpling_pixels / f_width) * (crop_size[0] / f_height)
        widths_offset = tf.cast(sumpling_pixels % f_width, tf.float64) * (crop_size[1] / f_width)
        heights_offset = tf.cast(heights_offset, tf.int32)
        widths_offset = tf.cast(widths_offset, tf.int32)

        mask_max_heights = tf.math.minimum(heights_offset+6, crop_size[0])
        mask_min_heights = tf.math.maximum(heights_offset-6, 0)
        mask_max_widths = tf.math.minimum(widths_offset+6, crop_size[1])
        mask_min_widths = tf.math.maximum(widths_offset-6, 0)
        h, w = tf.range(crop_size[0]), tf.range(crop_size[1])
        W, H = tf.meshgrid(w, h)

        ones = tf.ones([crop_size[0], crop_size[1]])
        reds = tf.stack([ones*255, ones, ones], axis=2)
        for i in range(SUMPLING_NUM):
            mask_ma_h = mask_max_heights[i]
            mask_mi_h = mask_min_heights[i]
            mask_ma_w = mask_max_widths[i]
            mask_mi_w = mask_min_widths[i]
            H_logical = tf.math.logical_and(tf.greater_equal(H, mask_mi_h), tf.greater_equal(mask_ma_h, H))
            W_logical = tf.math.logical_and(tf.greater_equal(W, mask_mi_w), tf.greater_equal(mask_ma_w, W))
            mask = tf.math.logical_and(H_logical, W_logical)
            temp_input = tf.where(tf.stack([mask, mask, mask], axis=2), reds, temp_input)

        tf.summary.image('samples/attention_input', tf.expand_dims(temp_input, 0))
        tf.summary.image('samples/attention_map', tf.expand_dims(temp_attention_map, -1))



    def predict_labels(self, images, add_flipped_images=False):

        logits = self.build(images)
        if add_flipped_images:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                logits_reverse = self.build(tf.reverse(images, [2]))
                logits = logits + tf.reverse(logits_reverse, [2])

        logits = tf.image.resize_bilinear(logits, tf.shape(images)[1:3], align_corners=True)

        predictions = tf.argmax(logits, 3)

        return predictions, tf.nn.softmax(logits)


    def predict_labels_for_multiscale(self, images, add_flipped_images=False, eval_scales=[1.0]):
        logits_list = []
        _, h, w, c = images.get_shape().as_list()

        for i, image_scale in enumerate(eval_scales):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if i else None):
                scaled_shape = (tf.cast(h * image_scale, dtype=tf.int32), tf.cast(w * image_scale, dtype=tf.int32))

                scaled_images = tf.image.resize_bilinear(images, scaled_shape, align_corners=True)

                logits = self.build(scaled_images)
            if add_flipped_images:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    logits_reverse = self.build(tf.reverse(scaled_images, [2]))
                    logits = logits + tf.reverse(logits_reverse, [2])

            logits = tf.image.resize_bilinear(logits, tf.shape(images)[1:3], align_corners=True)
            logits_list.append(logits)

        logits = tf.add_n(logits_list)

        predictions = tf.argmax(logits, 3)
        return predictions, tf.nn.softmax(logits)
