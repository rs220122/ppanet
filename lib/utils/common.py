# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

# lib
import tensorflow as tf
import numpy as np

# user packages

# Define common flags for building model.

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_enum('image_format', 'png', ['jpg', 'jpeg', 'png'],
                         'Image format')

tf.app.flags.DEFINE_integer('output_stride',
                            16,
                            'The output stride.')

tf.app.flags.DEFINE_integer('decoder_output_stride',
                            None,
                            'The decoder stride. Bring the feature map from backbone.' +
                            'If this value is 4, bring the feature map from where backbone stride is 4.')

# Defaults to None. Set [1, 2, 4] when using provided
# 'resnet_v1_{50, 101}_beta' checkpoints.
tf.app.flags.DEFINE_multi_integer('backbone_atrous_rates',
                                  None,
                                  'Emplay a hierarchy atrous rate for resnet.')

tf.app.flags.DEFINE_multi_integer('ppm_rates',
                                  # [1, 2, 3, 6],
                                  None,
                                  'Pyramid Pooling Module each rate.')

tf.app.flags.DEFINE_enum('ppm_pooling_type',
                         'avg',
                         ['max', 'avg'],
                         'Pyramid Pooling Module Pooling method type.')

tf.app.flags.DEFINE_multi_integer('atrous_rates',
                                  # [6, 12, 18],
                                  None,
                                  'Atrous rates fro atrous spatial pyramid pooling.')


# Define dataset keys.
# このキーは、データセットを読み込むときに使われる
# This keys are used when programs loads dataset.
IMAGE = 'image'
LABEL = 'label'
IMAGE_FILENAME = 'image_filename'
IMAGE_HEIGHT = 'height'
IMAGE_WIDTH = 'width'
ORIGINAL_IMAGE = 'original'



def print_args():
    """ Print arguments. """
    print('-' * 40)
    print('flags information')
    print('-' * 40)
    keys = FLAGS.__flags.keys()
    max_string_len = np.max([len(key) for key in keys])
    string_format = '{:%d} : {}' % max_string_len
    for key in keys:
        if key in ['h', 'help', 'helpfull', 'helpshort']:
            pass
        else:
            print(string_format.format(key, FLAGS[key].value))
    print('-' * 40 + '\n')
