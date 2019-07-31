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

# user packages

FLAGS = tf.app.flags.FLAGS

# DEFINE FLAGS
tf.app.flags.DEFINE_string('dataset_dir',
                           os.path.join('dataset', 'CamVid', 'tfrecord'),
                           'tfrecord directory')

def build_model(input, num_classes, fine_tun_batch_norm=True):
    pass
