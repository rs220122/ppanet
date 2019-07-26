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
from dataset import generator

FLAGS = tf.app.flags.FLAGS

# DEFINE FLAGS
# tf.app.flags.DEFINE_string('dataset_dir',
#                            os.path.join('dataset', 'CamVid', 'tfrecord'),
#                            'tfrecord directory')


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


def main(argv):

    print_args()

    print('Training on %s %s set' % (FLAGS.split_name, FLAGS.dataset_name))

    graph = tf.Graph()
    crop_size = [int(val) for val in FLAGS.crop_size]
    with graph.as_default():
        dataset = generator.Dataset(FLAGS.dataset_dir,
                                    FLAGS.dataset_name,
                                    FLAGS.split_name,
                                    32,
                                    crop_size,
                                    min_resize_value=FLAGS.min_resize_value,
                                    max_resize_value=FLAGS.max_resize_value,
                                    resize_factor=FLAGS.resize_factor,
                                    min_scale_factor=FLAGS.min_scale_factor,
                                    max_scale_factor=FLAGS.max_scale_factor,
                                    scale_factor_step_size=FLAGS.scale_factor_step_size,
                                    is_training=True,
                                    model_variant=FLAGS.model_variant,
                                    should_shuffle=True)

        iterator = dataset.get_one_shot_iterator()
        samples  = iterator.get_next()



if __name__ == '__main__':
    tf.app.run()
