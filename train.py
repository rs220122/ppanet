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
from lib.models import model

FLAGS = tf.app.flags.FLAGS

# DEFINE FLAGS
tf.app.flags.DEFINE_float('base_learning_rate',
                          0.0001,
                          'Initial learning rate.')

tf.app.flags.DEFINE_float('weight_decay',
                          0.00004,
                          'The rate of the weight decay for training.')

tf.app.flags.DEFINE_integer('output_stride',
                            8,
                            'The output stride.')

tf.app.flags.DEFINE_boolean('fine_tune_batch_norm',
                            True,
                            'Whether finetuning batch normalization value.')


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


def _build_model(input, num_classes, is_training=True, fine_tune_batch_norm):
    """ Build model.

    Args:
        iterator:
        num_classes:

    Returns:

    """

def main(argv):

    print_args()

    print('Training on %s %s set' % (FLAGS.split_name, FLAGS.dataset_name))

    graph = tf.Graph()
    crop_size = [int(val) for val in FLAGS.crop_size]
    with graph.as_default():
        # create dataset generator
        dataset = generator.Dataset(
                    dataset_dir=FLAGS.dataset_dir,
                    dataset_name=FLAGS.dataset_name,
                    split_name=FLAGS.split_name,
                    batch_size=FLAGS.batch_size,
                    crop_size=[int(val) for val in FLAGS.crop_size],
                    min_resize_value=FLAGS.min_resize_value,
                    max_resize_value=FLAGS.max_resize_value,
                    resize_factor=FLAGS.resize_factor,
                    min_scale_factor=FLAGS.min_scale_factor,
                    max_scale_factor=FLAGS.max_scale_factor,
                    scale_factor_step_size=FLAGS.scale_factor_step_size,
                    is_training=True,
                    model_variant=FLAGS.model_variant,
                    should_shuffle=True,
                    should_repeat=True)

        iterator = dataset.get_one_shot_iterator()
        global_step = tf.train.get_or_create_global_step()

        # create learning_rate
        learning_rate = tf.train.polynomial_decay(
                            learning_rate=FLAGS.base_learning_rate,
                            global_step=global_step,
                            decay_step=FLAGS.decay_step,
                            end_learning_rate=0,
                            power=0.9)
        tf.summary.scalar('learning_rate', learning_rate)

        # create optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)

        # build models
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            samples = iterator.get_next()
            input = tf.identity(samples['image'], name='Input Image')
            label = tf.identity(samples['label'], name='Semantic Label')

            is_training=True,
            logits = model.build_model(input,
                                       dataset.num_classes,
                                       fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)
            loss = _calculate_loss(logits, loss)
            train_op =





if __name__ == '__main__':
    tf.app.run()
