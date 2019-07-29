# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description: Tests for deeplab.datasets.generator.
#
# ===============================================

# lib
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import collections
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

# user packages
import generator

FLAGS = tf.app.flags.FLAGS


def main(argv):
    crop_size = [int(FLAGS.crop_size[0]), int(FLAGS.crop_size[1])]
    dataset = generator.Dataset(FLAGS.dataset_dir,
                                FLAGS.dataset_name,
                                FLAGS.split_name,
                                10,
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
    samples = iterator.get_next()

    sess = tf.Session()

    print('=' * 10 + 'samples content' + '=' * 10)
    for key, val in samples.items():
        print('samples[{}]: {}'.format(key, val.shape))
    print('=' * 35)
    res = sess.run(samples)
    print(res.keys())

    image = res['image']
    ann = res['label']

    print('image.shape => ', image.shape)
    print('ann.shape   => ', ann.shape)

    figure = plt.figure(figsize=(20, 10))
    num_imgs = 2
    gridspec_master = GridSpec(nrows=num_imgs, ncols=2)

    for i in range(num_imgs):
        grid_sub_1 = GridSpecFromSubplotSpec(nrows=1,
                                             ncols=1,
                                             subplot_spec=gridspec_master[i, 0])
        axes_1 = figure.add_subplot(grid_sub_1[:, :])
        axes_1.set_xticks([])
        axes_1.set_yticks([])
        axes_1.imshow(image[i].astype(np.uint8))
        axes_1.set_title('target')

        grid_sub_2 = GridSpecFromSubplotSpec(nrows=1,
                                             ncols=1,
                                             subplot_spec=gridspec_master[i, 1])
        axes_2 = figure.add_subplot(grid_sub_2[:, :])
        axes_2.set_xticks([])
        axes_2.set_yticks([])
        label = np.squeeze(ann[i], axis=2)
        label[label == 255] = np.max(label[label != 255]) +1
        axes_2.imshow(label, cmap='gray')
        axes_2.set_title('label')
    plt.show()

    print(ann.shape)


if __name__ == '__main__':
    tf.app.run()
