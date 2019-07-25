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

    plt.figure(figsize=(20, 10))
    num_imgs = 2
    for i in range(num_imgs):
        plt.suptitle(res['image_name'][i])
        plt.subplot(num_imgs, 2, (i*2)+1)
        plt.title('target')
        plt.imshow(image[i].astype(np.uint8))
        plt.subplot(num_imgs, 2, (i*2)+2)
        plt.title('label')
        label = np.squeeze(ann[i], axis=2)
        label[label == 255] = np.max(label[label != 255]) +1
        plt.imshow(label, cmap='gray')
    plt.show()

    print(ann.shape)


if __name__ == '__main__':
    tf.app.run()
