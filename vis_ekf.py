# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2020-01-30T07:03:19.397Z
# Description:
#
# ===============================================

# lib
import tensorflow as tf
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2

# user packages
from lib.utils import common
from dataset import generator
from lib.models import segModel

WIDTH=2048
HEIGHT=1024

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_dir',
                           os.path.join('da', 'logs', 'cityscapes', '2020-01-17_extra'),
                           'checkpoint directory')

tf.app.flags.DEFINE_multi_integer('mask_heights',
                                  [HEIGHT//4-10, HEIGHT//4+10],
                                  'height = (low_heihgt, high_height)')

tf.app.flags.DEFINE_multi_integer('mask_widths',
                                  [WIDTH//4-10, WIDTH//4+10],
                                  'width  = {low_width, high_width}')

def normalizeImg(img):
    return ((img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.float32)

MODEL_DICT = {'aspp': ['aspp'],
              'da'  : ['da'],
              'ppm' : ['ppm'],
              'ppa' : ['ppa'],
               'sppa': ['ppm', 'aspp', 'pam']}

def search_model():
    model_name = FLAGS.ckpt_dir.split('/')[0]

    return MODEL_DICT[model_name]


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    common.print_args()



    EKF_VIS_DIR = 'EKF_RESULTS'
    ekf_vis_dir = os.path.join(FLAGS.ckpt_dir, EKF_VIS_DIR)

    if not os.path.exists(ekf_vis_dir):
        os.makedirs(ekf_vis_dir)

    with tf.Graph().as_default():
        dataset = generator.Dataset(
            dataset_dir=os.path.join('dataset', 'cityscapes', 'tfrecord'),
            dataset_name='cityscapes',
            split_name=['val_fine'],
            batch_size=1,
            crop_size=[HEIGHT, WIDTH],
            is_training=False,
            model_variant='resnet_v1_101_beta',
            min_scale_factor=0.50,
            max_scale_factor=2.0,
            should_shuffle=False,
            should_repeat=False)

        ite = dataset.get_one_shot_iterator()
        sample = ite.get_next()
        images, labels = sample[common.IMAGE], sample[common.LABEL]

        module_order = search_model()

        model = segModel.SegModel(
            num_classes=dataset.num_classes,
            model_variant='resnet_v1_101_beta',
            output_stride=16,
            backbone_atrous_rates=[1, 2, 4],
            is_training=False,
            ppm_rates=[1, 2, 3, 6],
            module_order=module_order,
            decoder_output_stride=4)

        logits = model.build(images=images)
        logits = tf.image.resize_bilinear(logits, tf.shape(images)[1:3], align_corners=True)
        # logits = tf.nn.softmax(logits, axis=3)

        height_ind = tf.range(HEIGHT, dtype=tf.int32)
        width_ind = tf.range(WIDTH, dtype=tf.int32)
        height_ind = tf.expand_dims(tf.math.logical_and(tf.math.greater_equal(height_ind, FLAGS.mask_heights[0]),
                                                        tf.math.greater_equal(FLAGS.mask_heights[1], height_ind)), 1)

        width_ind = tf.expand_dims(tf.math.logical_and(tf.math.greater_equal(width_ind, FLAGS.mask_widths[0]),
                                                        tf.math.greater_equal(FLAGS.mask_widths[1], width_ind)), 0)
        # height_ind = tf.expand_dims(tf.math.equal(height_ind, HEIGHT//4), 1)
        # width_ind = tf.expand_dims(tf.math.equal(width_ind, WIDTH//4), 0)

        height_map = []
        width_map = []
        for w in range(WIDTH):
            height_map.append(height_ind)
        for h in range(HEIGHT):
            width_map.append(width_ind)
        height_map = tf.concat(height_map, axis=1)
        width_map = tf.concat(width_map, axis=0)

        height_map = tf.cast(height_map, tf.float32)
        width_map = tf.cast(width_map, tf.float32)
        mask = tf.expand_dims(tf.math.multiply(height_map, width_map), axis=2)

        m_concat = []
        for _ in range(dataset.num_classes):
            m_concat.append(mask)
        mask = tf.concat(m_concat, axis=2)
        masked_logits = tf.multiply(mask, logits)
        grad = tf.gradients(masked_logits, [images])

        ########### SESSION CREATING PROCESS #############
        checkpoints_iterator = tf.contrib.training.checkpoints_iterator(
            FLAGS.ckpt_dir)

        for checkpoint_path in checkpoints_iterator:
            restorer = tf.train.Saver()
            scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                         saver=restorer,
                                         ready_for_local_init_op=None)

            session_creator = tf.train.ChiefSessionCreator(
                scaffold=scaffold,
                master='',
                checkpoint_filename_with_path=checkpoint_path)

            with tf.train.MonitoredSession(
                    session_creator=session_creator, hooks=None) as sess:
                grad_list = []
                batch_num = 0
                while not sess.should_stop():
                    im, l, g = sess.run([images, logits, grad])
                    grad_list.append(g[0])
                    im = im[0]
                    g = np.abs(g[0][0])

                    if batch_num == 0:
                        sum_g = g
                    else:
                        sum_g += g
                    batch_num += 1

                    g = normalizeImg(g)
                    im = normalizeImg(im)

                    img_path = os.path.join(ekf_vis_dir, '{}_img.jpg'.format(batch_num))
                    grad_path = os.path.join(ekf_vis_dir, '{}_grad.jpg'.format(batch_num))

                    cv2.imwrite(img_path, cv2.cvtColor(im*255, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(grad_path, cv2.cvtColor(g*255, cv2.COLOR_RGB2BGR))

                    print('Processing {} done!'.format(batch_num))
                    if batch_num % 100 == 0:
                        print('100 second sleep')
                        time.sleep(100)
                    if batch_num >= 20:
                        break

            break

        sum_g = sum_g / batch_num
        print('max: {:.3f}, min: {:.3f}'.format(np.max(sum_g), np.min(sum_g)))
        sum_g = normalizeImg(sum_g)
        binary_g = sum_g.copy()
        binary_g[sum_g > np.mean(sum_g)] = 255

        cv2.imwrite(os.path.join(ekf_vis_dir, 'average_grad.jpg'), cv2.cvtColor(sum_g*255, cv2.COLOR_RGB2BGR))

        plt.title('average grad')
        plt.imshow(sum_g)
        plt.show()

        cv2.imwrite(os.path.join(ekf_vis_dir, 'sum_grad_{}.jpg'.format(batch_num)), cv2.cvtColor(binary_g.astype(np.uint8), cv2.COLOR_RGB2BGR))



if __name__ == '__main__':
    tf.app.run()
