# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

# lib
import tensorflow as tf
import os
import sys
from tensorflow.python import math_ops
from PIL import Image
import numpy as np
import time

# user packages
from lib.utils import common
from dataset import generator
from lib.models import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('vis_logdir',
                           None,
                           'Where to write the event logs.')

tf.app.flags.DEFINE_string('checkpoint_dir',
                           None,
                           'Directory of model checkpoints.')

# Settings for visualizing the model.

tf.app.flags.DEFINE_integer('eval_interval_secs',
                            200,
                            'How often to run evaluation.')

tf.app.flags.DEFINE_boolean('save_raw_predictions',
                            False,
                            'Also save raw predictions.')

tf.app.flags.DEFINE_integer('max_number_of_iterations',
                            1,
                            'Maximum number of visualization iterations.')

tf.app.flags.DEFINE_boolean('save_labels',
                            False,
                            'Also save labels')

_SEMANTIC_PREDICTION_DIR = 'semantic_prediction_result'
_RAW_PREDICTION_DIR = 'raw_prediction_result'


def create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark."""
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap


def create_camvid_label_colormap():
    """Creates a label colormap used in CamVid dataset."""
    colormap = np.zeros((12, 3), dtype=np.uint8)
    colormap[0] = [128, 128, 128] # sky
    colormap[1] = [128, 0, 0]     # building
    colormap[2] = [192, 192, 128] # column_pole
    colormap[3] = [128, 64, 128]  # road
    colormap[4] = [0, 0, 192]     # sidewalk
    colormap[5] = [128, 128, 0]   # Tree
    colormap[6] = [192, 128, 128] # SignSymbol
    colormap[7] = [64, 64, 128]   # Fence
    colormap[8] = [64, 0, 128]    # Car
    colormap[9] = [64, 64, 0]     # Pedestrian
    colormap[10] = [0, 128, 128]  # Bicyclist
    colormap[11] = [0, 0, 0]      # Void(undefined) => ignore label.

    class_vs_colormap = {0: 'sky', 1: 'building', 2: 'column_pole', 3: 'road',
                         4: 'sidewalk', 5: 'tree', 6: 'sign', 7: 'fence', 8: 'car',
                         9: 'pedestrian', 10: 'byciclist', 11: 'void (undefined)'}

    return colormap, class_vs_colormap


def label_to_color_image(label, colormap_type):
    """
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label. Got {}.'.format(label.shape))

    if colormap_type == 'camvid':
        colormap, _ = create_camvid_label_colormap()
    elif colormap_type =='cityscapes':
        colormap = create_cityscapes_label_colormap()
    else:
        raise ValueError('colormap_type {} is not defined'.format(colormap_type))
    return colormap[label]


def save_annotation(label,
                    save_dir,
                    filename,
                    add_colormap=True,
                    normalize_to_unit_values=False,
                    scale_values=False,
                    colormap_type=None):

    # Add colormap for visulizing the prediction.
    if add_colormap:
        colored_label = label_to_color_image(label, colormap_type)
    else:
        colored_label = label
        if normalize_to_unit_values:
            min_value = np.amin(colored_label)
            max_value = np.amax(colored_label)
            range_value = max_value - min_value
            if range_value != 0:
                colored_label = (colored_label - min_value) / range_value

        if scale_values:
            colored_label = 255 * colored_label

    pil_image = Image.fromarray(colored_label.astype(dtype=np.uint8))
    with tf.gfile.Open('%s/%s.png' %(save_dir, filename), mode='w') as f:
        pil_image.save(f, 'PNG')



def _process_batch(sess, original_images, semantic_predictions, image_names,
                   image_heights, image_widths, image_id_offset, save_dir,
                   raw_save_dir, train_id_to_eval_id=None, labels=None):

    run_list = [original_images,
                semantic_predictions,
                image_names,
                image_heights,
                image_widths]

    if FLAGS.save_labels:
        run_list.append(labels)

    result_list = sess.run(run_list)
    # (original_images,
    #  semantic_predictions,
    #  image_names,
    #  image_heights,
    #  image_widths,
    #  labels) = sess.run([original_images, semantic_predictions,
    #                            image_names, image_heights, image_widths, labels])

    original_images = result_list[0]
    semantic_predictions = result_list[1]
    image_names = result_list[2]
    image_heights = result_list[3]
    image_widths = result_list[4]
    num_images = original_images.shape[0]
    for i in range(num_images):
        image_height = np.squeeze(image_heights[i])
        image_width  = np.squeeze(image_widths[i])
        original_image = np.squeeze(original_images[i])
        semantic_prediction = np.squeeze(semantic_predictions[i])
        crop_semantic_prediction = semantic_prediction[:image_height, :image_width]
        image_name = np.squeeze(image_names[i])

        # Save image.
        save_annotation(original_image,
                        save_dir,
                        filename='{:06}_image'.format(image_id_offset+i),
                        add_colormap=False)

        save_annotation(crop_semantic_prediction,
                        save_dir,
                        filename='{:06}_pred'.format(image_id_offset+i),
                        add_colormap=True,
                        colormap_type=FLAGS.dataset_name)

        if FLAGS.save_labels:
            labels = result_list[5]
            label = np.squeeze(labels[i])
            save_annotation(label,
                            save_dir,
                            filename='{:06}_label'.format(image_id_offset+i),
                            add_colormap=True,
                            colormap_type=FLAGS.dataset_name)



def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    common.print_args()

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
                is_training=False,
                model_variant=FLAGS.model_variant,
                should_shuffle=False,
                should_repeat=False)


    tf.gfile.MakeDirs(FLAGS.vis_logdir)
    save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_DIR)
    tf.gfile.MakeDirs(save_dir)
    raw_save_dir = os.path.join(FLAGS.vis_logdir, _RAW_PREDICTION_DIR)
    tf.gfile.MakeDirs(raw_save_dir)

    tf.logging.info('Visualizing on %s set', FLAGS.split_name)

    with tf.Graph().as_default():
        iterator = dataset.get_one_shot_iterator()
        samples = iterator.get_next()

        predictions = model.predict_labels(
                samples[common.IMAGE],
                num_classes=dataset.num_classes,
                model_variant=FLAGS.model_variant,
                output_stride=FLAGS.output_stride,
                backbone_atrous_rates=FLAGS.backbone_atrous_rates,
                ppm_rates=FLAGS.ppm_rates,
                decoder_output_stride=FLAGS.decoder_output_stride,
                atrous_rates=FLAGS.atrous_rates)

        checkpoints_iterator = tf.contrib.training.checkpoints_iterator(
            FLAGS.checkpoint_dir, min_interval_secs=FLAGS.eval_interval_secs)

        num_iteration = 0
        max_num_iteration = FLAGS.max_number_of_iterations

        for checkpoint_path in checkpoints_iterator:
            num_iteration += 1
            tf.logging.info(
                'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                             time.gmtime()))
            tf.logging.info('Visualizing with model %s', checkpoint_path)

            scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
            session_creator = tf.train.ChiefSessionCreator(
                scaffold=scaffold,
                master='',
                checkpoint_filename_with_path=checkpoint_path)
            with tf.train.MonitoredSession(
                    session_creator=session_creator, hooks=None) as sess:
                batch = 0
                image_id_offset = 0

                while not sess.should_stop():
                    tf.logging.info('Visualizing batch %d', batch + 1)
                    _process_batch(sess=sess,
                                   original_images=samples[common.ORIGINAL_IMAGE],
                                   semantic_predictions=predictions,
                                   image_names=samples[common.IMAGE_FILENAME],
                                   image_heights=samples[common.IMAGE_HEIGHT],
                                   image_widths=samples[common.IMAGE_WIDTH],
                                   labels=samples[common.LABEL],
                                   image_id_offset=image_id_offset,
                                   save_dir=save_dir,
                                   raw_save_dir=raw_save_dir)
                    image_id_offset += FLAGS.batch_size
                    batch += 1

            tf.logging.info(
                'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                             time.gmtime()))

            if max_num_iteration > 0 and num_iteration >= max_num_iteration:
                break

if __name__ == '__main__':
    tf.app.run()
