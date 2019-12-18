# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================
"""
Show the segmentation results predicted by the traned model for the dataset.
Compared the correct labels, output the confusion matrix and IoU.
"""
# lib
import tensorflow as tf
import os
import sys
from tensorflow.python import math_ops
import numpy as np
import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.tools import inspect_checkpoint as ckpt
import io

# user packages
from lib.utils import common, colormap_utils
from dataset import generator
from lib.models import segModel

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

tf.app.flags.DEFINE_integer('max_number_of_iterations',
                            1,
                            'Maximum number of visualization iterations.')

tf.app.flags.DEFINE_boolean('save_labels',
                            False,
                            'Also save labels')

tf.app.flags.DEFINE_boolean('log_confusion',
                            True,
                            'Log the Confusion matrix to tensorboard.')

tf.app.flags.DEFINE_bool('add_flipped_images',
                         False,
                         'Add flipped images for prediction.')

tf.app.flags.DEFINE_multi_float('eval_scales',
                                [1.0],
                                'The scales to resize images for evaluation.')


_SEMANTIC_PREDICTION_DIR = 'semantic_prediction_result'


def calc_miou(confusion_matrix):
    """
    Calculate mean IoU(intersection over union) per image.
    """

    correct_pixels = np.sum(confusion_matrix, axis=1)
    predicted_pixels = np.sum(confusion_matrix, axis=0)
    num_classes = confusion_matrix.shape[0]
    iou_list = []
    for c in range(num_classes):
        # calculate IoU per class.
        gt = correct_pixels[c]
        pr = predicted_pixels[c]
        tp = confusion_matrix[c, c]

        if (pr+gt) == 0:
            # There are no predicted pixels and gt pixels.
            continue

        iou = tp / (pr+gt - tp)
        iou_list.append(iou)
        tf.logging.info('\t class {} iou: {:.4f}'.format(c, iou))
    miou = np.mean(iou_list)
    tf.logging.info('\tmean iou: {:.4f}'.format(miou))

    return miou

def _process_batch(sess, samples, predictions,
                   image_id_offset, save_dir,
                   conf_mat_op=None, conf_mat=None, batch_conf=None):
    """
    Do Visualizing process on batch.

    Args:
        sess            : TensorFlow session.
        samples         : samples created by iterator.
        predictions     : Predictions created by model.
        image_id_offset : Offset. This value is used when saving images and labels.
        save_dir        : Directory where images and labels are saved.
        conf_mat_op     : Confusion matrix operation.
        conf_mat        : Accumulated Confusion matrix.
        batch_conf      : Confusion matrix per batch.

    Return: Confusion matrix created by session, if confusion matrix is not None.
    """

    run_list = [
        samples[common.ORIGINAL_IMAGE],
        predictions,
        samples[common.IMAGE_FILENAME],
        samples[common.IMAGE_HEIGHT],
        samples[common.IMAGE_WIDTH]
    ]
    if FLAGS.save_labels:
        run_list.append(samples[common.LABEL])

    if FLAGS.log_confusion:
        run_list.append(conf_mat_op)
        run_list.append(batch_conf)
        run_list.append(conf_mat)

    result_list = sess.run(run_list)

    original_images = result_list[0]
    predictions = result_list[1]
    image_names = result_list[2]
    image_heights = result_list[3]
    image_widths = result_list[4]
    num_images = original_images.shape[0]
    for i in range(num_images):
        image_height = np.squeeze(image_heights[i])
        image_width  = np.squeeze(image_widths[i])
        original_image = np.squeeze(original_images[i])
        prediction = np.squeeze(predictions[i])
        crop_prediction = prediction[:image_height, :image_width]
        image_name = np.squeeze(image_names[i])

        if FLAGS.log_confusion:
            miou = calc_miou(result_list[-2])


        # Save image.
        image_path = os.path.join(save_dir, '{:06}_image'.format(image_id_offset+i))
        colormap_utils.save_annotation(original_image,
                        image_path,
                        add_colormap=False)

        # Save prediction.
        prediction_path = os.path.join(save_dir, '{:06}_pred_iou={:.4f}'.format(image_id_offset+i, miou))
        colormap_utils.save_annotation(crop_prediction,
                        prediction_path,
                        add_colormap=True,
                        colormap_type=FLAGS.dataset_name)

        if FLAGS.save_labels:
            # Save label.
            labels = result_list[5]
            label = np.squeeze(labels[i])
            crop_label = label[:image_height, :image_width]
            label_path = os.path.join(save_dir, '{:06}_label'.format(image_id_offset+i))
            colormap_utils.save_annotation(crop_label,
                            label_path,
                            add_colormap=True,
                            colormap_type=FLAGS.dataset_name)

    if conf_mat_op is not None:
        return result_list[-1]
    else:
        return None


def create_conf_mat_op(labels, logits, num_classes, ignore_label):
    """
    Create the confusion matrix operation.

    Args:
        labels       : Labels which are actually correct labels.
        logits       : Logits which are predicted by model.
        num_classes  : The number of class.
        ignore_label : Ignore label.

    Return: Confusion matrix operation.
    """
    zeros_like = tf.zeros_like(labels, dtype=tf.int32)
    mask = tf.equal(labels, ignore_label)
    not_ignore_mask = tf.cast(tf.not_equal(labels, ignore_label), dtype=tf.float32)
    labels = tf.where(mask, zeros_like, labels)
    # Compute a per-batch confusion
    batch_confusion = tf.confusion_matrix(tf.reshape(labels, [-1]),
                                          tf.reshape(logits, [-1]),
                                          num_classes=num_classes,
                                          weights=tf.reshape(not_ignore_mask, [-1]),
                                          name='batch_confusion_matrix')

    # Create an accumulator variable to hold the counts
    confusion = tf.Variable(tf.zeros([num_classes, num_classes],
                                     dtype=tf.int32),
                            name='confusion_matrix')
    # Create the update op for doing an accumulation on the batch
    confusion_update = confusion.assign(confusion + batch_confusion)

    return confusion_update, confusion, batch_confusion


def summary_conf_mat(conf_mat, logdir):
    """
    Convert the confusion matrix created by TensorFlow session using seaborn.
    Save the heatmap to tensorboard.
    For saving to tensorboard, create the new session. This session is different
    from the MonitoredSession.

    Args:
        conf_mat : Confusion matrix created by session. assuming that type is numpy 2D array.
        logdir   : Log directory.
    """
    miou = calc_miou(conf_mat)
    tf.logging.info('Mean IoU on all images: {:.4f}'.format(miou))
    # Creating the temporary graph for saving confusion matrix heatmap.
    temp_graph = tf.Graph()
    with temp_graph.as_default():
        norm_conf_mat = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)
        norm_conf_mat = np.around(norm_conf_mat, decimals=2)

        class_name = colormap_utils.get_class_name(FLAGS.dataset_name)

        conf_mat = pd.DataFrame(norm_conf_mat,
                                index=class_name,
                                columns=class_name)

        figure = plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
        ax = sns.heatmap(conf_mat, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout(True)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        save_path = os.path.join(logdir, 'confusion.png')
        plt.savefig(save_path, format='png')
        image_data = tf.gfile.GFile(save_path, 'rb').read()
        image = tf.image.decode_png(image_data, channels=4)
        image = tf.expand_dims(image, 0)

        file_writer = tf.summary.FileWriter(logdir=logdir)

        summary = tf.summary.image('Confusion_Matrix', image)

        sess = tf.Session(graph=temp_graph)
        summary = sess.run(summary)

        file_writer.add_summary(summary, global_step=0)
        sess.close()
        file_writer.close()


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    common.print_args()

    # Create dataset generator
    dataset = generator.Dataset(
                dataset_dir=FLAGS.dataset_dir,
                dataset_name=FLAGS.dataset_name,
                split_name=FLAGS.split_name,
                batch_size=1,
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

    tf.logging.info('Visualizing on %s set', ''.join(FLAGS.split_name))

    with tf.Graph().as_default():
        iterator = dataset.get_one_shot_iterator()
        samples = iterator.get_next()

        model = segModel.SegModel(
                num_classes=dataset.num_classes,
                model_variant=FLAGS.model_variant,
                output_stride=FLAGS.output_stride,
                backbone_atrous_rates=FLAGS.backbone_atrous_rates,
                is_training=False,
                ppm_rates=FLAGS.ppm_rates,
                ppm_pooling_type=FLAGS.ppm_pooling_type,
                atrous_rates=FLAGS.atrous_rates,
                module_order=FLAGS.module_order,
                decoder_output_stride=FLAGS.decoder_output_stride)

        if FLAGS.eval_scales == [1.0]:
            tf.logging.info('Evaluate the single scale image.')
            predictions = model.predict_labels(images=samples[common.IMAGE],
                                               add_flipped_images=FLAGS.add_flipped_images)
        else:
            tf.logging.info('Evaluate the multi-scale image.')
            predictions = model.predict_labels_for_multiscale(
                                    images=samples[common.IMAGE],
                                    add_flipped_images=FLAGS.add_flipped_images,
                                    eval_scales=FLAGS.eval_scales)

        checkpoints_iterator = tf.contrib.training.checkpoints_iterator(
            FLAGS.checkpoint_dir, min_interval_secs=FLAGS.eval_interval_secs)

        if FLAGS.log_confusion:
            # Get the confusion matrix op.
            conf_mat_op, conf_mat, batch_conf = create_conf_mat_op(samples[common.LABEL],
                                             predictions,
                                             dataset.num_classes,
                                             dataset.ignore_label)
            # Initializer for initializing the conf_mat.
            init_fn = tf.initialize_variables([conf_mat])
            # restore from checkpoint excluding variable "conf_mat".
            variables_to_restore = tf.global_variables()[:-1]
        else:
            # dummy op.
            conf_mat_op, conf_mat, batch_conf = None, None, None
            init_fn = None
            variables_to_restore = tf.global_variables()

        num_iteration = 0
        max_num_iteration = FLAGS.max_number_of_iterations

        for checkpoint_path in checkpoints_iterator:
            num_iteration += 1
            tf.logging.info(
                'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                             time.gmtime()))
            tf.logging.info('Visualizing with model %s', checkpoint_path)
            restorer = tf.train.Saver(variables_to_restore)
            scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                         saver=restorer,
                                         ready_for_local_init_op=init_fn)
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
                    accumulated_conf_mat = _process_batch(sess=sess,
                                samples=samples,
                                predictions=predictions,
                                image_id_offset=image_id_offset,
                                save_dir=save_dir,
                                conf_mat_op=conf_mat_op,
                                conf_mat=conf_mat,
                                batch_conf=batch_conf)
                    image_id_offset += 1
                    batch += 1



            if FLAGS.log_confusion:
                summary_conf_mat(accumulated_conf_mat, FLAGS.vis_logdir)

            tf.logging.info(
                'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                             time.gmtime()))

            if max_num_iteration > 0 and num_iteration >= max_num_iteration:
                break

if __name__ == '__main__':
    tf.app.run()
