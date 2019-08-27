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
import sys
from tensorflow.python import math_ops

# user packages
from dataset import generator
from lib.models import model
from lib.utils import common

FLAGS = tf.app.flags.FLAGS

# DEFINE FLAGS
tf.app.flags.DEFINE_float('base_learning_rate',
                          0.01,
                          'Initial learning rate.')

tf.app.flags.DEFINE_float('weight_decay',
                          0.00004,
                          'The rate of the weight decay for training.')

tf.app.flags.DEFINE_boolean('fine_tune_batch_norm',
                            True,
                            'Whether finetuning batch normalization value.')

tf.app.flags.DEFINE_float('momentum',
                          0.9,
                          'Momentum.')

tf.app.flags.DEFINE_string('train_logdir',
                           './train_logdir',
                           'Log directory for tensorboard and save trained checkpoint to the place.')

# This is for fine-tuning models.
tf.app.flags.DEFINE_string('tf_initial_checkpoint',
                           None,
                           'The initial checkpoint for backbone network.')

tf.app.flags.DEFINE_boolean('save_summaries_images',
                            False,
                            'whether save images to summaries or not.')

tf.app.flags.DEFINE_integer('train_steps',
                            50000,
                            'Training steps.')

tf.app.flags.DEFINE_integer('log_steps',
                            10,
                            'Display logging information at every log_steps.')

tf.app.flags.DEFINE_integer('save_interval_secs',
                            600,
                            'How often, in seconds, we save the model to disk.')

tf.app.flags.DEFINE_integer('save_summaries_secs',
                            200,
                            'How often, in seconds, we compute the summaries.')

tf.app.flags.DEFINE_integer('batch_size',
                            8,
                            'batch size')

tf.app.flags.DEFINE_list('crop_size',
                         '513, 513',
                         'Image crop size [height, width]')




def add_softmax_cross_entropy_loss(logits, labels, num_classes, ignore_label, loss_weight=1.0, resize_logits=True):
    """ """
    if labels is None:
        raise ValueError('No label for softmax cross entropy.')

    _, labels_height, labels_width, _ = labels.get_shape().as_list()
    _, logits_height, logits_width, _ = logits.get_shape().as_list()
    if logits_height != labels_height or logits_width != labels_width:
        if resize_logits:
            logits = tf.image.resize_bilinear(logits, [labels_height, labels_width], align_corners=True)
        else:
            raise ValueError('logits shape is not equal to labels. ' +
                             'logits.shape => {}, labels.shape=> {}'.format((logits_height, logits_width),
                                                                            (labels_height, labels_width)))

    labels = tf.reshape(labels, shape=[-1])
    not_ignore_mask = tf.cast(tf.not_equal(labels, ignore_label), dtype=tf.float32) * loss_weight

    one_hot_labels = tf.one_hot(
        labels, num_classes, on_value=1.0, off_value=0.0)

    # Conpute the loss for all pixels.
    tf.losses.softmax_cross_entropy(
        one_hot_labels,
        tf.reshape(logits, shape=[-1, num_classes]),
        weights=not_ignore_mask,
        scope='Loss')


def log_summaries(input, labels, num_classes, logits):
    """ Logs the summaries for the model.

    Args:
        input: Input image of the model. Its shape is [batch_size, height, width, channel].
        label: Label of the image. Its shape is [batch_size, height, width].
        num_classes: The number of classes of the dataset.
        output: Output of the model. Its shape is [batch_size, height, width].
    """
    # Add summaries for model variables.
    # model variables are variables created by a slim.conv2d and so on.
    # This variables are trained or fine-tuned during learning and are loaded
    # from a checkpoint during evaluation or inference.
    for model_var in tf.model_variables():
        tf.summary.histogram(model_var.op.name, model_var)

    # Add summaries for images, labels, semantic predictions.
    if FLAGS.save_summaries_images:
        tf.summary.image('samples/image', input)

        pixel_scaling = max(1, 255 // num_classes)
        summary_labels = tf.cast(labels * pixel_scaling, tf.uint8)
        tf.summary.image('samples/label', summary_labels)

        logits = tf.argmax(logits, 3)
        print('logits.shape: {}'.format(logits.get_shape().as_list()))
        predictions = tf.expand_dims(logits, axis=3)
        print('predictions.shape: {}'.format(predictions.get_shape().as_list()))
        summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
        tf.summary.image('sample/logits', summary_predictions)



def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      ignore_missing_vars=False):
    """ Gets the function initializing model variables from a checkpoint.

    Args:
        train_logdir: Log directory for training.
        tf_initial_checkpoint: TensorFlow checkpoint for initialization.
        ignore_missing_vars: Ignore missing variables in the checkpoint.

    Returns:
        Initialization function.
    """
    if tf_initial_checkpoint is None:
        tf.logging.info('Not initializing the model from a checkpoint.')
        return None

    if tf.train.latest_checkpoint(train_logdir):
        tf.logging.info('Ignoring pre-trained initialization; other trained checkpoint exists.')
        return None

    tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['global_step'])

    if variables_to_restore:
        init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
            tf_initial_checkpoint,
            variables_to_restore,
            ignore_missing_vars=ignore_missing_vars)
        global_step = tf.train.get_or_create_global_step()

        def restore_fn(unused_scaffold, sess):
            sess.run(init_op, init_feed_dict)
            sess.run([global_step])

        return restore_fn

    return None

def main(argv):

    common.print_args()

    print('Training on %s %s set' % (FLAGS.split_name, FLAGS.dataset_name))

    graph = tf.Graph()
    crop_size = [int(val) for val in FLAGS.crop_size]
    # create graph in new thread.
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
                            decay_steps=FLAGS.train_steps,
                            end_learning_rate=0,
                            power=0.9)
        tf.summary.scalar('learning_rate', learning_rate)

        # create optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)

        # build models
        with tf.name_scope('clone') as scope:
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                samples = iterator.get_next()
                input = tf.identity(samples[common.IMAGE], name='Input_Image')
                labels = tf.identity(samples[common.LABEL], name='Semantic_Label')

                is_training=True
                logits = model.build_model(images=input,
                                           num_classes=dataset.num_classes,
                                           model_variant=FLAGS.model_variant,
                                           output_stride=FLAGS.output_stride,
                                           weight_decay=FLAGS.weight_decay,
                                           backbone_atrous_rates=FLAGS.backbone_atrous_rates,
                                           fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
                                           is_training=True,
                                           ppm_rates=FLAGS.ppm_rates,
                                           ppm_pooling_type=FLAGS.ppm_pooling_type,
                                           decoder_output_stride=FLAGS.decoder_output_stride,
                                           atrous_rates=FLAGS.atrous_rates)
                print('logits.shape: {}'.format(logits.get_shape()))

                add_softmax_cross_entropy_loss(
                    logits,
                    labels,
                    dataset.num_classes,
                    dataset.ignore_label,
                    loss_weight=1.0,
                )
                logits = tf.identity(logits, name='dense_prediction')
                log_summaries(input, labels, dataset.num_classes, logits)

            # should_log
            should_log = math_ops.equal(math_ops.mod(global_step, FLAGS.log_steps), 0)

            losses = tf.losses.get_losses(scope=scope)

            print_losses = []
            for loss in losses:
                tf.summary.scalar('Losses:%s' % loss.op.name, loss)
                # print_ops.append(tf.cond(
                #     should_log,
                #     lambda: tf.print('raw loss is:',
                #                      loss,
                #                      output_stream=sys.stdout),
                #     lambda: False))

                print_losses.append(tf.cond(
                    should_log,
                    lambda: tf.Print(loss, [loss], 'raw loss is:'),
                    lambda: loss))

            regularization_loss = tf.losses.get_regularization_loss(scope=scope)
            tf.summary.scalar('Losses/%s' % regularization_loss.op.name,
                              regularization_loss)
            # print_ops.append(tf.cond(
            #     should_log,
            #     lambda: tf.print('regularization loss is:',
            #                      regularization_loss,
            #                      output_stream=sys.stdout),
            #     lambda: False))
            regularization_loss = tf.cond(
                should_log,
                lambda: tf.Print(regularization_loss, [regularization_loss], 'regularization loss is:'),
                lambda: regularization_loss)

            total_loss = tf.add_n([tf.add_n(losses), regularization_loss])
            tf.summary.scalar('Losses/total_loss', total_loss)

            grads = optimizer.compute_gradients(total_loss)

            grad_updates = optimizer.apply_gradients(
                grads, global_step=global_step)

            # Gather update_ops. These contain, for example,
            # the updates for the batch_norm variables created by model_fn.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            # Print total loss to the terminal.
            # This implementation is mirrored from tf.slim.summaries.
            total_loss = tf.cond(
                should_log,
                lambda: tf.Print(total_loss, [total_loss], 'Total loss is:'),
                lambda: total_loss)
            # print_ops.append(tf.cond(
            #     should_log,
            #     lambda: tf.print('total loss is:',
            #                      total_loss,
            #                      output_stream=sys.stdout),
            #     lambda: False))
            global_step = tf.cond(
                should_log,
                lambda: tf.Print(global_step, [global_step], 'global_step is:'),
                lambda: global_step)

            with tf.control_dependencies([update_op] + [global_step] + print_losses + [regularization_loss]):
                train_tensor = tf.identity(total_loss, name='train_op')

        summary_op = tf.summary.merge_all(scope='clone')

        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)

        init_fn = None
        if FLAGS.tf_initial_checkpoint:
            init_fn = get_model_init_fn(
                train_logdir=FLAGS.train_logdir,
                tf_initial_checkpoint=FLAGS.tf_initial_checkpoint,
                ignore_missing_vars=True)

        scaffold = tf.train.Scaffold(
            init_fn=init_fn,
            summary_op=summary_op)

        stop_hook = tf.train.StopAtStepHook(last_step=FLAGS.train_steps)

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_logdir,
            hooks=[stop_hook],
            # config=session_config,
            scaffold=scaffold,
            summary_dir=FLAGS.train_logdir,
            log_step_count_steps=FLAGS.log_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_checkpoint_secs=FLAGS.save_interval_secs
        ) as sess:
            while not sess.should_stop():
                sess.run([train_tensor])




    # with tf.Session(graph=graph) as sess:
    #     writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
    #     writer_op = tf.summary.merge_all()
    #         # loss = _calculate_loss(logits, loss)
    #         # train_op =





if __name__ == '__main__':
    tf.app.run()
