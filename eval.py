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

# user packages
from dataset import generator
from lib.utils import common
from lib.models import segModel

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_logdir',
                           None,
                           'Where to write the event logs.')

tf.app.flags.DEFINE_string('checkpoint_dir',
                           None,
                           'Directory of model checkpoints.')

tf.app.flags.DEFINE_integer('eval_interval_secs',
                            200,
                            'How often to run evaluation.')

tf.app.flags.DEFINE_integer('max_number_of_evaluations',
                            1,
                            'Maximum number of eval iterations.')


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

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

    tf.logging.info('Evaluation on %s set on %s', ''.join(FLAGS.split_name), FLAGS.dataset_name)

    with tf.Graph().as_default():
        samples = dataset.get_one_shot_iterator().get_next()

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

        predictions = model.predict_labels(images=samples[common.IMAGE])
        predictions = tf.reshape(predictions, shape=[-1])
        labels = tf.reshape(samples[common.LABEL], shape=[-1])
        weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))


        # Set ignore_label regions to label 0, because metrics.mean_iou requires
        # range of labels = [0, dataset.num_classes). Note the ignore_label regions
        # are not evaluated since the corresponding regions contain weights = 0.
        labels = tf.where(
            tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

        miou, update_op = tf.metrics.mean_iou(
            predictions, labels, dataset.num_classes, weights=weights)
        tf.summary.scalar('%s_miou' % ''.join(FLAGS.split_name), miou)

        summary_op = tf.summary.merge_all()
        # それぞれのepochごとに行いたい処理
        summary_hook = tf.contrib.training.SummaryAtEndHook(
            log_dir=FLAGS.eval_logdir, summary_op=summary_op)
        hooks = [summary_hook]

        num_eval = None
        if FLAGS.max_number_of_evaluations > 0:
            num_eval = FLAGS.max_number_of_evaluations

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
        tf.contrib.training.evaluate_repeatedly(
            master='',
            checkpoint_dir=FLAGS.checkpoint_dir,
            eval_ops=[update_op],
            max_number_of_evaluations=num_eval,
            hooks=hooks,
            eval_interval_secs=FLAGS.eval_interval_secs)



if __name__ == '__main__':
    tf.app.flags.mark_flag_as_required('checkpoint_dir')
    tf.app.flags.mark_flag_as_required('eval_logdir')
    tf.app.run()
