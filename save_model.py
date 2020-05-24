"""Export trained model to tensorfow frozen graph. """
import os
import tensorflow as tf

from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.python.tools import freeze_graph
from lib.utils import common, input_preprocess
from lib.models import segModel
from dataset import generator

slim = tf.contrib.slim
flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path')

flags.DEFINE_string('export_path', None,
                    "Path to output tensorflow frozen graph.")

flags.DEFINE_multi_float('inference_scales', [1.0],
                         'The scales to resize images for inference.')

flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped image during inference or not.')

flags.DEFINE_bool('save_inference_graph', False,
                   'Save inference graph in text proto.')


# Input name of the exported model.
_INPUT_NAME = "ImageTensor"

# Output name of the exported model.
_OUTPUT_NAME = 'SemanticPrediction'
_RAW_OUTPUT_NAME = 'RawSemanticPredictions'

# Output name of the exported probabilities.
_OUTPUT_PROB_NAME = 'SemanticProbabilities'
_RAW_OUTPUT_PROB_NAME = 'RawSemanticProbabilities'

def _create_input_tensor():
    """Create and prepares input tensors for model.

    This method creates a 4-D uint8 image tensor 'ImageTensor' with shape
    [1, None, None, 3]. The actual input tensor name to use during inference is
    'ImageTensor:0'.

    Returns:
        image: Preprocessed 4-D float32 tensorwith shape [1, crop_height,
        crop_width, 3].
        original_image_size: Original image shape tensor [height, width].
        resized_image_size: Resized image shape tensor [height, width].
    """
    # Input_preprocess takes 4-D image tensor as input.
    input_image = tf.placeholder(tf.uint8, [1, None, None, 3], name=_INPUT_NAME)
    original_image_size = tf.shape(input_image)[1:3]

    # Squeeze the dimension in axis=0 sizne 'preprocess_image_and_label' assumes
    # image to be 3-D.
    image= tf.squeeze(input_image, axis=0)
    resized_image, image, _ = input_preprocess.preprocess_image_and_label(
        image,
        label=None,
        crop_height=int(FLAGS.crop_size[0]),
        crop_width =int(FLAGS.crop_size[1]),
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        is_training=False,
        model_variant=FLAGS.model_variant)
    resized_image_size = tf.shape(resized_image)[:2]

    # Expand the dimension in axis=0, since the following operations assume the
    # image to be 4-D.
    image = tf.expand_dims(image, 0)

    return image, original_image_size, resized_image_size


def main(unused_args):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Prepare to export model to %s', FLAGS.export_path)
    common.print_args()



    with tf.Graph().as_default():
        image, image_size, resized_image_size = _create_input_tensor()

        model = segModel.SegModel(
            num_classes=generator._DATASETS_INFORMATION[FLAGS.dataset_name]['num_classes'],
            model_variant=FLAGS.model_variant,
            output_stride=FLAGS.output_stride,
            fine_tune_batch_norm=False,
            backbone_atrous_rates=FLAGS.backbone_atrous_rates,
            is_training=False,
            ppm_rates=FLAGS.ppm_rates,
            ppm_pooling_type=FLAGS.ppm_pooling_type,
            atrous_rates=FLAGS.atrous_rates,
            module_order=FLAGS.module_order,
            decoder_output_stride=FLAGS.decoder_output_stride)

        if FLAGS.inference_scales == [1.0]:
            tf.logging.info('Evaluate the single scale image.')
            predictions, probabilities = model.predict_labels(images=image,
                                               add_flipped_images=FLAGS.add_flipped_images)
        else:
            tf.logging.info('Evaluate the multi-scale image.')
            predictions, probabilities = model.predict_labels_for_multiscale(
                                    images=image,
                                    add_flipped_images=FLAGS.add_flipped_images,
                                    eval_scales=FLAGS.inference_scales)
        # prediction is a thing after argmax.
        raw_predictions = tf.identity(
            tf.cast(predictions, tf.float32),
            _RAW_OUTPUT_NAME)
        # probabilities is a thing berore argmax.
        raw_probabilities = tf.identity(
            probabilities,
            _RAW_OUTPUT_PROB_NAME)


        # Crop the valid regions from the predictions.
        semantic_predictions = raw_predictions[
            :, :resized_image_size[0], :resized_image_size[1]]
        semantic_probabilities = raw_probabilities[
            :, :resized_image_size[0], :resized_image_size[1]]


        # Resize back the prediction to the original image size.
        def _resize_label(label, label_size):
            # Expand dimension of label to [1, height, width, 1] for resize operation.
            label = tf.expand_dims(label, 3)
            resized_label = tf.image.resize_images(
                label,
                label_size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                align_corners=True)
            return tf.cast(tf.squeeze(resized_label, 3), tf.int32)
        semantic_predictions = _resize_label(semantic_predictions, image_size)
        semantic_predictions = tf.identity(semantic_predictions, name=_OUTPUT_NAME)

        semantic_probabilities = tf.image.resize_bilinear(
            semantic_probabilities, image_size, align_corners=True,
            name=_OUTPUT_PROB_NAME)

        saver = tf.train.Saver(tf.all_variables())

        dirname = os.path.dirname(FLAGS.export_path)
        tf.gfile.MakeDirs(dirname)
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        freeze_graph.freeze_graph_with_def_protos(
            graph_def,
            saver.as_saver_def(),
            FLAGS.checkpoint_path,
            _OUTPUT_NAME + ',' + _OUTPUT_PROB_NAME,
            restore_op_name=None,
            filename_tensor_name=None,
            output_graph=FLAGS.export_path,
            clear_devices=True,
            initializer_nodes=None
            )

        if FLAGS.save_inference_graph:
            tf.train.write_graph(graph_def, dirname, 'inference_graph.pbtxt')


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_path')
    flags.mark_flag_as_required('export_path')
    tf.app.run()
