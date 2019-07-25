# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description: Dataset Generator
#
# ===============================================
"""Wrapper for providing semantic segmentation data.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow.
"""

# packages
import tensorflow as tf
import collections
import os
import sys
add_path = os.path.realpath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(add_path)


# user packages
from lib.utils import input_preprocess


FLAGS = tf.app.flags.FLAGS

# DEFINE DATASET FLAGS
tf.app.flags.DEFINE_string('dataset_dir',
                           os.path.join('dataset', 'CamVid', 'tfrecord'),
                           'tfrecord directory')

tf.app.flags.DEFINE_string('split_name',
                           'train',
                           'dataset name')

tf.app.flags.DEFINE_string('dataset_name',
                           'camvid',
                           'dataset name')

tf.app.flags.DEFINE_float('min_scale_factor',
                           0.5,
                           'Minimum scale factor for data augmentation.')

tf.app.flags.DEFINE_float('max_scale_factor',
                           1.5,
                           'Maximum scale factor step size for data augmentation.')

tf.app.flags.DEFINE_float('scale_factor_step_size',
                           0.25,
                           'Scale factor step size for data augmentation.')

tf.app.flags.DEFINE_integer('min_resize_value',
                            None,
                            'Desired size of the smaller image side.')

tf.app.flags.DEFINE_integer('max_resize_value',
                            None,
                            'Maximum allowed size of the larger image side.')

tf.app.flags.DEFINE_integer('resize_factor',
                            None,
                            'Resized dimensions are multiple of factor plus one.')

tf.app.flags.DEFINE_integer('batch_size',
                            8,
                            'batch size')

tf.app.flags.DEFINE_list('crop_size',
                         '513, 513',
                         'Image crop size [height, width]')

tf.app.flags.DEFINE_string('model_variant',
                           'resnet_v1_50',
                           'model variant.')


_DATASETS_INFORMATION = {
    'cityscapes': {},
    'camvid'    : {'num_classes': 12,
                   'ignore_label': 255},
    'pascal_voc': {},
    'playing_for_data': {},
    'kitti'     : {}
}



class Dataset(object):
    """Represents input dataset."""

    def __init__(self,
                 dataset_dir,
                 dataset_name,
                 split_name,
                 batch_size,
                 crop_size,
                 min_resize_value=None,
                 max_resize_value=None,
                 resize_factor=None,
                 min_scale_factor=1.,
                 max_scale_factor=1.,
                 scale_factor_step_size=0,
                 model_variant=None,
                 num_readers=1,
                 is_training=False,
                 should_shuffle=False,
                 should_repeat=False):
        """Initializes the dataset.

        Args:
            dataset_dir   : The directory of the dataset sources.
            dataset_name  : Dataset name.
            batch_size    : Batch size.
            crop_size     : The size used to crop the image and label.
            num_reader    : Number of readers for data provider.
            model_variant : Model Variant (string) for choosing
                            how to mean-subtract the images. See feature_extractor.network_map
                            for supported model variants.
            is_training   : Boolean, if dataset if for training or not.
            num_readers   : Number of readers for data provider.
            should_repeat : Boolean, if should repeat the input data.
            should_shuffle: Boolean, if should shuffle the input data.

        Raises:
            V
        """
        if dataset_name not in _DATASETS_INFORMATION.keys():
            raise ValueError('The specified dataset is not supported yet.')
        self.dataset_name = dataset_name

        if model_variant is None:
            tf.logging.warning('Please specify a model_variant. '
                               'See feature_extractor.network_map for supported model'
                                'variants.')

        self.dataset_dir = dataset_dir
        self.split_name = split_name
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.resize_factor = resize_factor
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_factor_step_size = scale_factor_step_size
        self.model_variant = model_variant
        self.num_readers = num_readers
        self.is_training = is_training
        self.should_shuffle = should_shuffle
        self.should_repeat = should_repeat

        self.num_classes = _DATASETS_INFORMATION[self.dataset_name]['num_classes']
        self.ignore_label = _DATASETS_INFORMATION[self.dataset_name]['ignore_label']


    def _parse_function(self, example_proto):
        """Function to parse the example proto.

        Args:
            example_proto: Proto in the format of tf.Example.

        Raises:
            ValueError: Label is of wrong shape.
        """
        features = {
            'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/encoded' : tf.FixedLenFeature((), tf.string, default_value=''),
            'image/height'  : tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/width'   : tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/channels': tf.FixedLenFeature((), tf.int64, default_value=0),
            'label/encoded' : tf.FixedLenFeature((), tf.string, default_value='')
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        for key in parsed_features:
            print('{}: {}'.format(key, parsed_features[key]))

        # decode image and label.
        image = tf.image.decode_png(parsed_features['image/encoded'], 3)
        label = tf.image.decode_png(parsed_features['label/encoded'], 1)

        image_name = parsed_features['image/filename']
        if image_name is None:
            image_name = tf.constant('')

        sample = {
            'image'     : image,
            'image_name': image_name,
            'height'    : parsed_features['image/height'],
            'width'     : parsed_features['image/width']
        }
        # make sure label size
        if label is not None:
            if label.get_shape().ndims == 2:
                label = tf.expand_dims(label, 2)
            elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
                pass
            else:
                raise ValueError('Input label shape must be [height, width], or [height, width, 1].')

        label.set_shape([None, None, 1])
        sample['label'] = label

        return sample


    def _preprocess_image(self, sample):
        """Preprocesses the image and label.

        Args:
            sample: A sample containing image and label.

        Returns:
            sample: Sample with preprocessed image and label.

        Raises:
            valueError: Ground truth label not provided during training.
        """
        image = sample['image']
        label = sample['label']

        original_image, image, label = input_preprocess.preprocess_image_and_label(
            image=image,
            label=label,
            crop_height=self.crop_size[0],
            crop_width=self.crop_size[1],
            min_resize_value=self.min_resize_value,
            max_resize_value=self.max_resize_value,
            resize_factor=self.resize_factor,
            min_scale_factor=self.min_scale_factor,
            max_scale_factor=self.max_scale_factor,
            scale_factor_step_size=self.scale_factor_step_size,
            ignore_label=self.ignore_label,
            is_training=self.is_training,
            model_variant=self.model_variant
        )

        sample['image'] = image

        if not self.is_training:
            # Original image is only used during visualization.
            sample['original_image'] = original_image

        if label is not None:
            sample['label'] = label

        return sample


    def get_one_shot_iterator(self):
        """Gets an iterator that iterates across the dataset once.

        Returns:
            An iterator of type tf.data.Iterator.
        """
        files = self._get_all_files()

        dataset = (
            tf.data.TFRecordDataset(files, num_parallel_reads=self.num_readers)
            .map(self._parse_function, num_parallel_calls=self.num_readers)
            .map(self._preprocess_image, num_parallel_calls=self.num_readers)
        )

        if self.should_shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        if self.should_repeat:
            dataset = dataset.repeat() # Repeat forever for training.
        else:
            dataest = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
        return dataset.make_one_shot_iterator()


    def _get_all_files(self):
        """Gets all the files to read data from.

        Returns:
            A list of input files.
        """
        file_pattern = '%s-*'
        file_pattern = os.path.join(self.dataset_dir,
                                    file_pattern % self.split_name)
        return tf.gfile.Glob(file_pattern)
