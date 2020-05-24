# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

"""Contains common utility functions and classes for building dataset.

This script contains utility functions and classes to converts dataset to
TFRecord file format with Example protos.

The Example proto contains the following fields:
        image/filename    : image filename.
        image/encoded     : image content.
        image/height      : image height.
        image/width       : image width.
        image/channels     : image channels.
        label/encoded     : encoded semantic segmentation content.
"""
# lib
import tensorflow as tf
import collections
import six

# user packages


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, channels=3, img_format='png'):
        """Class constructor.

        Args:
            channels: Image channels.
        """
        self._decode_data = tf.placeholder(dtype=tf.string)
        self._session = tf.Session()

        if img_format == 'png':
            self._decode = tf.image.decode_png(self._decode_data,
                                               channels=channels)
        elif img_format == 'jpeg':
            self._decode = tf.image.decode_jpeg(self._decode_data,
                                                channels=channels)
        else:
            raise RuntimeError('This format {} is not implemented.'.format(img_format))

    def read_image_dims(self, image_data):
        """Reads the image dimensions.

        Args:
            image_data: string of image data.

        Returns:
            image_height and image_width.
        """
        image = self.decode_image(image_data)
        return image.shape[:2]


    def decode_image(self, image_data):
        """Decodes the image data string.

        Args:
            image_data: string of image data.

        Returns:
            Decoded image data.

        Raises:
            ValueError: Value of image channels not supported.
        """
        image = self._session.run(self._decode,
                                  feed_dict={self._decode_data: image_data})

        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')

        return image


def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.
  Args:
    values: A scalar or list of values.
  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    A TF-Feature.
  """
  def norm2bytes(value):
    return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def image_seg_to_tfexample(filename, image_data, height, width, ann_data):
  """Converts one image/segmentation pair to tf example.
  Args:
    image/filename    : image filename.
    image/encoded     : image content.
    image/height      : image height.
    image/width       : image width.
    image/channels     : image channels.
    label/encoded     : encoded semantic segmentation content.
  Returns:
    tf example of one image/segmentation pair.
  """
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_list_feature(image_data),
      'image/filename': _bytes_list_feature(filename),
      'image/height': _int64_list_feature(height),
      'image/width': _int64_list_feature(width),
      'image/channels': _int64_list_feature(3),
      'label/encoded': (_bytes_list_feature(ann_data))
  }))
