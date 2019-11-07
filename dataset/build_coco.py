# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-11-07T07:04:33.328Z
# Description:
#
# ===============================================

"""Converts camvid data to TFRecord file format with Example protos.

This dataset is expected to have the following directory structure:

    + dataset
      - build_coco.py (current  working file.)
      - build_data.py
      + coco
        - create_gt_map_for_pascal.py <= create the thing_annotations.
        + dataset
          + images                <= containing the COCO images.
            + {train2017, val2017}
          + annotations            <= containing the COCO stuff and thing-only annotations.
            + stuff
            + thing
          + json_annotations      <= containing the COCO json annotations.
            + {train2017, val2017}
        - labels.txt               <= written about the COCOStuff class labels.

Image Directory:
    ./coco/dataset/images/{}
    {}: train2017 or val2017
    default: train2017

Annotation Directory:
    ./coco/dataset/annotations/{a}/{b}
    {a}: stuff or thing
    {b}: train2017 or val2017
    default: thing/train2017

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:
    image/filename    : image filename.
    image/encoded     : image content.
    image/height      : image height.
    image/width       : image width.
    image/channels    : image channels.
    label/encoded     : encoded semantic segmentation content.
"""

# lib
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import math
import re
import glob

# user packages
import build_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('annotation_dir',
                           os.path.join('coco', 'dataset', 'annotations', 'thing'),
                           'cocostuff or cocothing annotation directory')

tf.app.flags.DEFINE_string('image_dir',
                           os.path.join('coco', 'dataset', 'images'),
                           'image directory.')

tf.app.flags.DEFINE_string('output_dir',
                           os.path.join('coco', 'tfrecord_thing'),
                           'Path to save converted SSTable of TensorFlow examples.')

NUM_SHARD = 1

def _convert_dataset(dataset_split):
    """Converts the specified dataset split to TFRecord format.

    Args:
        dataset_split: The dataset split (e.g., train, val, trainval)

    Raises:
        RuntimeError: If loaded image and label have different shape.
    """

    # get the annotation paths.
    pattern = os.path.join(FLAGS.annotation_dir, '%s2017' % dataset_split, '*.png')
    ann_list = sorted(glob.glob(pattern))

    image_reader = build_data.ImageReader(channels=3)
    label_reader = build_data.ImageReader(channels=1)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    num_img = len(ann_list)
    num_img_per_shard = math.ceil(num_img / NUM_SHARD)

    print('The number of %s image: %d' %(dataset_split, num_img))
    for shard_id in range(NUM_SHARD):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id+1, NUM_SHARD))
        start_idx = shard_id * num_img_per_shard
        end_idx = min((shard_id+1)*num_img_per_shard, num_img)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i in range(start_idx, end_idx):
                ann_path = ann_list[i]
                img_path = os.path.join(
                    FLAGS.image_dir,
                    ann_path.split(os.sep)[-2],
                    os.path.basename(ann_path).replace('.png', '.jpg'))
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_img, shard_id))
                sys.stdout.flush()

                if not os.path.exists(img_path):
                    raise ValueError('image {} dont exists'.format(img_path))

                # Read the image
                img_data = tf.gfile.FastGFile(img_path,  'rb').read()
                height, width = image_reader.read_image_dims(img_data)

                ann_data = tf.gfile.FastGFile(ann_path, 'rb').read()
                ann_height, ann_width = label_reader.read_image_dims(ann_data)
                if height != ann_height or width != ann_width:
                    raise RuntimeError('Shape mismatched between image and annotation.')

                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(os.path.basename(img_path),
                                                            img_data,
                                                            height,
                                                            width,
                                                            ann_data)
                tfrecord_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()



def main(argv):
    for dataset_split in ['train', 'val']:
        _convert_dataset(dataset_split)

if __name__ == '__main__':
    tf.app.run()
