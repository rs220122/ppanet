# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description: convert camvid dataset to tfrecord.
#
# ===============================================

"""
Converts camvid data to TFRecord file format with Example protos.

This dataset is expected to have the following directory structure:

    + dataset
      - build_camvid.py (current working directory).
      - build_data.py
      + CamVid
        + tfrecord
        + train

Image Directory:
    ./CamVid/{}
    {}: train or val
    default: train

Annotation Directory:
    ./CamVid/{}
    {}: trainannot or valannot
    default: trainannot

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

# user packages
import build_data

FLAGS = tf.app.flags.FLAGS


# 正解ラベルが保存されているディレクトリ
# directory in which is saved target labels.
tf.app.flags.DEFINE_string(
    'semantic_segmentation_dir',
    './CamVid/trainannot',
    'Folder containing semantic segmentation for training and validation')

# それぞれのデータセットのパスが書かれたテキストファイルが保存されているディレクトリ
# directory in which is saved text files written about each dataset path.
tf.app.flags.DEFINE_string(
    'list_dir',
    './CamVid/list',
    'Folder containing lists for training and validation')

# TFRecordに変換されたファイルを保存する場所
# directory in which is saved files converted to TFRecord.
tf.app.flags.DEFINE_string(
    'output_dir',
    './CamVid/tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')

NUM_SHARD = 1

def _convert_dataset(dataset_split):
    """Converts the specified dataset split to TFRecord format.

    Args:
        dataset_split: The dataset split (e.g., train, test).

    Raises:
        RuntimeError: If loaded image and label have different shape.
    """

    # get the image and annotation path from dataset text.
    dataset = os.path.basename(dataset_split)[:-4]
    sys.stdout.write('Processing ' + dataset + '\n')

    # split image and annotation by video type.
    img_path_dict = {'0001TP': [], '0006R0': [], '0016E5': [], 'Seq05VD': []}
    ann_path_dict = {'0001TP': [], '0006R0': [], '0016E5': [], 'Seq05VD': []}

    img_list = []
    ann_list = []

    # get the dataset path.
    with open(dataset_split, 'r') as f:
        for row in f:
            row_list = row.strip().split(' ')
            img_path = row_list[0]
            ann_path = row_list[1]
            img_fname = os.path.basename(img_path)
            ann_fname = os.path.basename(ann_path)

            assert img_fname.split('_')[0] == ann_fname.split('_')[0]
            img_list.append(img_path)
            ann_list.append(ann_path)

    image_reader = build_data.ImageReader(channels=3)
    label_reader = build_data.ImageReader(channels=1)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    assert len(img_list) == len(ann_list)

    # create each video type tfrecord
    num_img = len(img_list)
    num_img_per_shard = math.ceil(num_img / NUM_SHARD)

    print('The number of %s image: %d' %(dataset_split, num_img))
    for shard_id in range(NUM_SHARD):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id+1, NUM_SHARD))
        start_idx = shard_id * num_img_per_shard
        end_idx = min((shard_id + 1) * num_img_per_shard, num_img)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i in range(start_idx, end_idx):
                img_path = img_list[i]
                ann_path = ann_list[i]
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_img, shard_id
                ))
                sys.stdout.flush()

                # Read the image
                img_data = tf.gfile.FastGFile(img_path, 'rb').read()
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
    dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_dir, '*.txt'))
    for dataset_split in dataset_splits:
        _convert_dataset(dataset_split)


if __name__ == '__main__':
    tf.app.run()
