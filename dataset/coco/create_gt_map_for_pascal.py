# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-11-07T03:46:13.694Z
# Description: Create grandtruth map from coco.
#              This gt map has the same category as pascal voc.
# ===============================================

"""Create grandtruth map from coco.
This gt map has only the same category as pascal voc.
For more details, see below varialbe 'pascal_names'.

This file assume the following directory structure.

 - create_gt_map_for_pascal.py (current working file.)
 + dataset
   + images          <= please download train2017.zip and val2017.zip and unzip here.
     + val2017
     + train2017
   + json_annotations <= please download annotations_trainval2017.zip and unzip here.
     - instances_{train,val}2017.json
   + annotations
     + stuff          <= please download  stuff_annotations_trainval2017.zip and unzip here.
       + train2017
       + val2017
     + thing         <= created by this scripts.
       + train2017
       + val2017
"""

# lib
import tensorflow as tf
from pycocotools.coco import COCO
import numpy as np
import os
import cv2
from PIL import Image

# user packages

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir',
                           os.path.join('dataset', 'images'),
                           'directory containing the images.')

tf.app.flags.DEFINE_string('annotation_dir',
                           os.path.join('dataset', 'json_annotations'),
                           'directory containing the annotations.')

tf.app.flags.DEFINE_enum('data_type', 'train2017',
                         ['train2017', 'val2017'],
                         'data type. train or val.')

tf.app.flags.DEFINE_string('output_dir',
                           os.path.join('dataset', 'annotations', 'thing'),
                           'output directory.')


def main(argv):
    ann_file = os.path.join(FLAGS.annotation_dir, 'instances_%s.json' % FLAGS.data_type)

    coco = COCO(ann_file)

    pascal_names = [
                'airplane', # aeroplane
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'dining table',
                'dog',
                'horse',
                'motorcycle', # motorbike
                'person',
                'potted plant', # pottedplant
                'sheep',
                'couch', # sofa
                'train',
                'tv'] # tvmonitor

    # カテゴリー名からカテゴリーIDを取得
    catIds = coco.getCatIds(catNms=pascal_names)
    imgIds = set()

    for catId in catIds:
        imgId = set(coco.getImgIds(catIds=catId))
        imgIds = imgIds | imgId
    # カテゴリーIDからイメージIDを取り出す.
    imgIds = list(imgIds)

    imgs = coco.loadImgs(imgIds)
    num_imgs = len(imgs)
    if not os.path.exists(os.path.join(FLAGS.output_dir, FLAGS.data_type)):
        os.makedirs(os.path.join(FLAGS.output_dir, FLAGS.data_type))

    for i, img in enumerate(imgs):

        # Load image.
        img_path = os.path.join(FLAGS.image_dir,
                                FLAGS.data_type,
                                img['file_name'])
        I = np.asarray(Image.open(img_path))
        whole_mask = np.zeros(I.shape[:2])
        for j in range(len(pascal_names)):
            catId = coco.getCatIds(catNms=pascal_names[j])[0]
            # Load the annotation id from category id.
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catId, iscrowd=None)
            if len(annIds) == 0:
                continue
            anns = coco.loadAnns(annIds)
            mask = coco.annToMask(anns[0])
            for ann in anns[1:]:
                mask = mask | coco.annToMask(ann)
            mask = mask * (j+1)
            whole_mask += mask
        # occlusion appeared. and occulusion pixels set to zero.
        whole_mask[whole_mask > 21] = 0
        # Calculate the background rate.
        height, width = whole_mask.shape
        background_rate = np.sum(whole_mask == 0) / (height*width)
        if background_rate < 0.05 or background_rate > 0.995:
            # Theare are the almost all background or no background on image.
            continue
        save_path = os.path.join(FLAGS.output_dir, FLAGS.data_type, img['file_name'].replace('jpg', 'png'))
        img = Image.fromarray(np.uint8(whole_mask))
        img.save(save_path, quality=95)
        print('\r {}/{} saving to {}.'.format((i+1), num_imgs, save_path), end='')
    print()

if __name__ == '__main__':
    tf.app.run()
