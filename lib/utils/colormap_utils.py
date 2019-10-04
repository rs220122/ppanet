# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

# lib
import tensorflow as tf
import numpy as np
from PIL import Image

# user packages


COLORMAP_TYPE_LIST = ['cityscapes', 'camvid', 'pascal']


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

    class_vs_colormap = ['sky', 'building', 'column_pole', 'road', 'sidewalk', 'tree',
                         'sign', 'fence', 'car', 'pedestrian',
                         'byciclist', 'void (undefined)']
    return colormap, class_vs_colormap


def label_to_color_image(label, colormap_type):
    """Convert label to rgb image.

    Args:
        label         : Label which have values representing a class.
        colormap_type : Colormap Type.

    Return: color image.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label. Got {}.'.format(
            label.shape))

    if colormap_type == 'camvid':
        colormap, _ = create_camvid_label_colormap()
    elif colormap_type == 'cityscapes':
        colormap = create_cityscapes_label_colormap()
    else:
        raise ValueError('colormap_type {} is not defined.'.format(colormap_type))
    return colormap[label]


def save_annotation(label, save_path,
                    add_colormap=True,
                    normalize_to_unit_values=False,
                    scale_values=False,
                    colormap_type=None):
    """
    This function do following steps.
    1. Annotate to a predicted label using colormap.
    2. Save a annotated label.

    Args:
        label         : Label.
        save_path     : Path.
        add_colormap  : Whether save adding colormap or not.
        normalize_to_unit_values: Normalize or not.
        scale_values  : Scale up to 255 or not.
        colormap_type : Colormap type.
    """

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

    # Save a label.
    pil_image = Image.fromarray(colored_label.astype(dtype=np.uint8))
    with tf.gfile.Open('%s.png' % save_path, mode='w') as f:
        pil_image.save(f, 'PNG')
