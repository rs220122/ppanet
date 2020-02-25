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
    colormap[0] = [128, 64, 128]  # road
    colormap[1] = [244, 35, 232]  # sidewalk
    colormap[2] = [70, 70, 70]    # building
    colormap[3] = [102, 102, 156] # wall
    colormap[4] = [190, 153, 153] # fence
    colormap[5] = [153, 153, 153] # pole
    colormap[6] = [250, 170, 30]  # traffic_light
    colormap[7] = [220, 220, 0]   # traffic_sign
    colormap[8] = [107, 142, 35]  # vegetation
    colormap[9] = [152, 251, 152] # terrain
    colormap[10] = [70, 130, 180] # sky
    colormap[11] = [220, 20, 60]  # person
    colormap[12] = [255, 0, 0]    # rider
    colormap[13] = [0, 0, 142]    # car
    colormap[14] = [0, 0, 70]     # truck
    colormap[15] = [0, 60, 100]   # bus
    colormap[16] = [0, 80, 100]   # train
    colormap[17] = [0, 0, 230]    # motorcycle
    colormap[18] = [119, 11, 32]  # bicycle

    class_vs_colormap = ['road', 'sidewalk', 'building', 'wall', 'fence',
                         'pole', 'traffic_light', 'traffic_sign', 'vegetation',
                         'terrain', 'sky', 'person', 'rider', 'car', 'truck',
                         'bus', 'train', 'motorcycle', 'bicycle']
    return colormap, class_vs_colormap


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
                         'byciclist']# , 'void (undefined)']
    return colormap, class_vs_colormap


def create_pascal_label_colormap():
    """Creates a label colormap used in Pacaldataset."""
    def bitget(val, bit_ind):
        return (val & (1 << bit_ind)) != 0

    colormap = np.zeros((256, 3), dtype=int)
    ind  = np.arange(256, dtype=int)

    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c >>= 3

        colormap[i] = [r, g, b]
    class_vs_colormap = ['background', 'Aeroplane', 'Bicycle', 'Bird', 'Boat',
                         'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                         'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                         'Sheep', 'Sofa', 'Train', 'Tvmonitor']
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
        colormap, _ = create_cityscapes_label_colormap()
    elif colormap_type == 'pascal':
        colormap, _ = create_pascal_label_colormap()
    else:
        raise ValueError('colormap_type {} is not defined.'.format(colormap_type))
    return colormap[label]

def get_class_name(dataset_name):
    """Get class name.

    Args:
        dataset_name: Dataset name.
    Returns:
        class name.
    """
    if dataset_name == 'camvid':
        _, class_name = cerate_camvid_label_colormap()
    elif dataset_name == 'cityscapes':
        _, class_name = create_cityscapes_label_colormap()
    elif dataset_name == 'pascal':
        _, class_name = create_pascal_label_colormap()
    else:
        raise ValueError('dataset_name {} is not defined.'.format(dataset_name))
    return class_name


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
