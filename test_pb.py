import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import glob
import time
import cv2

import tensorflow as tf

from lib.utils import colormap_utils
os.environ['KMP_DUPLICATE_LIB_OK']='True'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('pb_path', None,
                   'Path to tensorflow frozen graph.')

flags.DEFINE_string('dataset_name', None,
                    'Dataset name.')

flags.DEFINE_string('dataset_path', None,
                    'Dataset path')

flags.DEFINE_string('output_dir', None,
                    'Path to save output.')


class PPANet(object):
    """Class to load deeplab model and run inference"""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPrediction:0'
    INPUT_SIZE = (1024, 2048)

    def __init__(self):
        '''Creates and loads pretrained deeplab model.'''
        self.graph = tf.Graph()

        graph_def = None
        with open(FLAGS.pb_path, 'rb') as f:
            graph_def = tf.GraphDef.FromString(f.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """

        resized_image = image.convert('RGB').resize(self.INPUT_SIZE[::-1], Image.ANTIALIAS)
        print('start inference')
        start = time.perf_counter()
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        print('end inference time: {:.2f}sec'.format(time.perf_counter() - start))
        return np.asarray(resized_image), seg_map


    def vis_segmentation(self, seg_map, img=None, output_path=None):
        label = colormap_utils.label_to_color_image(seg_map, FLAGS.dataset_name)

        if img is not None:
            label = cv2.addWeighted(img, 0.8, label, 0.5, 0)

        if output_path is not None:
            tf.gfile.MakeDirs(os.path.dirname(output_path))
            pil_label = Image.fromarray(label)
            pil_label.save(output_path)
        # plt.imshow(label)
        # plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('pb_path')
    flags.mark_flag_as_required('dataset_name')
    flags.mark_flag_as_required('dataset_path')
    flags.mark_flag_as_required('output_dir')

    model = PPANet()
    img_pattern = FLAGS.dataset_path
    # img_pattern = os.path.join('dataset', 'cityscapes', 'leftImg8bit', 'val',
    #                        '*', '*_leftImg8bit.png')
    img_list = sorted(glob.glob(img_pattern))
    print('number of image: {}'.format(len(img_list)))

    for i, img in enumerate(img_list):
        image = Image.open(img)
        im, seg = model.run(image)
        model.vis_segmentation(seg, im, os.path.join(FLAGS.output_dir, '{}.png'.format(i)))
