# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description: Create labels' video for visualization.
#
# ===============================================

# lib
import tensorflow as tf
import os
import cv2

# user packages

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('result_dir',
                           None,
                           'log directory.')

tf.app.flags.DEFINE_string('output_path',
                           'output.mp4',
                           'output file name.')

def main(argv):
    fnames = os.listdir(result_dir)
    imgs = sorted(fnames)

    height, width, _ = cv2.imread(os.path.join(result_dir, imgs[0])).shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(FLAGS.output_path, fourcc, 20.0, (width, height))

    for img_fname, pred_fname in zip(imgs, row_pred_fnames):
        img = cv2.imread(os.path.join(result_dir, img_fname))

        cv2.imshow('result', img)
        out.write(img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    tf.app.flags.mark_flag_as_required('result_dir')
    tf.app.run()
