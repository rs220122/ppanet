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

tf.app.flags.DEFINE_string('log_dir',
                           None,
                           'log directory.')

tf.app.flags.DEFINE_string('output_path',
                           'output.mp4',
                           'output file name.')

def main(argv):
    result_dir = os.path.join(FLAGS.log_dir, 'semantic_prediction_result')
    fnames = os.listdir(result_dir)
    row_img_fnames = sorted([fname for fname in fnames if 'image' in fname])
    row_pred_fnames = sorted([fname for fname in fnames if 'pred' in fname])

    height, width, _ = cv2.imread(os.path.join(result_dir, row_img_fnames[0])).shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(FLAGS.output_path, fourcc, 20.0, (width, height))

    for img_fname, pred_fname in zip(row_img_fnames, row_pred_fnames):
        img = cv2.imread(os.path.join(result_dir, img_fname))
        pred = cv2.imread(os.path.join(result_dir, pred_fname))

        weighted_img = cv2.addWeighted(img, 0.3, pred, 0.7, 1)
        cv2.imshow('result', weighted_img)
        out.write(weighted_img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    tf.app.flags.mark_flag_as_required('log_dir')
    tf.app.run()
