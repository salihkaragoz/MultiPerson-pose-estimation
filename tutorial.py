import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

flags = tf.app.flags
flags.DEFINE_string('test_images_dir', '',
                    'Path to test images.')
flags.DEFINE_string('model_dir', '',
                    'Path to model dir.')
flags.DEFINE_string('label_file', '',
                    'Path to label file.')
FLAGS = flags.FLAGS

MODEL_NAME = FLAGS.model_dir
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = FLAGS.label_file

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  # img = np.array(image.getdata())
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  # if img.shape[1] == 3:
  #     return np.array(image.getdata()).reshape(
  #         (im_height, im_width, 3)).astype(np.uint8)
  # else:
  #     return np.array(image.getdata()).reshape(
  #         (im_height, im_width, 1)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = FLAGS.test_images_dir
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, i) for i in os.listdir(PATH_TO_TEST_IMAGES_DIR)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        f = open('output.txt', 'w')
        f.close()

        overall_time = time.time()

        for idx, image_path in enumerate(TEST_IMAGE_PATHS):
            # image = Image.open(image_path)
            # im_width, im_height = image.size
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            start_time = time.time()
            image_np = cv2.imread(image_path)
            im_height, im_width, _ = image_np.shape
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            output = zip(
                np.squeeze(boxes).tolist(),
                np.squeeze(classes).astype(np.int32).tolist(),
                np.squeeze(scores).tolist(),
            )
            output = [i for i in output if i[1] == 1]

            for o in output:
                txt = image_path[-16:] + ' ' \
                      + str(o[2]) + ' '\
                      + str(int(o[0][0]*im_height)) + ' '\
                      + str(int(o[0][1]*im_width)) + ' '\
                      + str(int(o[0][2]*im_height)) + ' '\
                      + str(int(o[0][3]*im_width)) + ' \n'
                with open('output.txt','a') as f:
                    f.write(txt)

            print 'Img: %s %d/%d time: %s' % (image_path[-16:], idx, len(TEST_IMAGE_PATHS), time.time() - start_time)

            # plt.figure(figsize=IMAGE_SIZE)
            # plt.imshow(image_np)
            # plt.show()
        print 'Finished. Elapsed time: %s'%(time.time() - overall_time)