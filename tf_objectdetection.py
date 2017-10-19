import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import re

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
traffic_sign_class_id = 1
NUM_CLASSES = 90

print("starting download")
opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
print("Has downloaded")
tar_file = tarfile.open(MODEL_FILE)
print("where is the error")
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())

file_name = os.path.basename(MODEL_NAME)


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

# ------------------------------------------------------------
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

print("before path_to")


def detect(detection_graph,
           image_tensor, 
           detection_boxes,
           detection_scores,
           detection_classes,
           image_np,
           categories,
           runs=1):
  
    with tf.Session(graph=detection_graph) as sess:                
        # Actual detection.
        
        times = np.zeros(runs)
        for i in range(runs):
            t0 = time.time()
            (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes], 
                                                feed_dict={image_tensor: image_np})
            t1 = time.time()
            times[i] = (t1 - t0) * 1000

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        return boxes, scores, classes, times

# -----------------------------------------------------------

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4) ]
print(TEST_IMAGE_PATHS)
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

print("before detection")
min_score = 0.1

def filter_boxes(min_score, boxes, scores, classes, categories):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if classes[i] in categories and scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes


def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

i = 1
line = ""

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    #print(detection_boxes.get_shape())
    #print(detection_boxes[1])
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    
    #print(detection_scores.get_shape())
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    #print(detection_classes.get_shape())
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    #print(num_detections.get_shape())
    
    for img in os.listdir('/home/salih/pose-hg-demo/val2017'):
      ims = '/home/salih/pose-hg-demo/val2017/{}'.format(img)
      print(i)
      i = i+1
      #for image_path in TEST_IMAGE_PATHS:
      #print("My laptop take longer" +str(TEST_IMAGE_PATHS))
      #print("is it is it ")
      image = Image.open(ims)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      width, height = image.size


      #print(width)
      #print(height)
      b = scores[0][0]

      #print(scores[0][0])
      #print(boxes[0][0])
      #print(classes)
      #print(num[0])
      #print("that did it.")
      human_id = 1
      
      boxes,scores,classes,_ = detect(detection_graph,
                                  image_tensor,
                                  detection_boxes,
                                  detection_scores,
                                  detection_classes,
                                  image_np_expanded,
                                  traffic_sign_class_id)


      boxes, scores, classes = filter_boxes(min_score, boxes, scores, classes, [human_id])
      # Visualizatio  of the results of a detection.
      #print("iste skorlar")
      #print("scores" + str(scores[0]))
      #print(boxes)


      box_coords =to_image_coords(boxes, height, width)
      score_idx = 0 
      for score in scores:
          #print(score)
          #print(box_coords[score_idx])
          box_cords = str(box_coords[score_idx])
          box_cords = box_cords[2:-1]
          #if(box_cords[0] == " "):
          #  box_cords = box_cords[1:]
            #if (box_cords[0] == " "):
            #  box_cords = box_cords[1:]

          box_cords = box_cords.replace("   ", " ")
          box_cords = box_cords.replace("  ", " ")
          box_cords = box_cords.replace("0. ", "0 ")
          
          #box_cords = re.sub(r"\.\d+","", box_cords)
          line = line + img + " " + str(score) + " "+  str(box_cords) + "\n"
          line = line.replace("   ", " ")
          line = line.replace("  ", " ")
           
          score_idx = score_idx + 1
          #print("end")

      #lime = line.split(" ")
      #if(lime[2] == "0."):
      #    lime[2] == "0"
      #    line = str(lime) 
      #box_coords = str(box_coords[0])
      #box_coords = box_coords[2:-1]
      #if(box_coords[0] == " "):
      #    box_coords = box_coords[1:]
      #    if (box_coords[0] == " "):
      #        box_coords = box_coords[1:]

      #box_coords = box_coords.replace("   ", " ")
      #box_coords = box_coords.replace("  ", " ")
      
      #box_coords = re.sub(r"\.\d+", "", box_coords)
       

      #print(box_coords)
      #print(scores)

      #line =  line +  str(scores[0]) + " " + str(box_coords)+ "\n"
      #print(box_coords)
      #print(classes)
      
      
      
            
      #vis_util.visualize_boxes_and_labels_on_image_array(
      #    image_np,
      #    np.squeeze(boxes),
      #    np.squeeze(classes).astype(np.int32),
      #    np.squeeze(scores),
      #    category_index,
      #    use_normalized_coordinates=True,
      #    line_thickness=3)
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)
      #plt.show()




with open('output_of_tfc.txt','a') as f:
            f.write(str(line))


print("This is the end")

