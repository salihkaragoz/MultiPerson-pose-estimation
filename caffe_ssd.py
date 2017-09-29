import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mlt
#mlt.use('GTK')
#import cv2
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/home/muhammed/caffe/'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

import sys
sys.path.insert(0, 'python')

import caffe
#caffe.set_device(2)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    #print("num_labels -->")
    #print(num_labels)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames



model_def = '/home/muhammed/caffe/models/ssd/models/VGGNet/VOC0712/SSD_512x512_ft/deploy.prototxt'
model_weights = '/home/muhammed/caffe/models/ssd/models/VGGNet/VOC0712/SSD_512x512_ft/VGG_VOC0712_SSD_512x512_ft_iter_120000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model     
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 1
image_resize = 512
net.blobs['data'].reshape(1,3,image_resize,image_resize)

#image = caffe.io.load_image('/home/salih/caffe/examples/images/cat.jpg')
#plt.imshow(image)
#plt.show()
#for file_type in ['/home/salih/caffe/examples/images/test']:
#ComparePlus = open("/home/salih/Desktop/folder/get_name_of_compare_absent.txt").readlines()
#Comprp = [line.split('\n') for line in ComparePlus.readlines()]
#ComparePlus = [x.strip() for x in ComparePlus]
#print len(ComparePlus)
#print (ComparePlus)
count = 0
for img in os.listdir('/home/muhammed/caffe/eval/test2017'):
    #for zimra in ComparePlus:
    #print(zimra,"1")
    count = count + 1
    ims = '/home/muhammed/caffe/eval/test2017/{}'.format(img)
    image = caffe.io.load_image(ims)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    #plt.imshow(transformed_image)
    #plt.show()
    # Forward pass.
    detections = net.forward()['detection_out']
    #print(open("/home/salih/data/VOCdevkit/results/VOC2007/SSD_512x512/Main/comp4_det_test_person.txt").readline().rstrip(),img)

    #print(zimra,"2")

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    #top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.01 and i for i, labz in enumerate(det_label) if labz == 12.0]
    #print(zimra,"3")
    top_indices  = [i for i, labz in enumerate(det_label) if labz == 15.0 ]
    #top_indices  = [x for x, conf in top_indices if x >
    #print(top_indices)
    top_temporary = []
    for i in top_indices:
        #print(i)
        if det_conf[i] <= 0.01:
            #print(i, det_conf[i], "silindi")
            top_indices.remove(i)
        if det_conf[i] >0.01:
            #print(i, det_conf[i], "duruyor")
            top_temporary.append(i)

    #print(zimra,"4")
    top_indices = top_temporary
    #top_indices = list(filter(lambda x: x!=2, top_indices))
    #print(top_indices)
    #top_indices = list(filter(lambda a: a !=5, top_indices))
    #top_indices  = [i for i, conf in top_indices if det_conf[i] > 0.8  
    #print(top_indices)
    top_indices = list(set(top_indices)) 
    #print(labz, conf)
    #print("I think")
    #print(top_indices)
    top_conf = det_conf[top_indices]
    #print(top_conf)
    top_label_indices = det_label[top_indices].tolist()
    #print(top_label_indices)
    top_labels = get_labelname(labelmap, top_label_indices)
    #print(top_labels)
    top_xmin = det_xmin[top_indices]
    #print(top_xmin)
    top_ymin = det_ymin[top_indices]
    #print(top_ymin)
    top_xmax = det_xmax[top_indices]
    #print(top_xmax)
    top_ymax = det_ymax[top_indices]
    #print(top_ymax)
    #print(top_xmin, top_ymin , top_xmax, top_ymax) 
    #print(zimra,"5")
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    #plt.imshow(image)
    #plt.show()
    #currentAxis = plt.gca()
    #print(zimra,"6")
    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
    	ymin = int(round(top_ymin[i] * image.shape[0]))
    	xmax = int(round(top_xmax[i] * image.shape[1]))
    	ymax = int(round(top_ymax[i] * image.shape[0]))
        line = str(img)+ " "+ str(top_conf[i])+" "+ str(xmin)+" "+ str(ymin)+" "+ str(xmax)+" "+ str(ymax)+" "+"\n"
        with open('/home/muhammed/caffe/eval/output_of_test2017.txt','a') as f:
            f.write(str(line))
    	print(str(line), count)
    	#score = top_conf[i]
    	#label = int(top_label_indices[i])
    	#label_name = top_labels[i]
    	#display_txt = '%s: %.2f'%(label_name, score)
    	#coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    	#print(coords)
    	#color = colors[label]
    	#currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    	#currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    	#print(currentAxis)

	#plt.imshow()
    #print(zimra,"7")
    #plt.imshow(image)
    #plt.show()
