import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
#  caffe_root = '/SegNet/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
caffe_root = "/home/swf/work/caffe-segnet/"
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
def extract_seg(image):
# Import arguments

    model = "/home/swf/work/segNet-tutorial/models/bayesian_segnet_deploy_nyud_fcn.prototxt"
    weights= "/home/swf/work/segNet-tutorial/example_models/bayesian_segnet_nyud_dataori_fcn_iter_600000.caffemodel"

    caffe.set_mode_gpu()

    net = caffe.Net(model,
                    weights,
                    caffe.TEST)
    input_image_raw = image
    input_shape=[1,3,360,480]

    #  input_image_raw = caffe.io.load_image(input_image_file)
    input_image_raw = image[0]
    #  ground_truth = cv2.imread(ground_truth_file, 0)

    print input_image_raw.shape
    input_image = input_image_raw
    #  input_image = caffe.io.resize_image(input_image_raw[0], (input_shape[2],input_shape[3]))
    print input_image.shape
    #  input_image = input_image*255
    #  input_image = input_image.transpose((2,0,1))
    #  input_image = input_image[(2,1,0),:,:]
    input_image = np.asarray([input_image])
    #  input_image = np.repeat(input_image,input_shape[0],axis=0)
    print input_image.shape

    out = net.forward_all(data=input_image)

    #  predicted = net.blobs['prob'].data
    #  net.forward()
    #  image = net.blobs['data'].data
    #  print image.shape
    #  label = net.blobs['label'].data
    predicted = net.blobs['prob'].data
    image = np.squeeze(image[0,:,:,:])
    output = np.squeeze(predicted[0,:,:,:])
    ind = np.argmax(output, axis=0)

    r = ind.copy()
    g = ind.copy()
    b = ind.copy()
    rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
    rgb[:,:,0] = r/255.0
    rgb[:,:,1] = g/255.0
    rgb[:,:,2] = b/255.0
    return rgb
