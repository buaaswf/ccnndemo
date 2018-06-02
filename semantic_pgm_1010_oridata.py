#!/usr/bin/env python
# encoding: utf-8
import sys
sys.path.insert(0,"python")
import  caffe
import numpy as np
from PIL import Image
from skimage import data, img_as_float
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage import feature
from extract_segmentation_nyud import extract_seg
from bidirectional import probabilitygraph
def predict_seg( bottom):
    rgb = bottom[0].data
    #  print "rbg", rgb
    #  print rgb.shape
    #  depth = bottom[1].data
    predict_seg = probabilitygraph(rgb)
    return predict_seg
class Pre_SegLayer(caffe.Layer):

    def setup(self, bottom, top):
        pass
        # check input pair
        #  if len(bottom) != 3:
            #  raise Exception("Need two inputs to compute distance.")
    def reshape(self, bottom, top):
        #  pass
        self.rgb = bottom[0].data[0]
        top[0].reshape(1,*self.rgb.shape)
    def forward(self, bottom, top):
        from scipy import  ndimage
        #  try:
            #  seg =  predict_seg(bottom)
            #  seg = np.uint8(seg*255)
            #  rgb = bottom[0].data
            #  from skimage import measure
            #  cont = np.gradient(seg)
            #  cont = cont[0]*cont[0] + cont[1]*cont[1]
            #  cont = np.transpose(cont,(2,0,1))
            #  top[0].data[...] = rgb + cont.reshape(1,*cont.shape)
        #  except Exception as e:
            #  print e
            #  rgb = bottom[0].data
            #  top = rgb
        rgb = bottom[0].data
        #  depth = bottom[1].data
        top[0].data[...] = rgb
    def backward(self, top, propagate_down, bottom):
        pass

