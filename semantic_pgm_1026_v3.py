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
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from guided_filter.core.filters import FastGuidedFilter, GuidedFilter
import cv2
from inference_rgbd import crf
import random, string

def predict_seg( bottom):
    #rgb = bottom[0].data
    rgb = bottom[0].data
    #  print rgb.shape
    #  depth = bottom[1].data
    predict_seg = probabilitygraph(rgb)
    return predict_seg
def crfdepth( bottom):
    rgb = bottom[0].data
    depth = bottom[1].data
    print depth.shape
    print depth
    print np.max(depth),np.min(depth)
    predict_seg = crf(rgb,depth)
    return predict_seg
def crfdepthoridata(rgbimage,depth):
    rgb = rgbimage
    #depth = bottom[1].data
    print depth.shape
    #print depth
    #print np.max(depth),np.min(depth)
    predict_seg = crf(rgb,depth)
    return predict_seg
class Pre_SegLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")
    def reshape(self, bottom, top):
        #  pass
        self.depth = bottom[1].data[0]
        top[0].reshape(1,*self.depth.shape)
    def forward(self, bottom, top):
        from scipy import  ndimage
        #try:
    #    seg =  crfdepth(bottom)
    #    sem = seg
        depth = bottom[1].data
        mindepth = np.min(depth)
        depthrange = np.max(depth) - np.min(depth)
        depth = (depth -mindepth)/depthrange
        rgb = bottom[0].data
        depth *= 255
        depth = np.uint8(depth)
        #print "---->>",depth.shape
        sem = np.argmax(depth[0],0)
        #for i in range(0,depth.shape[1]):
        #    depth[0][i]=crfdepthoridata(rgb,depth[0][i])
        sem=crfdepthoridata(rgb,sem)
        from skimage import measure
        sem = np.argmax(depth[0],0)
        sem = Image.fromarray(np.uint8(sem)).convert('L')
        sem = np.array(sem,dtype=np.uint8)
        depth = depth.astype(float)
        depth /= 255
        depth *= depthrange
        depth += mindepth
        depth = bottom[1].data
        sem = sem.astype(float)
        #print sem
        sem -= np.min(sem)
        sem /= np.max(sem)-np.min(sem)
        #print depthrange
        sem *= depthrange
        sem += mindepth
        #print sem
        #print depth
        top[0].data[...] = depth
        #top[0].data[...] = depth
        #top[0].data[...] = sem
        #except Exception as e:
        #print e
        #depth = bottom[1].data
        #top = depth


    def backward(self, top, propagate_down, bottom):
        pass

    def semantic_prob_conv(self, bottom):
        from scipy import  ndimage
        #  print "======>>"
        rgb = bottom[0].data
        print bottom[1].data.shape
        print type(rgb)
        rgb = np.transpose(rgb[0],(1,2,0))
        rgb = np.uint8(rgb)
        print rgb.shape
        rgb = Image.fromarray(rgb)
        gray = rgb.convert('L')
        sem = np.array(gray.getdata(), np.uint8).reshape(100,360, 480)

        print "======>>"
        #  sem = bottom[0].data
        depth = bottom[1].data
        #  rgb = bottom[2].data
        from skimage import measure
        contours = np.zeros(sem.shape)
        print "======>>"
        for i in range(0,100):
            contours[i] = measure.find_contours(sem[i], 0.8)
            print "======>>"
            depth[i] = ndimage.denoise_bilateral(depth[i]+contours[i], sigma_range=0.05, sigma_spatial=15)
        #  print "depth===>>>"
        print "======>>",depth.shape
        return depth
