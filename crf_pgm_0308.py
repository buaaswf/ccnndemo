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
from skimage.feature import hog
from skimage import data, color, exposure
from inference_rgbd import crf
import cv2
####
# input bottom:depth_pred, depth_gt
# output top: gradient(depth_pred)hog_pred gradient(depth_gt)hog_gt
###
class CRFLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
    def reshape(self, bottom1,top):
        self.depth_pred = bottom1[0].data

        #  print self.depth_pred.shape
        self.depth_img = bottom1[1].data
        #  print self.depth_gt.shape
        top[0].reshape(*self.depth_pred.shape)
        top[1].reshape(*self.depth_img.shape)
    def forward(self, bottom, top):
        from scipy import  ndimage
        depth_pred = bottom[0].data
        depth_pred = crf(depth_pred)
        depth_gt = bottom[1].data
        mindepth_pred = np.min(depth_pred)
        depthrange_pred = np.max(depth_pred) - np.min(depth_pred)
        depth_pred = (depth_pred -mindepth_pred)/depthrange_pred
        depth_pred *= 255
        depth_pred = np.uint8(depth_pred)
        mindepth_gt = np.min(depth_gt)
        depthrange_gt = np.max(depth_gt) - np.min(depth_gt)
        depth_gt = (depth_gt -mindepth_gt)/depthrange_gt
        depth_gt *= 255
        depth_gt = np.uint8(depth_gt)
        #  print depth_gt[0][0].shape
        fd, depth_gt[0][0] = hog(depth_gt[0][0], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
        depth_gt[0][0] = exposure.rescale_intensity(depth_gt[0][0], in_range=(0, 0.02))
        depth_pred = depth_pred.astype(float)
        depth_pred /= 255
        depth_pred *= depthrange_pred
        depth_pred += mindepth_pred
        depth_pred = np.argmax(depth_pred[0],axis=0)
        top[0].data[...] = depth_pred
        depth_gt = depth_gt.astype(float)
        depth_gt /= 255
        depth_gt *= depthrange_gt
        depth_gt += mindepth_gt
        top[1].data[...] = depth_gt


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
