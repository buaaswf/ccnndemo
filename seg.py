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
class SegLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
    def reshape(self, bottom, top):
        self.depth = bottom[1].data[0]
        #  print "===========>>", bottom[1].data[0].shape
        #  label = bottom[2].data[0][0]
        #  self.label=np.resize(label,(100,360,480))
        #  print  self.depth.shape, self.gray.shape, self.label.shape
        #  top[0].reshape(self.depth.shape[0],self.depth.shape[1],self.depth.shape[2])
        top[0].reshape(1,*self.depth.shape)
        #  top[0].reshape(self.depth.shape)
        #  if bottom[0].count !=bottom[1].count:
           #  print bottom[0].count, bottom[1].count, bottom[2].count
           #  bottom.thumbnail(bottom[1].data.shape,Image.ANTIALIAS)
        #  if bottom[0].count !=bottom[1].count:
            #  raise Exception("Inputs must have the same dimension.")
    def forward(self, bottom, top):

        from scipy import  ndimage
        rgb = bottom[0].data
        print rgb
        rgb = np.transpose(rgb[0],(1,2,0))
        rgb = np.uint8(rgb)
        rgb = Image.fromarray(rgb)
        gray = rgb.convert('L')
        sem = np.array(gray.getdata(), np.uint8).reshape(360, 480)
        depth = bottom[1].data
        from skimage import measure
        cont = feature.canny(sem, sigma=3)
        sem = cont
        sem = np.array([cont,cont,cont])
        sem = np.transpose(sem,(1,2,0))
        sem = (Image.fromarray(np.uint8(sem))).convert('L')

        #  print np.array(sem).shape
        #  print np.array(rgb).shape
        top[0].data[...] =  np.array(sem,dtype=np.uint8) # np.transpose(sem,(1,2,0))



    def backward(self, top, propagate_down, bottom):
        pass
    def semantic_prob_conv(self, bottom):
        from scipy import  ndimage
        #  print "======>>"
        rgb = bottom[0].data
        #  print bottom[1].data.shape
        #  print type(rgb)
        rgb = np.transpose(rgb[0],(1,2,0))
        rgb = np.uint8(rgb)
        #  print rgb.shape
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
