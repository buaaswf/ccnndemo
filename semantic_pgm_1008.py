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
    #  depth = bottom[1].data
    predict_seg = probabilitygraph(rgb)
    return predict_seg
class Pre_SegLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")
    def reshape(self, bottom, top):
        self.depth = bottom[1].data[0]
        top[0].reshape(1,*self.depth.shape)
    def forward(self, bottom, top):
        from scipy import  ndimage
        label = bottom[2].data
        label = np.transpose(label[0],(1,2,0))
        label = np.uint8(label)
        sem = label
        sem = np.transpose(sem,(2,0,1))
        print sem.shape
        sem = np.gradient(sem[0])
        rgb = bottom[0].data
        from skimage import measure
        cont = np.array(sem)
        cont = cont[0]*cont[0] + cont[1]*cont[1]
        cont = cont.reshape(1, cont.shape[0], cont.shape[1])
        rgb = rgb + cont
        rgb = rgb - np.min(rgb)
        from sklearn.preprocessing import normalize
        rgb = normalize(rgb[0][0])*255
        top[0].data[...] = rgb.reshape(1,*rgb.shape)

    def backward(self, top, propagate_down, bottom):
        pass

    def semantic_prob_conv(self, bottom):
        from scipy import  ndimage
        print "======>>"
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
