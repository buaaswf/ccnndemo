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

def predict_seg( bottom):
    rgb = bottom[0].data
    #  print rgb.shape
    #  depth = bottom[1].data
    predict_seg = probabilitygraph(rgb)
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
        try:
            seg =  predict_seg(bottom)
            #  print seg
            seg = np.uint8(seg*255)
            sem = seg
            depth = bottom[1].data
            mindepth = np.min(depth)
            depthrange = np.max(depth) - np.min(depth)
            depth = (depth -mindepth)/depthrange
            #  print depth
            #  print depth.shape
            from skimage import measure
            sem = Image.fromarray(np.uint8(sem)).convert('L')
            import random, string
            #  sem.save(''.join(random.choice(string.lowercase) for i in range(20))+".png")
            sem = np.array(sem,dtype=np.uint8)
            cont = np.gradient(sem)
            cont = cont[0]*cont[0] + cont[1]*cont[1]
            #  cont_tmp = Image.fromarray(np.uint8(cont)).convert('L')
            #  cont_tmp.save("cont"+''.join(random.choice(string.lowercase) for i in range(20))+".png")
            for i  in range(0,256,1):
                depth[0][i] = denoise_bilateral(depth[0][i], sigma_range=0.05, sigma_spatial=1,multichannel=False)
                #  print i,
            #  cont = np.transpose(cont,(2,0,1))
            depth -= mindepth
            depth *= depthrange
            top[0].data[...] = depth + cont
            #  print top[0].shape
            #  print bottom[0].data
            #  top = bottom
            #  print top[0].data
            #  print bottom[0].data

            #  top = bottom
        except Exception as e:
            print e
            depth = bottom[1].data
            top = depth


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
