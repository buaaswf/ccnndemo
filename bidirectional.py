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
from  extract_segmentation_nyud import extract_seg
from skimage import data, segmentation, color
from skimage.future import graph
from sklearn import preprocessing
#  from matplotlib import pyplot as plt
def predict_seg( bottom):
    rgb = bottom[0].data
    #  depth = bottom[1].data
    predict_seg = extract_seg(rgb)
    return predict_seg
def probabilitygraph(rgb):
    #  print rgb.shape
    rgb = rgb[0]
    #  print bottom[0][:,:,0].shape
    #  rgb[0] = (rgb[0]-np.min(rgb[0]))/(np.max(rgb[0])-np.min(rgb[0]))
    #  rgb[1] = (rgb[1]-np.min(rgb[1]))/(np.max(rgb[1])-np.min(rgb[1]))
    #  rgb[2] = (rgb[2]-np.min(rgb[2]))/(np.max(rgb[2])-np.min(rgb[2]))
    #  bottom[0][:,:,0] = preprocessing.normalize(bottom[0][:,:,0])
    #  bottom[0][:,:,1] = preprocessing.normalize(bottom[0][:,:,1])
    #  bottom[0][:,:,2] = preprocessing.normalize(bottom[0][:,:,2])
    img = rgb #bottom[0].data
    #  print "img", img.shape
    #  print img
    img = np.transpose(img,(1,2,0))
    img = np.array(img,dtype=np.float)
    #  print img

    #  print img.shape
    labels1 = segmentation.slic(img, compactness=30, n_segments=400)
    #  print "labels1",labels1
    #  out1 = color.label2rgb(labels1, img, kind='avg')
    #  print labels1.shape
    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, img, kind='avg')
    #  print "out2,",out2
    return out2

class Pre_SegLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
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
        #####################################################3
        #######
        #######
        ###
        #1 semantic labels
        #2 depth
        #3
        #####################################################33
        #  self.diff[...] = bottom[0].data - bottom[1].data
        #  label_depth = {}
        #  semantic = bottom[0].data
        #  depth_1 = bottom[1].data
        #  for labels in set(list(semantic)):
            #  label_depth["labels"] = [np.min(depth_1[semantic==labels]), np.max(depth_1[semantic==labels]), np.mean(depth_1[semantic==labels])]
        #  print label_depth
        #  print "==========>>"
        #  top[0].data[...] =   #  import operator
        #  top[1].data[...] = self.depth     #  import operator
        #  sorted_x = sorted(label_depth.items(), key=operator.itemgetter(1)[0])
        #  top[0].data[...] = semantic_prob_conv(bottom)

        from scipy import  ndimage
        #  print "======>>"
        rgb = bottom[0].data
        seg = probabilitygraph(bottom)
        #  seg =  predict_seg(bottom)
        print bottom[1].data.shape
        print type(rgb)
        rgb = np.transpose(rgb[0],(1,2,0))
        rgb = np.uint8(rgb)
        print rgb.shape
        rgb = Image.fromarray(rgb)
        gray = rgb.convert('L')
        #  sem = np.array(gray.getdata(), np.uint8).reshape(360, 480)
        sem = seg

        #  print "======>>"
        #  sem = bottom[0].data
        depth = bottom[1].data
        #  rgb = bottom[2].data
        from skimage import measure
        #  print "======>>"
        #  print sem.shape
        #  sem = np.transpose(sem,(2,0,1))
        #  print sem.shape
        sem = Image.fromarray(np.uint8(sem)).convert('L')
        sem = np.array(sem,dtype=np.uint8)

        #  cont = measure.find_contours(sem, 0.8)
        #  cont = feature.canny(sem, sigma=3)
        cont = sem
            #  print "======>>"
        #  print depth.shape
        #  depth = denoise_bilateral(np.uint8(depth+cont), sigma_range=0.05, sigma_spatial=15)
        #  print cont.shape
        #  print depth[0][1]+cont
        for i in range(0,100):
            #  print depth[0].shape
            #  print "======>>"
            #  depth[i] = denoise_bilateral((depth[0][i]+cont).astype(float), sigma_range=0.05, sigma_spatial=15)
            depth[0][i] = depth[0][i]+cont.astype(float)

        #  print "depth===>>>"
        #  print "======>>",depth.shape
        top[0].data[...] = depth



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
