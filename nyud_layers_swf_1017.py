import sys
sys.path.insert(0,"./python")
import caffe

import numpy as np
from PIL import Image
import scipy.io

import random
#import cv2
x = 480
y = 360
class SegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from NYUDv2
    one-at-a-time while reshaping the net to preserve dimensions.

    The labels follow the 40 class task defined by

        S. Gupta, R. Girshick, p. Arbelaez, and J. Malik. Learning rich features
        from RGB-D images for object detection and segmentation. ECCV 2014.

    with 0 as the void label and 1-40 the classes.

    Use this to feed data to a fully convolutional network.
    ''"""
    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - nyud_dir: path to NYUDv2 dir
        - split: train / val / test
        - tops: list of tops to output from {color, depth, hha, label}
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for NYUDv2 semantic segmentation.

        example: params = dict(nyud_dir="/path/to/NYUDVOC2011", split="val",
                               tops=['color', 'hha', 'label'])
        """
        # config
        #  params = eval(self.param_str)
        self.nyud_dir ='data/nyud/'# params['nyud_dir']
        self.split = 'train'#params['split']
        self.tops = ['color', 'depth'] #params['tops']
        #  self.tops = ['color', 'label'] #params['tops']
        self.random =True #params.get('randomize', True)
        self.seed = '1337'#params.get('seed', None)

        # store top data for reshape + forward
        self.data = {}

        # means
        self.mean_bgr = np.array((116.190, 97.203, 92.318), dtype=np.float32)
        self.mean_hha = np.array((132.431, 94.076, 118.477), dtype=np.float32)
        self.mean_logd = np.array((7.844,), dtype=np.float32)

        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.nyud_dir, self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit (1 is the batch dim)
        for i, t in enumerate(self.tops):
            self.data[t] = self.load(t, self.indices[self.idx])
            top[i].reshape(1, *self.data[t].shape)

    def forward(self, bottom, top):
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]
            #  print top[i].data.shape

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load(self, top, idx):
        if top == 'color':
            return self.load_image(idx)
        elif top == 'label':
            return self.load_label(idx)
        elif top == 'depth':
            return self.load_depth(idx)
        elif top == 'hha':
            return self.load_hha(idx)
        else:
            raise Exception("Unknown output type: {}".format(top))

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/data/images/img_{}.png'.format(self.nyud_dir, idx))
        #  print "ooooooooooo", im.size
        #  im = cv2.resize(im,(360,480))
        #  im.thumbnail((x,y),Image.ANTIALIAS)
        im=im.resize((480, 360), Image.BILINEAR)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean_bgr
        in_ = in_.transpose((2,0,1))
        #  print np.max(in_), np.min(in_)
        #  print in_
        #  in_= in_.reshape(1,*in_.shape)
        #  print "in_",in_.shape
        return in_

    def load_image_bak(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/data/images/img_{}.png'.format(self.nyud_dir, idx)).convert('L')
        #  print "ooooooooooo", im.size
        #  im = cv2.resize(im,(360,480))
        #  im.thumbnail((x,y),Image.ANTIALIAS)
        im=im.resize((480, 360), Image.BILINEAR)
        in_ = np.array(im, dtype=np.float32)
        #  in_ = in_[:,:,::-1]
        #  in_ -= self.mean_bgr
        #  in_ = in_.transpose((1,2,0))
        #  print in_
        in_= in_.reshape(1,*in_.shape)
        #  print "in_",in_.shape
        return in_
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        Shift labels so that classes are 0-39 and void is 255 (to ignore it).
        The leading singleton dimension is required by the loss.
        """
        label = scipy.io.loadmat('{}/segmentation/img_{}.mat'.format(self.nyud_dir, idx))['groundTruth'][0][0][0][0][2].astype(np.uint8)
        label = Image.fromarray(label)
        #  label.thumbnail((x,y), Image.ANTIALIAS)
        label=label.resize((480, 360), Image.BILINEAR)
        label = np.array(label.getdata(), np.uint8).reshape(label.size[1], label.size[0])
        #  print type(label)
        #  label -= 1  # rotate labels
        label = label[np.newaxis, ...]
        #  label = cv2.resize(label,(360,480))
        return label

    def load_depth(self, idx):
        """
        Load pre-processed depth for NYUDv2 segmentation set.
        """
        im = Image.open('{}/data/depth_ori1000/img_{}.png'.format(self.nyud_dir, idx))
        #  d = Image.fromarray(d)
        #  im.thumbnail((x,y), Image.ANTIALIAS)
        im = im.resize((480, 360), Image.BILINEAR)
        #  label = np.array(label.getdata(), np.uint8).reshape(label.size[1], label.size[0])
        d = np.array(im, dtype=np.float32)
        d = np.log(d)
        #  d -= self.mean_logd
        d = (d-np.min(d))/(np.max(d)-np.min(d))*255
        d = d[np.newaxis, ...]
        d = d.astype(int)
        #  d *= 1000
        #  d = d.astype(int)
        #  print d
        #  d = cv2.resize(d,(360,480))
        #  print np.max(d),np.min(d)
        return d

    def load_depth_bak(self, idx):
        """
        Load pre-processed depth for NYUDv2 segmentation set.
        """
        im = Image.open('{}/data/depth_ori1000/img_{}.png'.format(self.nyud_dir, idx))
        d = np.array(im, dtype=np.float32)
        d -= np.min(d)
        d /= (np.max(d)-np.min(d))
        d *= 1000
        d = d.astype(int)
        print d
        return d
    def load_hha(self, idx):
        """
        Load HHA features from Gupta et al. ECCV14.
        See https://github.com/s-gupta/rcnn-depth/blob/master/rcnn/saveHHA.m
        """
        im = Image.open('{}/data/hha/img_{}.png'.format(self.nyud_dir, idx))
        hha = np.array(im, dtype=np.float32)
        hha -= self.mean_hha
        hha = hha.transpose((2,0,1))
        return hha
