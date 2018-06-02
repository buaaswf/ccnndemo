"""
Adapted from the inference.py to demonstate the usage of the util functions.
"""

import sys
import numpy as np
import pydensecrf.densecrf as dcrf
import cv2
from skimage import color
# Get im{read,write} from somewhere.
#  try:
    #  from cv2 import imread, imwrite
#  except ImportError:
    #  # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    #  # so you'll need them if you don't have OpenCV. But you probably have them.
    #  from skimage.io import imread, imsave
    #  imwrite = imsave
    # TODO: Use scipy instead.


from skimage.io import imread, imsave
imwrite = imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

def crf(inimage,img_anno):
        fn_im = inimage
        fn_anno = img_anno
        img = inimage
        anno_rgb = img_anno
        rgb = anno_rgb
        print "=========>>", anno_rgb.shape
        #rgb= np.argmax(anno_rgb[0],axis=0)
        print "=======>>",rgb.shape
        print np.max(rgb), np.min(rgb)
        anno_lbl=rgb
        img = img[0]
        img = img.transpose(1, 2, 0)
        colors, labels = np.unique(anno_lbl, return_inverse=True)
        colors = colors[1:]
        colorize = np.empty((len(colors), 3), np.uint8)
        colorize[:,0] = (colors & 0x0000FF)
        colorize[:,1] = (colors & 0x00FF00) >> 8
        colorize[:,2] = (colors & 0xFF0000) >> 16
        n_labels = len(set(labels.flat))-1
        if n_labels <= 1:
            return rgb
        use_2d = True
        if use_2d:
            img = img.astype(int)
            d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
            print n_labels
            U = unary_from_labels(labels, n_labels, gt_prob=0.99, zero_unsure=True)
            d.setUnaryEnergy(U)
            d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NO_NORMALIZATION)
            img = counts = np.copy(np.array(img,dtype = np.uint8),order='C')
            d.addPairwiseBilateral(sxy=(201,401), srgb=(1, 1, 1), rgbim=img,
                                compat=101,
                                kernel=dcrf.FULL_KERNEL,
                                normalization=dcrf.NO_NORMALIZATION)

        else:

            # Example using the DenseCRF class and the util functions
            d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

            # get unary potentials (neg log probability)
            U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=True)
            d.setUnaryEnergy(U)

            # This creates the color-independent features and then add them to the CRF
            feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
            d.addPairwiseEnergy(feats, compat=3,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # This creates the color-dependent features and then add them to the CRF
            feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                            img=img, chdim=2)
            d.addPairwiseEnergy(feats, compat=10,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(5)


# Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)

        return MAP.reshape(img.shape[:2])
if __name__=="__main__":
    crf(sys.argv[1],sys.argv[2],sys.argv[3])
