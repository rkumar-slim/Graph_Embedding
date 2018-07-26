"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
import scipy.io as io
import h5py
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(idx, use_nrm=1, filetest=None, dataset="test_dataset", normal_fac=1.):

    img = filetest[dataset][idx]
    img = np.array(img).astype(np.float32)
    img = img[None, :, :]

    if use_nrm==1:
        img = img * normal_fac

    return img

def load_train_data(idx, use_nrm=1, is_testing=False, batch_size=1, fileA=None, fileB=None, dataset="train_dataset", normal_fac=1.):

    img_A = fileA[dataset][idx*batch_size:(idx+1)*batch_size]
    img_A = np.array(img_A).astype(np.float32)
    img_A = img_A[:, :, :, None]

    img_B = fileB[dataset][idx*batch_size:(idx+1)*batch_size]
    img_B = np.array(img_B).astype(np.float32)
    img_B = img_B[:, :, :, None]

    if not is_testing:
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    img_AB = np.concatenate((img_A, img_B), axis=3)

    if use_nrm==1:
        img_AB = img_AB * normal_fac

    return img_AB
