import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import scipy.misc
#import cv2
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')
#    return cv2.imread(path + file_name)

def scale_imgs_fn_input(x):
    """for 0-1 input"""
    x = x/(255.)
#    x = x-1.

    return x
def scale_imgs_fn(x):
    x = x-127.5
    x = x/(255./2.)
    return x

def get_labels_fn(file_name, path):
    """ for semantic segmentation data """
    num_class = 20
    # return scipy.misc.imread(path + file_name).astype(np.float)
    labelmap = scipy.misc.imread(path + file_name,mode='P')
    label = np.zeros([labelmap.shape[0], labelmap.shape[1], num_class], dtype=np.bool)
    for i in range(num_class):
        temp = np.zeros([labelmap.shape[0],labelmap.shape[1]],dtype=np.bool)
        temp[labelmap==(i+1)] = 1
        label[:,:,i] = temp
        
    return label

def crop_sub_imgs_fn(x, is_random=True, is_thread=False, patch_size=384):
    if(is_random):
        xhw = crop(x, wrg=patch_size, hrg=patch_size, is_random=is_random)

        x = xhw[0]
        h_offset = xhw[1]
        w_offset = xhw[2]
        x = x / (255. / 2.)
        x = x - 1.
#        x = x - 128.
        if(is_thread):
            return x         
        else:		
            return x,h_offset,w_offset
    else:
        x = crop(x, wrg=patch_size, hrg=patch_size, is_random=is_random)
        x = x / (255. / 2.)
        x = x - 1.
#        x = x - 128.
        
        return x


def crop_sub_labels_fn(x, h_offset, w_offset,patch_size):

    wrg=patch_size
    hrg=patch_size
    return x[h_offset: hrg+h_offset ,w_offset: wrg+w_offset].astype(np.float32)                
    

def downsample_fn(x,patch_size):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=(patch_size, patch_size), interp='bicubic',mode=None)    

#    x = cv2.resize(x, (patch_size, patch_size), interpolation = cv2.INTER_CUBIC)
    x = x / (255. / 2.)
    x = x - 1.
#    x = x -128.
    return x

