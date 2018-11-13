#! /usr/bin/python
# -*- coding: utf8 -*-
import matplotlib.pyplot as plt 
import os, time, pickle, random
import numpy as np

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config 


## Adam
batch_size = 9
lr_init = 1e-4
beta1 = 0.9
## initialize G
n_epoch_init = 20
## adversarial learning 
n_epoch = 4
lr_decay = 0.1
decay_every = 2
patch_size_l = 74
patch_size_h = 296

ni = int(np.sqrt(batch_size))

def  modcrop(imgs, modulo):

    tmpsz = imgs.shape
    sz = tmpsz[0:2]

    h = sz[0] - sz[0]%modulo
    w = sz[1] - sz[1]%modulo
    imgs = imgs[0:h+1, 0:w+1,:]
    return imgs

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def evaluate():
    ## create folders to save result images
    save_dir = config.TEST.save_path
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = config.TEST.checkpoint
    im_path_lr = config.TEST.input_path

    ###====================== PRE-LOAD DATA ===========================###
    valid_lr_img_list = sorted(tl.files.load_file_list(path=im_path_lr, regx='.*.*', printable=False))
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    
    t_image = tf.placeholder('float32', [None, None, None, 3], name='input_image')
      
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/SRFeat_full.npz', network=net_g)
    
    
    for imid in range(len(valid_lr_img_list)):
        valid_lr_img = get_imgs_fn(valid_lr_img_list[imid],im_path_lr)

        print(valid_lr_img.shape)

        valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale to ［－1, 1]
        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
            
        print("took: %4.4fs" % (time.time() - start_time))
    
        print("LR size: %s /  generated HR size: %s" % (valid_lr_img.shape, out.shape)) # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir+'/' + valid_lr_img_list[imid])    



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='srgan')
    parser.add_argument('--set',type=str,default='Set5')
    
    args = parser.parse_args()

    tl.global_flag['model'] = args.model
    tl.global_flag['set'] = args.set
    
    evaluate()
 