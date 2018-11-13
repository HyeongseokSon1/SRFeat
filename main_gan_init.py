#! /usr/bin/python
# -*- coding: utf8 -*-
#import matplotlib.pyplot as plt 
#import cv2
import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config


###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = 20
lr_decay = config.TRAIN.lr_decay
decay_every = 17
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

def train():
    ## create folders to save result images and trained model
    checkpoint_dir = config.TRAIN.checkpoint 
    tl.files.exists_or_mkdir(checkpoint_dir)
    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.bmp', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.bmp', printable=False))


    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, patch_size_l, patch_size_l, 3], name='t_image_input')
    t_target_image = tf.placeholder('float32', [batch_size, patch_size_h, patch_size_h, 3], name='t_target_image')

    net_g= SRGAN_g(t_image, is_train=True, reuse=False)

    ## test inference
    t_sample_image = tf.placeholder('float32', [5, 56, 56, 3], name='t_sample_image')
    net_g_test = SRGAN_g(t_sample_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    tf.summary.scalar('mse_loss', mse_loss)
    merged = tf.summary.merge_all()

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init_= tf.train.AdamOptimizer(lr_v, beta1=beta1)
    g_optim_init = g_optim_init_.minimize(mse_loss, var_list=g_vars)
    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto( log_device_placement=False))
    train_writer = tf.summary.FileWriter(config.TRAIN.summary_g,sess.graph)
    train_writer.add_graph(sess.graph)
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format('SRFeat'), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format('SRFeat'), network=net_g)

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = read_all_imgs(valid_hr_img_list[0:5], path=config.VALID.hr_img_path, n_threads=5) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=scale_imgs_fn)
    
    sample_imgs_96 = read_all_imgs(valid_hr_img_list[0:5], path=config.VALID.lr_img_path, n_threads=5)
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_96, fn=scale_imgs_fn)
   
    val_mset = tl.cost.mean_squared_error(net_g_test.outputs , sample_imgs_384, is_mean=True)  
    val_summary = tf.summary.scalar('val_mse', val_mset)
    
    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    
    lr_hr_list = list(zip(train_hr_img_list,train_lr_img_list))
    random.shuffle(lr_hr_list)
    train_hr_img_list, train_lr_img_list = zip(*lr_hr_list)
    
    i =0
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init+1):
#        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0
 
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)
        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
#        random.shuffle(train_hr_img_list)
#        for idx in range(0, len(train_hr_img_list), batch_size):
        for idx in range(0, len(train_hr_img_list) -batch_size , batch_size):        
            step_time = time.time()
            b_imgs_list = train_hr_img_list[idx : idx + batch_size]
            b_imgs_list_lr = train_lr_img_list[idx : idx + batch_size]
#            print(b_imgs_list)
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_list_lr, fn=get_imgs_fn, path=config.TRAIN.lr_img_path)

            b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=scale_imgs_fn)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_96, fn=scale_imgs_fn)          
        ## If your machine have enough memory, please pre-load the whole train set.
#        for idx in range(0, len(train_hr_imgs), batch_size):
#            step_time = time.time()
#            b_imgs_384 = tl.prepro.threading_data(
#                    train_hr_imgs[idx : idx + batch_size],
#                    fn=crop_sub_imgs_fn, is_random=True,is_thread=True, patch_size=patch_size_h)
#            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn,patch_size=patch_size_l)
#            
#            b_imgs_192 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn2,patch_size=patch_size_l*2)
            ## update G
            summary,errM, out, _ = sess.run([merged,mse_loss,net_g.outputs, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            train_writer.add_summary(summary,i)
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                     
            total_mse_loss += errM
            n_iter += 1
    
            ## quick evaluation on train set
            if (i != 0) and (i % 20 == 0):
                out,val_,val_summ = sess.run([net_g_test.outputs,val_mset,val_summary], {t_sample_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())

                train_writer.add_summary(val_summ,i)
                print("validate")
    
            # save model
            if (i != 0) and (i % 100 == 0):
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format('SRFeat'), sess=sess)
            i= i+1


    train_writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='srgan')
    parser.add_argument('--set',type=str,default='Set5')
    
    args = parser.parse_args()

    tl.global_flag['model'] = args.model
    tl.global_flag['set'] = args.set
    
    train()
