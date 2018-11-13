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
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = 20
## adversarial learning 
n_epoch = 4
lr_decay = config.TRAIN.lr_decay
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

def train():
    ## create folders to save result images and trained model
    save_dir_gan = config.TRAIN.save_valid_results
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = config.TRAIN.checkpoint 
    tl.files.exists_or_mkdir(checkpoint_dir)
    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
#    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.bmp', printable=False))
#    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.bmp', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
#    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, patch_size_l, patch_size_l, 3], name='t_image_input_to_generator')
    t_target_image = tf.placeholder('float32', [batch_size, patch_size_h, patch_size_h, 3], name='t_target_image')

    # Generator
    net_g= SRGAN_g(t_image, is_train=False, reuse=False)
    # Discriminator
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)
    # VGG network
    net_vgg, vgg_target_emb= Vgg19_simple_api((t_target_image+1)/2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((net_g.outputs+1)/2, reuse=True)

    # Feature Discriminator
    vgg_scale5 = 1/12.75
#    vgg_scale4 = 1.4e-3
#    vgg_scale2 = 1e-3
    net_vgg_d, logits_vgg_real = SRGAN_vgg_d(vgg_scale5*vgg_target_emb.outputs, is_train=True, reuse=False)
    _,     logits_vgg_fake = SRGAN_vgg_d(vgg_scale5*vgg_predict_emb.outputs, is_train=True, reuse=True)

######### 
    ## validation
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    #Discriminator loss 
    d_loss1 = 1e-3 *(tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1'))
    d_loss2 = 1e-3 *(tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2'))

    d_vgg_loss1 = 1e-3 *(tl.cost.sigmoid_cross_entropy(logits_vgg_real, tf.ones_like(logits_vgg_real), name='d_vgg_1'))
    d_vgg_loss2 = 1e-3 *(tl.cost.sigmoid_cross_entropy(logits_vgg_fake, tf.zeros_like(logits_vgg_fake), name='d_vgg_2'))

    d_loss = d_loss1 + d_loss2 + d_vgg_loss1 + d_vgg_loss2

    ##
    d_loss1_summary = tf.summary.scalar('d_loss1', d_loss1)
    d_loss2_summary = tf.summary.scalar('d_loss2', d_loss2)
    d_vgg_loss1_summary = tf.summary.scalar('d_vgg_loss1', d_vgg_loss1)
    d_vgg_loss2_summary = tf.summary.scalar('d_vgg_loss2', d_vgg_loss2)    
    merged_d = tf.summary.merge([d_loss1_summary, d_loss2_summary,d_vgg_loss1_summary,d_vgg_loss2_summary])    
    
    ##
    #GAN loss
    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    g_gan_vgg_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_vgg_fake, tf.ones_like(logits_vgg_fake), name='g')    

    #vgg loss
    vgg_loss = tl.cost.mean_squared_error(vgg_scale5*vgg_predict_emb.outputs, vgg_scale5*vgg_target_emb.outputs, is_mean=True) # weight..? feature map rescale?
   
    
    ##
    vgg_summary = tf.summary.scalar('vgg_loss', vgg_loss)
    g_gan_summary = tf.summary.scalar('g_gan_loss', g_gan_loss)
    g_gan_vgg_summary = tf.summary.scalar('g_gan_vgg_loss', g_gan_vgg_loss)
    merged_g = tf.summary.merge([vgg_summary, g_gan_summary, g_gan_vgg_summary])   

    ##
    #Total loss 
    g_loss = vgg_loss + g_gan_loss + g_gan_vgg_loss 

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)
    d_vgg_vars = tl.layers.get_variables_with_name('SRGAN_vgg_d', True, True)
    
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, epsilon=1e-10).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, epsilon=1e-10).minimize(d_loss, var_list=[d_vars, d_vgg_vars])    
    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    summary_writer = tf.summary.FileWriter(config.TRAIN.summary_adv,sess.graph)
    summary_writer.add_graph(sess.graph)
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format('SRFeat'), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format('SRFeat'), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format('SRFeat'), network=net_d)
    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()
#
    params = []
    count_layers =0
    for val in sorted( npz.items() ):
        if(count_layers<16):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        count_layers += 1
        
    tl.files.assign_params(sess, params, net_vgg)

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_list = [0,10001,20001,30001,40001,50001,60001,70001,80001]
    sample_imgs = read_all_imgs([train_hr_img_list[k] for k in sample_list], path=config.TRAIN.hr_img_path, n_threads=batch_size) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=scale_imgs_fn)
    
    sample_imgs_96 = read_all_imgs([train_lr_img_list[k] for k in sample_list], path=config.TRAIN.lr_img_path, n_threads=batch_size)
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_96, fn=scale_imgs_fn)

    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan+'/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan+'/_train_sample_384.png')

    
    ###========================= train GAN (SRGAN) =========================###
    iters =0
    lr_hr_list = list(zip(train_hr_img_list,train_lr_img_list))
    random.shuffle(lr_hr_list)
    train_hr_img_list, train_lr_img_list = zip(*lr_hr_list)

    for epoch in range(0, n_epoch+1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)
    
        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
#        random.shuffle(train_hr_img_list)
        for idx in range(0, len(train_hr_img_list)-batch_size, batch_size):
            iters = iters+1
            
            b_imgs_list = train_hr_img_list[idx : idx + batch_size]
            b_imgs_list_lr = train_lr_img_list[idx : idx + batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_list_lr, fn=get_imgs_fn, path=config.TRAIN.lr_img_path)            
            b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=scale_imgs_fn)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_96, fn=scale_imgs_fn)                      

        ## If your machine have enough memory, please pre-load the whole train set.
#        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
#            b_imgs_384 = tl.prepro.threading_data(
#                    train_hr_imgs[idx : idx + batch_size],
#                    fn=crop_sub_imgs_fn, is_random=True,is_thread=True)
#            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            
        
#            
            summary_d, errD, errD1, errD2, errD3, errD4, _ = sess.run([merged_d, d_loss, d_loss1, d_loss2, d_vgg_loss1, d_vgg_loss2, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})

            ## update G
            summary_g, errG, errV, errA, errA2, _ = sess.run([merged_g, g_loss, vgg_loss, g_gan_loss, g_gan_vgg_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})            
 
            ## summary
            summary_writer.add_summary(summary_d, iters)
            summary_writer.add_summary(summary_g, iters)
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f (d1: %.8f, d2: %.8f, d3_vgg: %.8f, d4_vgg: %.8f), g_loss: %.8f (vgg: %.6f adv: %.6f adv2: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errD1, errD2, errD3, errD4,  errG, errV, errA, errA2))
                      
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

            ## quick evaluation on train set
            if (iters != 0) and (iters % 100 == 0):
                out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
                print("[*] save images")
                tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.png' % iters)
    
            ## save model
            if (iters != 0) and (iters % 1000 == 0):
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}.npz'.format('SRFeat'), sess=sess)
            if (iters != 0) and (iters % 10000 == 0):                
                tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}.npz'.format('SRFeat'), sess=sess)

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        print(log)
            
    summary_writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='srgan')
    parser.add_argument('--set',type=str,default='Set5')
    
    args = parser.parse_args()

    tl.global_flag['model'] = args.model
    tl.global_flag['set'] = args.set
    
    train()
