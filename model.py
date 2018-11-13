#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py

## new generator- denser skip 
def SRGAN_g(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.01)
#    w_init = tf.contrib.layers.variance_scaling_initializer()
#    w_init = tf.contrib.layers.xavier_initializer()

#    b_init = tf.constant_initializer(value=0.0)
    b_init = None
#    g_init = tf.random_normal_initializer(1., 0.01)
    g_init = tf.ones_initializer()
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vss:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 128, (9, 9), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = []

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn       
            
            t = Conv2d(nn, 128, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,b_init=b_init, name='n64s1/c3/%s' % i)
            temp.append(t)
            

#        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,b_init=b_init, name='n64s1/c/m')
#        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp[0],temp[1],temp[2],temp[3],
                              temp[4],temp[5],temp[6],temp[7], 
                              temp[8],temp[9],temp[10],temp[11], 
                              temp[12],temp[13],temp[14]], tf.add, 'add3')
        # B residual blacks end

        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, act=lrelu, n_out_channel=None, name='pixelshufflerx2/1')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init ,name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, act=lrelu, n_out_channel=None,  name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='out')
        return n


def SRGAN_d(t_image, is_train=True, reuse=False):
    """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
#    w_init = tf.truncated_normal_initializer(stddev=0.01)
    w_init = tf.contrib.layers.variance_scaling_initializer()

    b_init = tf.constant_initializer(value=0.0)
#    g_init = tf.random_normal_initializer(1., 0.02)
    g_init = tf.ones_initializer()

    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init,b_init=b_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2),  padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n,  act=lrelu, is_train=is_train,  gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, 128, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, 256, (3, 3), (2, 2), padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = Conv2d(n, 512, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n512s1/b')

        n = Conv2d(n, 512, (3, 3), (2, 2), padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n512s2/b')

        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024,act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1,name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits

def SRGAN_d2(input_images, is_train=True, reuse=False):
#    w_init = tf.random_normal_initializer(stddev=0.01)
    w_init = tf.contrib.layers.variance_scaling_initializer()
    
    b_init = None # tf.constant_initializer(value=0.0)
#    gamma_init=tf.random_normal_initializer(1., 0.02)
    gamma_init = tf.ones_initializer()
    
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu,
                padding='SAME', W_init=w_init, name='h0/c')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h1/bn')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h2/bn')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h3/bn')
        net_h4 = Conv2d(net_h3, df_dim*16, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h4/bn')
        net_h5 = Conv2d(net_h4, df_dim*32, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h5/bn')
        net_h6 = Conv2d(net_h5, df_dim*16, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h6/bn')
        net_h7 = Conv2d(net_h6, df_dim*8, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net_h7 = BatchNormLayer(net_h7, is_train=is_train,
                gamma_init=gamma_init, name='h7/bn')

        net = Conv2d(net_h7, df_dim*2, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='res/bn')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='res/bn2')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = BatchNormLayer(net, is_train=is_train,
                gamma_init=gamma_init, name='res/bn3')
        net_h8 = ElementwiseLayer(layer=[net_h7, net],
                combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity,
                W_init = w_init, name='ho/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits


def SRGAN_vgg_d2(t_image, is_train=True, reuse=False):
#    w_init = tf.truncated_normal_initializer(stddev=0.01)
    w_init = tf.contrib.layers.variance_scaling_initializer()

    b_init = tf.constant_initializer(value=0.0)
#    g_init = tf.random_normal_initializer(1., 0.02)
    g_init = tf.ones_initializer()

    feat_dim = 64
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_vgg_d", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, feat_dim, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init,b_init=b_init, name='n64s1/c')

        n = Conv2d(n, feat_dim, (3, 3), (2, 2),  padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n,  act=lrelu, is_train=is_train,  gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, feat_dim*2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, feat_dim*2, (3, 3), (2, 2), padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, feat_dim*4, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, feat_dim*4, (3, 3), (2, 2), padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = Conv2d(n, feat_dim*8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n512s1/b')

        n = Conv2d(n, feat_dim*8, (3, 3), (2, 2), padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n512s2/b')

#         network = Conv2d(network, n_filter=1024, filter_size=(3, 3),
#                          strides=(1,1), act=tf.nn.relu, padding='SAME',  W_init=w_init, b_init=b_init,name='conv6')
#         network = Conv2d(network, n_filter=1024, filter_size=(3, 3),
#                          strides=(1,1), act=tf.nn.relu, padding='SAME',  W_init=w_init, b_init=b_init,name='conv7')             
#         network = Conv2d(network, n_filter=1, filter_size=(3, 3),
#                          strides=(1,1), act=None, padding='SAME',  W_init=w_init, b_init=b_init, name='out')
#         logits = network.outputs
#         network.outputs = tf.nn.sigmoid(network.outputs)
        
        n = Conv2d(n, 1024, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='d1024')
        n = Conv2d(n, 1, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='out')
#        n = FlattenLayer(n, name='f')
#        n = DenseLayer(n, n_units=1024,act=lrelu, name='d1024')
#        n = DenseLayer(n, n_units=1,name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits
def SRGAN_vgg_d(t_image, is_train=True, reuse=False):
#    w_init = tf.truncated_normal_initializer(stddev=0.01)
    w_init = tf.contrib.layers.variance_scaling_initializer()

    b_init = tf.constant_initializer(value=0.0)
#    g_init = tf.random_normal_initializer(1., 0.02)
    g_init = tf.ones_initializer()

    feat_dim = 64
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_vgg_d", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, feat_dim, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init,b_init=b_init, name='n64s1/c')

        n = Conv2d(n, feat_dim, (3, 3), (2, 2),  padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n,  act=lrelu, is_train=is_train,  gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, feat_dim*2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, feat_dim*2, (3, 3), (2, 2), padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, feat_dim*4, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, feat_dim*4, (3, 3), (2, 2), padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = Conv2d(n, feat_dim*8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n512s1/b')

        n = Conv2d(n, feat_dim*8, (3, 3), (2, 2), padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train,  gamma_init=g_init, name='n512s2/b')

        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024,act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1,name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits


def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb = tf.maximum(0.0,tf.minimum(rgb,1.0))        
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else: # TF 1.0
            # print(rgb_scaled)
            
            red, green, blue = tf.split(rgb_scaled, 3, 3)
#        assert red.get_shape().as_list()[1:] == [224, 224, 1]
#        assert green.get_shape().as_list()[1:] == [224, 224, 1]
#        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
#        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_2')
        conv2 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool3')
        conv3 = network
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_4')
        conv4 = network
        
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool4')                               # (batch_size, 14, 14, 512)
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_4')
        conv5 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool5')                               # (batch_size, 7, 7, 512)
        """ fc 6~8 """
#        network = FlattenLayer(network, name='flatten')
#        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
#        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
#        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv5