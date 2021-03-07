from __future__ import division
import tensorflow as tf

from ops import *
from utils import *


def DCGAN(data, options, reuse=False, name="DCGAN"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

            # at this point assuming my data is 512x512

            # project and reshape
            d0 = linear(data, 4*4*1024)
            d0 = tf.nn.relu(batch_norm(tf.reshape(d0, [-1, 4, 4, 1024]), 'g_e1_bn'))
            # conv1 from 4x4 to 8x8
            d1 = tf.nn.relu(batch_norm(deconv2d(d0, 512, 5, 2, name='g_e2_c'), 'g_e2_bn'))

            # conv2 from 8x8 to 16x16
            d2 = tf.nn.relu(batch_norm(deconv2d(d1, 256, 5, 2, name='g_e3_c'), 'g_e3_bn'))

            # conv2 from 16x16 to 32x32
            d3 = tf.nn.relu(batch_norm(deconv2d(d2, 128, 5, 2, name='g_e4_c'), 'g_e4_bn'))

            # conv2 from 32x32 to 64x64
            d4 = tf.nn.relu(batch_norm(deconv2d(d3, 64, 5, 2, name='g_e5_c'), 'g_e5_bn'))

            # conv2 from 64x64 to 128x128
            d5 = tf.nn.relu(batch_norm(deconv2d(d4, 32, 5, 2, name='g_e6_c'), 'g_e6_bn'))

            # conv2 from 128x128 to 256x256
            d6 = tf.nn.relu(batch_norm(deconv2d(d5, 16, 5, 2, name='g_e7_c'), 'g_e7_bn'))

            # conv2 from 256x256 to 512x512
            output = deconv2d(d6, 1, 5, 2, name='g_e8_c')


            return output




def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
