from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import h5py
from module import *
from utils import *
from math import floor
from random import shuffle

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.itr_num = args.itr_num
        self.time_samples = args.time_samples
        self.x_samples = args.x_samples
        self.noise_size = noise_size
        self.fixed_noise = tf.random_normal(np.array([1, self.noise_size, 1, 1]), mean=0.0, stddev=1.0, dtype=tf.float32)
        self.dataset_dir = args.dataset_dir
        self.log_dir = args.log_dir

        self.generator = DCGAN

        OPTIONS = namedtuple('OPTIONS', 'noise_size time_samples x_samples, is_training')
        self.options = OPTIONS._make((args.noise_size, args.time_samples, args.x_samples, args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.input_noise = tf.placeholder(tf.float32,
                                        [None, self.noise_size, 1, 1], name='input_noise')

        self.target = tf.placeholder(tf.float32,
                                        [None, self.x_samples, self.time_samples, 1], name='input_noise')

        self.SNR_diff = tf.placeholder(tf.float32, [None, self.time_samples*self.x_samples], name='SNR_diff')
        self.SNR_real = tf.placeholder(tf.float32, [None, self.time_samples*self.x_samples], name='SNR_real')
        self.Rec_SNR = -20.0* tf.log(tf.norm(self.SNR_diff, ord='euclidean')/tf.norm(self.SNR_real, ord='euclidean'))/tf.log(10.0)
        self.Rec_SNR_sum = tf.summary.scalar("Rec_SNR", self.Rec_SNR)

        self.fake_A = self.generator(self.input_noise, self.options, False, name="DCGAN")

        self.g_loss = mae_criterion(self.fake_A, self.target)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        self.testA = self.generator(self.input_noise, self.options, True, name="DCGAN")

        t_vars = tf.trainable_variables()

        var_size = 0
        for var in t_vars:
            var_size = var_size + int(np.prod(np.array(var.shape)))
        print(("Number of unknowns: %d" % (var_size)))

        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.t_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train and self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for idx in range(self.itr_num):

            lr = args.lr if idx < args.itr_step else args.lr*(self.itr_num - idx)/(self.itr_num - args.itr_step)

            # Update network
            fake_A, _, summary_str = self.sess.run(
                [self.fake_A, self.g_optim, self.g_loss_sum],
                feed_dict={self.input_noise: self.fixed_noise, self.lr: lr})
            self.writer.add_summary(summary_str, counter)

            counter += 1
            print(("Iteration:[%4d/%4d] time: %4.4f" % (
                int(idx), int(self.itr_num), time.time() - start_time)))

            if np.mod(counter, args.print_freq) == 1:
                self.sample_model(args.sample_dir, epoch, idx, counter-1)

            if np.mod(counter, int(floor(args.save_freq))) == 2:
                self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size0)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.noise_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False



    def sample_model(self, sample_dir, epoch, idx, counter, mask=None):

        out_var = self.testA

        fake_A = self.sess.run(out_var, feed_dict={self.input_noise: self.fixed_noise})

        result_img = np.zeros(np.shape(fake_A), dtype=np.float32)
        result_img = fake_A
        diff_img = np.absolute(self.target - result_img)

        diff_img = diff_img.reshape((1, self.time_samples*self.x_samples))
        reshaped_target = self.target.reshape((1, self.time_samples*self.x_samples))

        Rec_SNR, summary_str = self.sess.run(
            [self.Rec_SNR, self.Rec_SNR_sum],
            feed_dict={self.SNR_diff: diff_img,
                       self.SNR_real: reshaped_target})
        self.writer.add_summary(summary_str, counter)

        print(("Mapping SNR: %4.4f" % (Rec_SNR)))



    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        out_var = self.testA

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        sample_file = 0

        start_time_interp = time.time()
        fake_img = self.sess.run(out_var, feed_dict={self.input_noise: self.fixed_noise})

        print(("Mapping time: %4.4f seconds" % (time.time() - start_time_interp)))

        fake_output={}
        fake_output['result'] = fake_img
        fake_output['original'] = self.target
        fake_output['dispersed'] = self.fixed_noise

        io.savemat('mapping_result{0}_{1}'.format(args.which_direction, os.path.basename(str(sample_file))),fake_output)

        result_img = np.zeros(np.shape(fake_img), dtype=np.float32)
        result_img = fake_img
        diff_img = np.absolute(self.target-result_img)

        diff_img = diff_img.reshape((1, self.time_samples*self.x_samples))
        self.target = self.target.reshape((1, self.time_samples*self.x_samples))

        Rec_SNR = self.sess.run(
            [self.Rec_SNR],
            feed_dict={self.SNR_diff: diff_img,
                       self.SNR_real: self.target})

        print(("Mapping SNR: %4.4f" % (Rec_SNR[0])))


##################################################################################
