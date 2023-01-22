# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

from dnnlib import util as ut
import glob
import pickle

#----------------------------------------------------------------------------

class FID(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        # inception = misc.load_pkl('https://drive.google.com/uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn') # inception_v3_features.pkl
        inception = misc.load_pkl('./pkl/inception_v3_features.pkl')
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        pca_file = glob.glob("/mnt/3/zxy/stylegan8.0/PCA.pkl")
        if len(pca_file) == 1:
            pca_file = open(pca_file[0], "rb")
        else:
            raise Exception('Failed to find the PCA file')
        pca_matrix = pickle.load(pca_file)

        pca_mean_file = glob.glob("/mnt/3/zxy/stylegan8.0/PCA_mean.pkl")
        if len(pca_mean_file) == 1:
            pca_mean_file = open(pca_mean_file[0], "rb")
        else:
            raise Exception('Failed to find the PCA_mean file')
        pca_mean = pickle.load(pca_mean_file)

        prepro = ut.SRMDPreprocessing(4, pca_matrix, pca_mean, random=True, para_input=10,
                                      noise=False,
                                      sig=2.6, sig_min=0.2, sig_max=4.0,
                                      rate_iso=1.0, scaling=3,
                                      rate_cln=0.2, noise_high=0.0)  # random(sig_min, sig_max)
        ax = np.arange(-21 // 2 + 1., 21 // 2 + 1.).astype(np.float32)
        xx, yy = np.meshgrid(ax, ax)
        xy = -(xx ** 2 + yy ** 2).astype(np.float32)
        xy = tf.convert_to_tensor(xy)
        xy2 = np.zeros((21, 21))
        xy2[10, 10] = 1
        xy2 = tf.convert_to_tensor(xy2)
        # Calculate statistics for reals.
        cache_file = self._get_cache_file_for_reals(num_images=self.num_images)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.isfile(cache_file):
            mu_real, sigma_real = misc.load_pkl(cache_file)
        else:
            for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
                begin = idx * minibatch_size
                end = min(begin + minibatch_size, self.num_images)
                activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)
                if end == self.num_images:
                    break
            mu_real = np.mean(activations, axis=0)
            sigma_real = np.cov(activations, rowvar=False)
            misc.save_pkl((mu_real, sigma_real), cache_file)

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])

                k_code_one = prepro(xy, xy2, self.minibatch_per_gpu, kernel=False)
                images = Gs_clone.get_output_for(latents, None, k_code_one, is_validation=True, randomize_noise=True, add_zero=False)
                images = tflib.convert_images_to_uint8(images)
                result_expr.append(inception_clone.get_output_for(images))

        # Calculate statistics for fakes.
        for begin in range(0, self.num_images, minibatch_size):
            end = min(begin + minibatch_size, self.num_images)
            activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]
        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)

        # Calculate FID.
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist))

#----------------------------------------------------------------------------
