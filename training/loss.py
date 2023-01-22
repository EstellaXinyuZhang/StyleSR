# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Loss functions."""

import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import pickle
import glob
import dnnlib.util as ut
import os
from training import misc
#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# WGAN & WGAN-GP loss functions.

def G_wgan(G, D, opt, training_set, minibatch_size): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -fake_scores_out
    return loss

def D_wgan(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_epsilon = 0.001): # Weight for the epsilon term, \epsilon_{drift}.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss

def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss

#----------------------------------------------------------------------------
# Hinge loss functions. (Use G_wgan with these)

def D_hinge(G, D, opt, training_set, minibatch_size, reals, labels): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.maximum(0., 1.+fake_scores_out) + tf.maximum(0., 1.-real_scores_out)
    return loss

def D_hinge_gp(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.maximum(0., 1.+fake_scores_out) + tf.maximum(0., 1.-real_scores_out)

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    return loss


#----------------------------------------------------------------------------
# Loss functions advocated by the paper
# "Which Training Methods for GANs do actually Converge?"

def G_logistic_saturating(G, D, opt, training_set, minibatch_size): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -tf.nn.softplus(fake_scores_out)  # log(1 - logistic(fake_scores_out))
    return loss
# use this
def G_logistic_nonsaturating(G, D_zero, D_one, opt, training_set, minibatch_size, pca_matrix, pca_mean, xy, xy2, mse_weight=1.0): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    prepro = ut.SRMDPreprocessing(4, pca_matrix, pca_mean, random=True, para_input=10,
                                  noise=False,
                                  sig=2.6, sig_min=0.2, sig_max=4.0,
                                  rate_iso=1.0, scaling=3,
                                  rate_cln=0.2, noise_high=0.0)  # random(sig_min, sig_max)
    k_code_one, kernel = prepro(xy, xy2, minibatch_size, kernel=True)
    #k_code_zero = tf.zeros([minibatch_size] + [10])

    fake_images_out_zero = G.get_output_for(latents, labels, k_code_one, is_training=True, add_zero=True)
    fake_scores_out_zero = fp32(D_zero.get_output_for(fake_images_out_zero, labels, is_training=True))

    fake_images_out_one = G.get_output_for(latents, labels, k_code_one, is_training=True, add_zero=False)
    fake_scores_out_one = fp32(D_one.get_output_for(fake_images_out_one, labels, is_training=True))

    '''
    sr_blured = ut.BatchBlur(fake_images_out_one, kernel, minibatch_size, l=21)
    '''
    size = G.output_shapes[0][2]
    if size < 64:
        lr = fake_images_out_one    # without blur
    elif size == 128:
        sr_withoutblured = tf.transpose(fake_images_out_one, [0, 2, 3, 1])
        sr_to_lr = tf.image.resize_bilinear(sr_withoutblured,(fake_images_out_one.shape[2] // 2, fake_images_out_one.shape[3] // 2), align_corners=True)
        lr_s = tf.image.resize_bilinear(sr_to_lr, (fake_images_out_one.shape[2], fake_images_out_one.shape[3]), align_corners=True)
        lr = tf.transpose(lr_s, [0, 3, 1, 2])
    elif size == 256:
        sr_withoutblured = tf.transpose(fake_images_out_one, [0, 2, 3, 1])
        sr_to_lr = tf.image.resize_bilinear(sr_withoutblured,(fake_images_out_one.shape[2] // 4, fake_images_out_one.shape[3] // 4), align_corners=True)
        lr_s = tf.image.resize_bilinear(sr_to_lr, (fake_images_out_one.shape[2], fake_images_out_one.shape[3]), align_corners=True)
        lr = tf.transpose(lr_s, [0, 3, 1, 2])
    else:
        lr = fake_images_out_one

    mse_loss = tf.losses.mean_squared_error(lr, fake_images_out_zero)
    mse_loss = autosummary('Loss/scores/mse_loss', mse_loss)

    gan_loss = tf.nn.softplus(-(fake_scores_out_zero + fake_scores_out_one))  # -log(logistic(fake_scores_out))

    mse_loss_gradient_zero = tf.reduce_sum(mse_loss)
    mse_loss_gradient_zero = fp32(tf.gradients(mse_loss_gradient_zero, [fake_images_out_zero])[0])
    mse_loss_gradient_zero = tf.reduce_sum(tf.abs(mse_loss_gradient_zero), axis=[1,2,3])
    # mse_loss_gradient = tf.reduce_sum(fp32(tf.gradients(tf.reduce_sum(mse_loss), [fake_images_out_zero])[0]), axis=[1,2,3])
    mse_loss_gradient_zero = autosummary('Gradients/mse_zero_loss', mse_loss_gradient_zero)

    #fake_zero_gradient = tf.reduce_sum(fp32(tf.gradients(tf.reduce_sum(gan_loss), [fake_images_out_zero])[0]), axis=[1,2,3])
    fake_zero_gradient = tf.reduce_sum(gan_loss)
    fake_zero_gradient = fp32(tf.gradients(fake_zero_gradient, [fake_images_out_zero])[0])
    fake_zero_gradient = tf.reduce_sum(tf.abs(fake_zero_gradient), axis=[1, 2, 3])
    fake_zero_gradient = autosummary('Gradients/fake_zero_loss', fake_zero_gradient)

    #mse_loss_gradient = tf.reduce_sum(fp32(tf.gradients(tf.reduce_sum(mse_loss), [fake_images_out_one])[0]), axis=[1,2,3])
    mse_loss_gradient_one = tf.reduce_sum(mse_loss)
    mse_loss_gradient_one = fp32(tf.gradients(mse_loss_gradient_one, [lr])[0])
    mse_loss_gradient_one = tf.reduce_sum(tf.abs(mse_loss_gradient_one), axis=[1, 2, 3])
    # mse_loss_gradient = tf.reduce_sum(fp32(tf.gradients(tf.reduce_sum(mse_loss), [fake_images_out_zero])[0]), axis=[1,2,3])
    mse_loss_gradient_one = autosummary('Gradients/mse_one_loss', mse_loss_gradient_one)

    #fake_one_gradient = tf.reduce_sum(fp32(tf.gradients(tf.reduce_sum(gan_loss), [fake_images_out_one])[0]), axis=[1,2,3])
    #autosummary('Gradients/fake_one_loss', fake_one_gradient)

    fake_one_gradient = tf.reduce_sum(gan_loss)
    fake_one_gradient = fp32(tf.gradients(fake_one_gradient, [fake_images_out_one])[0])
    fake_one_gradient = tf.reduce_sum(tf.abs(fake_one_gradient), axis=[1, 2, 3])
    fake_one_gradient = autosummary('Gradients/fake_one_loss', fake_one_gradient)

    loss = gan_loss + mse_weight*mse_loss + 0.0 * (mse_loss_gradient_one + mse_loss_gradient_zero + fake_one_gradient + fake_zero_gradient)
    return loss

def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out)  # -log(1 - logistic(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type
    return loss
# use this
def D_logistic_simplegp_zero(G, D_zero, opt, training_set1, training_set2, minibatch_size, reals2, labels2, pca_matrix, pca_mean, xy, xy2, r1_gamma=10.0, r2_gamma=0.0): # pylint: disable=unused-argument
    prepro = ut.SRMDPreprocessing(4, pca_matrix, pca_mean, random=True, para_input=10,
                                  noise=False,
                                  sig=2.6, sig_min=0.2, sig_max=4.0,
                                  rate_iso=1.0, scaling=3,
                                  rate_cln=0.2, noise_high=0.0)  # random(sig_min, sig_max)
    k_code_one, kernel = prepro(xy, xy2, minibatch_size, kernel=True)
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])

    reals2.set_shape(D_zero.input_shapes[0])
    fake_images_out_zero = G.get_output_for(latents, labels2, k_code_one, is_training=True, add_zero=True)
    # reals_blured = ut.BatchBlur(reals2, kernel, minibatch_size, l=21)
    size = G.output_shapes[0][2]
    if size < 64:
        reals_lr = reals2    # without blur
    elif size == 128:
        reals_t = tf.transpose(reals2, [0, 2, 3, 1])
        reals_lr = tf.image.resize_bilinear(reals_t, (fake_images_out_zero.shape[2] // 2, fake_images_out_zero.shape[3] // 2), align_corners=True)
        reals_lr = tf.image.resize_bilinear(reals_lr, (fake_images_out_zero.shape[2], fake_images_out_zero.shape[3]), align_corners=True)
        reals_lr = tf.transpose(reals_lr, [0, 3, 1, 2])
    elif size == 256:
        reals_t = tf.transpose(reals2, [0, 2, 3, 1])
        reals_lr = tf.image.resize_bilinear(reals_t, (fake_images_out_zero.shape[2] // 4, fake_images_out_zero.shape[3] // 4), align_corners=True)
        reals_lr = tf.image.resize_bilinear(reals_lr, (fake_images_out_zero.shape[2], fake_images_out_zero.shape[3]), align_corners=True)
        reals_lr = tf.transpose(reals_lr, [0, 3, 1, 2])
    else:
        reals_lr = reals2

    real_scores_out_zero = fp32(D_zero.get_output_for(reals_lr, labels2, is_training=True))
    fake_scores_out_zero = fp32(D_zero.get_output_for(fake_images_out_zero, labels2, is_training=True))
    real_scores_out_zero = autosummary('Loss/scores/real_zero', real_scores_out_zero)
    fake_scores_out_zero = autosummary('Loss/scores/fake_zero', fake_scores_out_zero)

    loss = tf.nn.softplus(fake_scores_out_zero)  # -log(1 - logistic(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out_zero)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type


    if r1_gamma != 0.0:
        with tf.name_scope('R1Penalty'):
            real_loss_zero = opt.apply_loss_scaling(tf.reduce_sum(real_scores_out_zero))
            real_grads_zero = opt.undo_loss_scaling(fp32(tf.gradients(real_loss_zero, [reals_lr])[0]))
            r1_penalty_zero = tf.reduce_sum(tf.square(real_grads_zero), axis=[1, 2, 3])
            r1_penalty_zero = autosummary('Loss/r1_penalty_zero', r1_penalty_zero)

        loss += r1_penalty_zero * (r1_gamma * 0.5)


    if r2_gamma != 0.0:
        with tf.name_scope('R2Penalty'):
            fake_loss_zero = opt.apply_loss_scaling(tf.reduce_sum(fake_scores_out_zero))
            fake_grads_zero = opt.undo_loss_scaling(fp32(tf.gradients(fake_loss_zero, [fake_images_out_zero])[0]))
            r2_penalty_zero = tf.reduce_sum(tf.square(fake_grads_zero), axis=[1, 2, 3])
            r2_penalty_zero = autosummary('Loss/r2_penalty_zero', r2_penalty_zero)

        loss += r2_penalty_zero * (r2_gamma * 0.5)

    return loss

def D_logistic_simplegp_one(G, D_one, opt, training_set1, training_set2, minibatch_size, reals1, labels1, pca_matrix, pca_mean, xy, xy2, r1_gamma=10.0, r2_gamma=0.0): # pylint: disable=unused-argument
    prepro = ut.SRMDPreprocessing(4, pca_matrix,  pca_mean, random=True, para_input=10,
                                  noise=False,
                                  sig=2.6, sig_min=0.2, sig_max=4.0,
                                  rate_iso=1.0, scaling=3,
                                  rate_cln=0.2, noise_high=0.0)  # random(sig_min, sig_max)
    k_code_one, kernel = prepro(xy, xy2, minibatch_size, kernel=True)
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out_one = G.get_output_for(latents, labels1, k_code_one, is_training=True, add_zero=False)
    real_scores_out_one = fp32(D_one.get_output_for(reals1, labels1, is_training=True))
    fake_scores_out_one = fp32(D_one.get_output_for(fake_images_out_one, labels1, is_training=True))
    real_scores_out_one = autosummary('Loss/scores/real_one', real_scores_out_one)
    fake_scores_out_one = autosummary('Loss/scores/fake_one', fake_scores_out_one)


    loss = tf.nn.softplus(fake_scores_out_one)  # -log(1 - logistic(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out_one)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type

    if r1_gamma != 0.0:
        with tf.name_scope('R1Penalty'):
            real_loss_one = opt.apply_loss_scaling(tf.reduce_sum(real_scores_out_one))
            real_grads_one = opt.undo_loss_scaling(fp32(tf.gradients(real_loss_one, [reals1])[0]))
            r1_penalty_one = tf.reduce_sum(tf.square(real_grads_one), axis=[1,2,3])
            r1_penalty_one = autosummary('Loss/r1_penalty_one', r1_penalty_one)

        loss += (r1_penalty_one) * (r1_gamma * 0.5)


    if r2_gamma != 0.0:
        with tf.name_scope('R2Penalty'):
            fake_loss_one = opt.apply_loss_scaling(tf.reduce_sum(fake_scores_out_one))
            fake_grads_one = opt.undo_loss_scaling(fp32(tf.gradients(fake_loss_one, [fake_images_out_one])[0]))
            r2_penalty_one = tf.reduce_sum(tf.square(fake_grads_one), axis=[1,2,3])
            r2_penalty_one = autosummary('Loss/r2_penalty_one', r2_penalty_one)

        loss += (r2_penalty_one) * (r2_gamma * 0.5)

    return loss

#----------------------------------------------------------------------------
