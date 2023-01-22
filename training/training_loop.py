# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Main training script."""

import os
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

import config
import train
from training import dataset
from training import misc
from metrics import metric_base

import pickle
import glob
import dnnlib.util as ut

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x = tflib.lerp(x, y, lod - tf.floor(lod))
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    training_set,
    num_gpus,
    lod_initial_resolution  = 4,        # Image resolution used at the beginning.
    lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
    minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
    minibatch_dict          = {},       # Resolution-specific overrides.
    max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
    G_lrate_base            = 0.001,    # Learning rate for the generator.
    G_lrate_dict            = {},       # Resolution-specific overrides.
    D_lrate_base            = 0.001,    # Learning rate for the discriminator.
    D_lrate_dict            = {},       # Resolution-specific overrides.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 160,      # Default interval of progress snapshots.
    tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40}): # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    s.lod = training_set.resolution_log2
    s.lod -= np.floor(np.log2(lod_initial_resolution))
    s.lod -= phase_idx
    if lod_transition_kimg > 0:
        s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
    s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch = minibatch_dict.get(s.resolution, minibatch_base)
    s.minibatch -= s.minibatch % num_gpus
    if s.resolution in max_minibatch_per_gpu:
        s.minibatch = min(s.minibatch, max_minibatch_per_gpu[s.resolution] * num_gpus)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

#----------------------------------------------------------------------------
# Main training script.

def training_loop(
    submit_config,
    G_args                  = {},       # Options for generator network.
    D_zero_args             = {},       # Options for discriminator network.
    D_one_args              = {},
    G_opt_args              = {},       # Options for generator optimizer.
    D_zero_opt_args         = {},       # Options for discriminator optimizer.
    D_one_opt_args          = {},
    G_loss_args             = {},       # Options for generator loss.
    D_zero_loss_args        = {},       # Options for discriminator loss.
    D_one_loss_args         = {},
    dataset_args            = {},       # Options for dataset.load_dataset().
    sched_args              = {},       # Options for train.TrainingSchedule.
    grid_args               = {},       # Options for train.setup_snapshot_image_grid().
    metric_arg_list         = [],       # Options for MetricGroup.
    tf_config               = {},       # Options for tflib.init_tf().
    G_smoothing_kimg        = 10.0,     # Half-life of the running average of generator weights.
    D_repeats               = 1,        # How many times the discriminator is trained per G iteration.
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters.
    reset_opt_for_new_lod   = True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 15000,    # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,    # Enable mirror augment?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 1,        # How often to export image snapshots?
    network_snapshot_ticks  = 3,       # How often to export network snapshots?
    save_tf_graph           = False,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,    # Include weight histograms in the tfevents file?
    resume_run_id           = "/mnt/3/zxy/stylegan10.0/results/00006-sgan-ffhq256-4gpu/network-snapshot-010966.pkl",     # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,     # Snapshot index to resume training from, None = autodetect.
    resume_kimg             = 0.0,      # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0):     # Assumed wallclock time at the beginning. Affects reporting.

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

    ax = np.arange(-21 // 2 + 1., 21 // 2 + 1.).astype(np.float32)
    xx, yy = np.meshgrid(ax, ax)
    xy = -(xx ** 2 + yy ** 2).astype(np.float32)
    xy = tf.convert_to_tensor(xy)

    xy2 = np.zeros((21, 21)).astype(np.float32)
    xy2[10, 10] = 1
    xy2 = tf.convert_to_tensor(xy2)


    # Initialize dnnlib and TensorFlow.
    ctx = dnnlib.RunContext(submit_config, train)
    tflib.init_tf(tf_config)

    # Load training set.

    training_set2 = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **dataset_args)
    training_set1 = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **dataset_args)
    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            G, D_zero, D_one, Gs = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            G = tflib.Network('G', num_channels=training_set1.shape[0], resolution=training_set1.shape[1], label_size=training_set1.label_size, **G_args)
            D_zero = tflib.Network('D_zero', num_channels=training_set1.shape[0], resolution=training_set1.shape[1], label_size=training_set1.label_size, **D_zero_args)
            D_one = tflib.Network('D_one', num_channels=training_set1.shape[0], resolution=training_set1.shape[1], label_size=training_set1.label_size, **D_one_args)
            Gs = G.clone('Gs')
    G.print_layers(); D_zero.print_layers(); D_one.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // submit_config.num_gpus
        Gs_beta         = 0.5 ** tf.div(tf.cast(minibatch_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

    G_opt = tflib.Optimizer(name='TrainG', learning_rate=lrate_in, **G_opt_args)
    D_zero_opt = tflib.Optimizer(name='TrainD_zero', learning_rate=lrate_in, **D_zero_opt_args)
    D_one_opt = tflib.Optimizer(name='TrainD_one', learning_rate=lrate_in, **D_one_opt_args)

    for gpu in range(submit_config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_zero_gpu = D_zero if gpu == 0 else D_zero.clone(D_zero.name + '_shadow')
            D_one_gpu = D_one if gpu == 0 else D_one.clone(D_one.name + '_shadow')
            lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_zero_gpu.find_var('lod'), lod_in), tf.assign(D_one_gpu.find_var('lod'), lod_in)]
            reals1, labels1 = training_set1.get_minibatch_tf()
            reals1 = process_reals(reals1, lod_in, mirror_augment, training_set1.dynamic_range, drange_net)
            reals2, labels2 = training_set2.get_minibatch_tf()
            reals2 = process_reals(reals2, lod_in, mirror_augment, training_set2.dynamic_range, drange_net)

            with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
                G_loss = dnnlib.util.call_func_by_name(G=G_gpu, D_zero=D_zero_gpu, D_one=D_one_gpu, opt=G_opt, training_set=training_set1, minibatch_size=minibatch_split, pca_matrix=pca_matrix, pca_mean=pca_mean, xy=xy, xy2=xy2, **G_loss_args)
            with tf.name_scope('D_zero_loss'), tf.control_dependencies(lod_assign_ops):
                D_zero_loss = dnnlib.util.call_func_by_name(G=G_gpu, D_zero=D_zero_gpu, opt=D_zero_opt, training_set1=training_set1, training_set2=training_set2, minibatch_size=minibatch_split, reals2=reals2, labels2=labels2, pca_matrix=pca_matrix, pca_mean=pca_mean, xy=xy, xy2=xy2, **D_zero_loss_args)
            with tf.name_scope('D_one_loss'), tf.control_dependencies(lod_assign_ops):
                D_one_loss = dnnlib.util.call_func_by_name(G=G_gpu, D_one=D_one_gpu, opt=D_one_opt, training_set1=training_set1, training_set2=training_set2, minibatch_size=minibatch_split, reals1=reals1, labels1=labels1, pca_matrix=pca_matrix, pca_mean=pca_mean, xy=xy, xy2=xy2, **D_one_loss_args)
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_zero_opt.register_gradients(tf.reduce_mean(D_zero_loss), D_zero_gpu.trainables)
            D_one_opt.register_gradients(tf.reduce_mean(D_one_loss), D_one_gpu.trainables)
    G_train_op = G_opt.apply_updates()
    D_zero_train_op = D_zero_opt.apply_updates()
    D_one_train_op = D_one_opt.apply_updates()

    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)

    print('Setting up snapshot image grid...')
    grid_size, grid_reals, grid_labels, grid_latents = misc.setup_snapshot_image_grid(G, training_set1, **grid_args)
    sched = training_schedule(cur_nimg=total_kimg*1000, training_set=training_set1, num_gpus=submit_config.num_gpus, **sched_args)

    # print(pca_matrix)

    prepro = ut.SRMDPreprocessing(4, pca_matrix, pca_mean, random=True, para_input=10,
                                  noise=False,
                                  sig=2.6, sig_min=0.2, sig_max=4.0,
                                  rate_iso=1.0, scaling=3,
                                  rate_cln=0.2, noise_high=0.0)  # random(sig_min, sig_max)
    #print(sched.minibatch//submit_config.num_gpus)

    k_code_one = prepro(xy, xy2, grid_latents.shape[0])
    # k_code = k_code.eval()
    k_code_one = k_code_one.eval()
    #print(k_code_one)

    # (28,3,1024,1024)
    grid_fakes_one = Gs.run(grid_latents, grid_labels, k_code_one, is_validation=True, minibatch_size=sched.minibatch//submit_config.num_gpus, add_zero=False)
    grid_fakes_zero = Gs.run(grid_latents, grid_labels, k_code_one, is_validation=True, minibatch_size=sched.minibatch//submit_config.num_gpus, add_zero=True)

    print('Setting up run dir...')
    misc.save_image_grid(grid_reals, os.path.join(submit_config.run_dir, 'reals.png'), drange=training_set1.dynamic_range, grid_size=grid_size)
    misc.save_image_grid(grid_fakes_one, os.path.join(submit_config.run_dir, 'fakes_one%06d.png' % resume_kimg), drange=drange_net, grid_size=grid_size)
    misc.save_image_grid(grid_fakes_zero, os.path.join(submit_config.run_dir, 'fakes_zero%06d.png' % resume_kimg), drange=drange_net, grid_size=grid_size)
    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D_zero.setup_weight_histograms(); D_one.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training...\n')
    ctx.update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = ctx.get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    while cur_nimg < total_kimg * 1000:
        if ctx.should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set1, num_gpus=submit_config.num_gpus, **sched_args)
        training_set1.configure(sched.minibatch // submit_config.num_gpus, sched.lod)
        training_set2.configure(sched.minibatch // submit_config.num_gpus, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); D_zero_opt.reset_optimizer_state(); D_one_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        for _mb_repeat in range(minibatch_repeats):

            for _D_repeat in range(D_repeats):
                tflib.run([D_zero_train_op, D_one_train_op, Gs_update_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
                cur_nimg += sched.minibatch
            tflib.run([G_train_op], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch})

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = ctx.get_time_since_last_update()
            total_time = ctx.get_time_since_start() + resume_time
            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %-4.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/lod', sched.lod),
                autosummary('Progress/minibatch', sched.minibatch),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save snapshots.
            if cur_tick % image_snapshot_ticks == 0 or done:
                grid_fakes_one = Gs.run(grid_latents, grid_labels, k_code_one, is_validation=True, minibatch_size=sched.minibatch//submit_config.num_gpus, add_zero=False)
                grid_fakes_zero = Gs.run(grid_latents, grid_labels, k_code_one, is_validation=True, minibatch_size=sched.minibatch//submit_config.num_gpus, add_zero=True)
                misc.save_image_grid(grid_fakes_one, os.path.join(submit_config.run_dir, 'fakes_one%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
                misc.save_image_grid(grid_fakes_zero, os.path.join(submit_config.run_dir, 'fakes_zero%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
            if cur_tick % network_snapshot_ticks == 0 or done or cur_tick == 1:
                pkl = os.path.join(submit_config.run_dir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((G, D_zero, D_one, Gs), pkl)
                metrics.run(pkl, run_dir=submit_config.run_dir, num_gpus=submit_config.num_gpus, tf_config=tf_config)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            ctx.update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = ctx.get_last_update_interval() - tick_time

    # Write final results.
    misc.save_pkl((G, D_zero, D_one, Gs), os.path.join(submit_config.run_dir, 'network-final.pkl'))
    summary_log.close()

    ctx.close()

#----------------------------------------------------------------------------
