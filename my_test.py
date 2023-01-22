import pickle
import glob
import dnnlib.util as ut
import tensorflow as tf
from training import dataset
import config
from training import misc
import os
import dnnlib.tflib as tflib
import numpy as np
from PIL import Image

tflib.init_tf({'rnd.np_random_seed': 1000})
resume_run_id= "/home/cma/zxy/styleGAN/stylegan-master/results/00000-sgan-seeprettyface_race_yellow-2gpu"
network_pkl = misc.locate_network_pkl(None, '/mnt/4/zxy/program/stylegan/cache/karras2019stylegan-ffhq-1024x1024.pkl')
print('Loading networks from "%s"...' % network_pkl)
G, D, Gs = misc.load_pkl(network_pkl)

pca_file = glob.glob("/mnt/4/zxy/program/stylegan4.0/pca_matrix.pkl")
if len(pca_file) == 1:
    pca_file = open(pca_file[0], "rb")
else:
    raise Exception('Failed to find the PCA file')
pca_matrix = pickle.load(pca_file)
ax = np.arange(-21 // 2 + 1., 21 // 2 + 1.).astype(np.float32)
xx, yy = np.meshgrid(ax, ax)
xy = -(xx ** 2 + yy ** 2).astype(np.float32)
xy = tf.convert_to_tensor(xy)

xy2 = np.zeros((21, 21)).astype(np.float32)
xy2[10, 10] = 1
xy2 = tf.convert_to_tensor(xy2)
xy2 = tf.reshape(xy2, (1, 21, 21))
b_kernles_one = tf.tile(xy2, [4, 1, 1])
# print(pca_matrix)
prepro = ut.SRMDPreprocessing(4, pca_matrix, random=True, para_input=10,
                              noise=False,
                              sig=2.6, sig_min=0.2, sig_max=4.0,
                              rate_iso=1.0, scaling=3,
                              rate_cln=0.2, noise_high=0.0)  # random(sig_min, sig_max)
k_code, k_code_one, kernel = prepro(xy, xy2, 4, kernel = True)

dataset_args={'tfrecord_dir': '/mnt/4/zxy/program/stylegan/datasets', 'resolution': 1024}
training_set = dataset.load_dataset(data_dir='/mnt/4/zxy/program/stylegan2.0/datasets/', verbose=True, **dataset_args)
latents = tf.random_normal([4] + G.input_shapes[0][1:])
labels = training_set.get_random_labels_tf(4)

fake_images_out = G.get_output_for(latents, labels, is_training=False)
kernel=tf.dtypes.cast(kernel, tf.float32)
# print(kernel.eval())
sr_blured = ut.BatchBlur(fake_images_out, kernel, 4, l=21)
'''
n=tf.reshape(sr_blured[0],(3,1024,1024))
n = n.eval()
img = Image.fromarray(n)
img.show()
'''
misc.save_image_grid(sr_blured.eval(), os.path.join('/mnt/3/zxy/stylegan6.0/results/', 'kb.png'), drange=[-1,1], grid_size=(2,2))
#misc.save_image_grid(fake_images_out.eval(), os.path.join('/mnt/4/zxy/program/stylegan4.0/results/', 'k2.png'), drange=[-1,1], grid_size=(2,2))
#sr_blured = tf.transpose(sr_blured, [0, 2, 3, 1])
#sr_to_lr = tf.image.resize_bilinear(sr_blured, (fake_images_out_one.shape[2]//4, fake_images_out_one.shape[3]//4), align_corners=True)
#lr_s = tf.image.resize_bilinear(sr_to_lr, (fake_images_out_one.shape[2], fake_images_out_one.shape[3]), align_corners=True)
