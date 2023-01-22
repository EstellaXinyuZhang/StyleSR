# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Miscellaneous utility classes and functions."""

import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import uuid
import tensorflow as tf
import torchvision as tv

from distutils.util import strtobool
from typing import Any, List, Tuple, Union


# Util classes
# ------------------------------------------------------------------------------------------


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


# Small util functions
# ------------------------------------------------------------------------------------------


def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def ask_yes_no(question: str) -> bool:
    """Ask the user the question until the user inputs a valid answer."""
    while True:
        try:
            print("{0} [y/n]".format(question))
            return strtobool(input().lower())
        except ValueError:
            pass


def tuple_product(t: Tuple) -> Any:
    """Calculate the product of the tuple elements."""
    result = 1

    for v in t:
        result *= v

    return result


_str_to_ctype = {
    "uint8": ctypes.c_ubyte,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_byte,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double
}


def get_dtype_and_ctype(type_obj: Any) -> Tuple[np.dtype, Any]:
    """Given a type name string (or an object having a __name__ attribute), return matching Numpy and ctypes types that have the same size in bytes."""
    type_str = None

    if isinstance(type_obj, str):
        type_str = type_obj
    elif hasattr(type_obj, "__name__"):
        type_str = type_obj.__name__
    elif hasattr(type_obj, "name"):
        type_str = type_obj.name
    else:
        raise RuntimeError("Cannot infer type name from input")

    assert type_str in _str_to_ctype.keys()

    my_dtype = np.dtype(type_str)
    my_ctype = _str_to_ctype[type_str]

    assert my_dtype.itemsize == ctypes.sizeof(my_ctype)

    return my_dtype, my_ctype


def is_pickleable(obj: Any) -> bool:
    try:
        with io.BytesIO() as stream:
            pickle.dump(obj, stream)
        return True
    except:
        return False


# Functionality to import modules/objects by name, and call functions by name
# ------------------------------------------------------------------------------------------

def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:
    """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    assert is_top_level_function(obj)
    return obj.__module__ + "." + obj.__name__


# File system helpers
# ------------------------------------------------------------------------------------------

def list_dir_recursively_with_ignore(dir_path: str, ignores: List[str] = None, add_base_to_relative: bool = False) -> List[Tuple[str, str]]:
    """List all files recursively in a given directory while ignoring given file and directory names.
    Returns list of tuples containing both absolute and relative paths."""
    assert os.path.isdir(dir_path)
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        assert len(absolute_paths) == len(relative_paths)
        result += zip(absolute_paths, relative_paths)

    return result


def copy_files_and_create_dirs(files: List[Tuple[str, str]]) -> None:
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])


# URL helpers
# ------------------------------------------------------------------------------------------

def is_url(obj: Any) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert is_url(url)
    assert num_attempts >= 1

    # Lookup from cache.
    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache_dir is not None:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            return open(cache_files[0], "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive quota exceeded")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache_dir is not None:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic

    # Return data as file object.
    return io.BytesIO(url_data)

# =========================================================================================
from scipy import signal
import math
from PIL import Image
import collections
try:
    import accimage
except ImportError:
    accimage = None


def PCA(data, k=2):
    with tf.device('/gpu:0'):
        # X = torch.from_numpy(data)
        X = tf.cast(tf.convert_to_tensor(data), dtype=tf.float32)
        # X_mean = torch.mean(X, 0)
        X_mean = tf.reduce_mean(X, 0)
        X = X - X_mean.expand_as(X)
        # U, S, V = torch.svd(torch.t(X))
        U, S, V = tf.svd(tf.transpose(X))
    return U[:, :k] # PCA matrix

def cal_sigma(sig_x, sig_y, radians):
    with tf.device('/gpu:0'):
        D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]]).astype(np.float32)
        U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]]).astype(np.float32)
        sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    with tf.device('/gpu:0'):
        ax = np.arange(-l // 2 + 1., l // 2 + 1.).astype(np.float32)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
        # return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)
    return tf.convert_to_tensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    with tf.device('/gpu:0'):
        ax = np.arange(-l // 2 + 1., l // 2 + 1.).astype(np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    # return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)
    return tf.convert_to_tensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, scaling=3, l=21, tensor=False):
    with tf.device('/gpu:0'):
        pi = np.random.random() * math.pi * 2 - math.pi
        x = np.random.random() * (sig_max - sig_min) + sig_min
        y = np.clip(np.random.random().astype(np.float32) * scaling * x, sig_min, sig_max)
        sig = cal_sigma(x, y, pi)
        k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(xy, sig_min=0.2, sig_max=4.0, l=21, tensor=False):
    with tf.device('/gpu:0'):
        #x = tf.random_uniform(shape=(), minval=0, maxval=1) * (sig_max - sig_min) + sig_min
        x = np.random.random() * (sig_max - sig_min) + sig_min
        #k = tf.exp(xy/(2. * x ** 2))
        k = np.exp(xy / (2. * x ** 2))
        #k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def random_gaussian_kernel(xy, l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=False ):
    with tf.device('/gpu:0'):
        return random_isotropic_gaussian_kernel(xy)


def random_batch_kernel(xy, batch, l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=True):
    with tf.device('/gpu:0'):
        #batch_kernel = tf.zeros((batch, l, l), dtype=tf.float32)
        # batch_kernel = tf.zeros([batch]+[l,l])
        #def isotropic_gaussian_kernel(l, sigma, tensor=False):

        '''
        batch_kernel = random_gaussian_kernel(xy, l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso,
                                                scaling=scaling, tensor=False)
        batch_kernel = tf.expand_dims(batch_kernel, 0)
        #isotropic_gaussian_kernel=tf.convert_to_tensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)
        for i in range(batch-1):
            batch_kernel_i = random_gaussian_kernel(xy, l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso,
                                                     scaling=scaling, tensor=False)
            batch_kernel_i = tf.expand_dims(batch_kernel_i, 0)
            batch_kernel = tf.concat([batch_kernel, batch_kernel_i], axis = 0)
        '''

        batch_kernel = tf.zeros((int(batch), l, l), dtype=tf.float32)
        '''
        xy = tf.tile(xy, (int(batch), 1, 1))
        sigma = tf.random_uniform((int(batch),1), minval=0, maxval=1) * (sig_max - sig_min) + sig_min
        sigma = tf.reshape(sigma, (int(batch), 1, 1))
        xy = tf.multiply (xy , (2. * sigma ** 2))
        # print(xy.eval())
        kernel = tf.exp(xy)
        sum = tf.reduce_sum(kernel, [1,2])
        sum = tf.reshape(sum, (int(batch), 1, 1))
        batch_kernel = kernel / sum
        '''

        # return torch.FloatTensor(batch_kernel) if tensor else batch_kernel

    return tf.convert_to_tensor(batch_kernel) if tensor else batch_kernel


def random_batch_noise(batch, high, rate_cln=1.0):
    with tf.device('/gpu:0'):
        noise_level = np.random.uniform(size=(batch, 1)) * high
        noise_mask = np.random.uniform(size=(batch, 1))
        noise_mask[noise_mask < rate_cln] = 0
        noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask

# ????
def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    with tf.device('/gpu:0'):
        if noise_size is None:
            size = tensor.size()
        else:
            size = noise_size
        # noise = torch.mul(torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
        noise = tf.multiply(tf.convert_to_tensor(np.random.normal(loc=mean, scale=1.0, size=size)),
                            sigma.view(sigma.size() + (1, 1)))
        # return torch.clamp(noise + tensor, min=min, max=max)
    return tf.clip_by_value(noise + tensor, min=min, max=max)


def BatchSRKernel(xy, batch, tensor=False, l=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3):
    with tf.device('/gpu:0'):
        #batch_kernel = tf.zeros((int(batch), l, l))
        #batch_kernel = batch
        '''
        # batch_kernel = tf.zeros([batch]+[l,l])
        for i in range(int(batch)):
            batch_kernel[i] = random_gaussian_kernel(xy, tensor=False)
        '''
        xy = xy * 2.0
        #xy = tf.reshape(xy, (1 ,l, l))
        '''
        xy = tf.tile(xy, (int(batch), 1, 1))
        sigma = tf.random_uniform((int(batch),1), minval=0, maxval=1) * (sig_max - sig_min) + sig_min
        sigma = tf.reshape(sigma, (int(batch), 1, 1))
        xy = tf.multiply (xy , (2. * sigma ** 2))
        # print(xy.eval())
        kernel = tf.exp(xy)
        sum = tf.reduce_sum(kernel, [1,2])
        sum = tf.reshape(sum, (int(batch), 1, 1))
        batch_kernel = kernel / sum
        '''
        return xy
        # return torch.FloatTensor(batch_kernel) if tensor else batch_kernel

    # return tf.convert_to_tensor(batch_kernel) if tensor else batch_kernel


class PCAEncoder(object):
    def __init__(self, weight):
        self.weight = weight #[l^2, k]
        self.size = self.weight.size()

        # self.weight = Variable(self.weight)
        # self.weight = tf.Variable(self.weight)
    def __call__(self, batch_kernel):
        with tf.device('/gpu:0'):
            B, H, W = batch_kernel.shape  # [B, l, l]
            batch_kernel = tf.cast(batch_kernel, tf.float32)
        # return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))
        return tf.reshape(tf.matmul(tf.reshape(batch_kernel,(B, 1, H * W)), self.weight.expand((B, ) + self.size)), (B, -1))



def BatchBlur(input, kernel, B, l=15):
    with tf.device('/gpu:0'):
        if l % 2 == 1:
            # self.pad = nn.ReflectionPad2d(l // 2)
            pad = tf.pad(input, [[0, 0], [0, 0], [l // 2, l // 2], [l // 2, l // 2]], mode='REFLECT')
        else:
            # self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
            pad = tf.pad(input, [[0, 0], [0, 0], [l // 2, l // 2 - 1], [l // 2, l // 2 - 1]], mode='REFLECT')

        b, C, H, W = input.shape
        H_p, W_p = pad.shape[-2:]

        # input_CBHW = pad.view((1, C * B, H_p, W_p))
        # B=1, C=C*B, H=H_p, W_p
        input_CBHW = tf.reshape(pad, (1, C * B, H_p, W_p))
        # kernel_var = kernel.contiguous().view((B, 1, l, l)).repeat(1, C, 1, 1).view((B * C, 1, l, l))
        kernel_var = tf.reshape(kernel, (B, 1, l, l))
        kernel_var = tf.tile(kernel_var, [1, C, 1, 1])  # (B, C, l, l)
        kernel_var = tf.reshape(kernel_var, (B * C, 1, l, l))
        kernel_var = tf.transpose(kernel_var, [2, 3, 0, 1])
        # pytorch (out_channels, in_channels/groups, kH, kW):(B * C, 1, l, l)
        # tf: [filter_height, filter_width, in_channels, channel_multiplier], out_channels=in_channels * channel_multiplier
        # return F.conv2d(input_CBHW, kernel_var, groups=B*C).view((B, C, H, W))
        # input(1, B * C, H_p, W_p),  output (1, B * C, H, W)
    return tf.reshape(tf.nn.depthwise_conv2d(input_CBHW, kernel_var, padding='VALID',strides=[1,1,1,1], data_format='NCHW'), (B, C, H, W))



def SRMDPreprocessing(xy, B, kernel=False, scale=4, para_input=10, l=21, noise=False, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, rate_cln=0.2, noise_high=0.0):
    # self.encoder = PCAEncoder(pca)
    #self.kernel_gen = BatchSRKernel(l=kernel, sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling)
    # with tf.device('/gpu:0'):
    # with tf.name_scope('Bkernel'):

    b_kernels = BatchSRKernel(xy, B, tensor=True)

    # kernel encode  PCA
    # with tf.name_scope('KernelCode'):
    # kernel_code = tf.cast(self.encoder(b_kernels), dtype=tf.float32) # B x self.para_input
    # Noisy
    '''
    if noise:
        # Noise_level = torch.FloatTensor(random_batch_noise(B, self.noise_high, self.rate_cln))
        Noise_level = tf.convert_to_tensor(random_batch_noise(B, noise_high,rate_cln))
    else:
        Noise_level = tf.zeros((B, 1))
    '''

        # Noise_level = tf.Variable(Noise_level)
        # with tf.name_scope('ReCode'):
        # re_code = tf.concat([kernel_code, Noise_level * 10], dim=1) if self.noise else kernel_code


        #return (re_code, b_kernels) if kernel else re_code
    return b_kernels
