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

from sklearn.decomposition import PCA

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

def _is_tensor_image(img):
    # return torch.is_tensor(img) and img.ndimension() == 3
    return tf.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    # if isinstance(pic, torch.FloatTensor):
    if isinstance(pic, tf.Tensor):
        pic = pic.mul(255).byte()
    if tf.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def isogkern(kernlen, std):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d = gkern2d/np.sum(gkern2d)
    return gkern2d


def anisogkern(kernlen, std1, std2, angle):
    gkern1d_1 = signal.gaussian(kernlen, std=std1).reshape(kernlen, 1)
    gkern1d_2 = signal.gaussian(kernlen, std=std2).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d_1, gkern1d_2)
    gkern2d = gkern2d/np.sum(gkern2d)
    return gkern2d


def PCA(data, k=2):
    # X = torch.from_numpy(data)
    X = tf.convert_to_tensor(data)
    # X_mean = torch.mean(X, 0)
    X_mean = tf.reduce_mean(X, 0)
    X = X - X_mean.expand_as(X)
    # U, S, V = torch.svd(torch.t(X))
    U, S, V = tf.svd(tf.transpose(X))
    return U[:, :k] # PCA matrix

def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    # return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)
    return tf.convert_to_tensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    # return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)
    return tf.convert_to_tensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, scaling=3, l=21, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, l=21, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def stable_isotropic_gaussian_kernel(sig=2.6, l=21, tensor=False):
    x = sig
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def random_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)


def stable_gaussian_kernel(l=21, sig=2.6, tensor=False):
    return stable_isotropic_gaussian_kernel(sig=sig, l=l, tensor=tensor)


def random_batch_kernel(batch, l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=True):

    batch_kernel = np.zeros((batch, l, l))
    # batch_kernel = tf.zeros([batch]+[l,l])
    for i in range(batch):
        batch_kernel[i] = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
    # return torch.FloatTensor(batch_kernel) if tensor else batch_kernel
    return tf.convert_to_tensor(batch_kernel) if tensor else batch_kernel


def stable_batch_kernel(batch, l=21, sig=2.6, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = stable_gaussian_kernel(l=l, sig=sig, tensor=False)
    # return torch.FloatTensor(batch_kernel) if tensor else batch_kernel
    return tf.convert_to_tensor(batch_kernel) if tensor else batch_kernel

def b_GPUVar_Bicubic(variable, scale):
    tensor = variable.cpu().data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_view = tensor.view((B*C, 1, H, W))
    # re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    re_tensor = tf.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_view[i])
        tran = tv.transforms.ToTensor()
        re_tensor[i] = tran(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_view = re_tensor.view((B, C, H_new, W_new))
    # re_tensor_view = tf.image.resize_images(tensor, [H_new, W_new], method=tf.image.ResizeMethod.BICUBIC)
    return re_tensor_view


def b_CPUVar_Bicubic(variable, scale):
    tensor = variable.data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)

    tensor_v = tensor.view((B*C, 1, H, W))
    re_tensor = tf.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_v[i])
        tran = tv.transforms.ToTensor()
        re_tensor[i] = tran(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_v = re_tensor.view((B, C, H_new, W_new))

    # re_tensor_v = tf.image.resize_images(tensor, [H_new, W_new], method=tf.image.ResizeMethod.BICUBIC)
    return re_tensor_v


def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask

# ????
def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    # noise = torch.mul(torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    noise = tf.multiply(tf.convert_to_tensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    # return torch.clamp(noise + tensor, min=min, max=max)
    return tf.clip_by_value(noise + tensor, min=min, max=max)


class BatchSRKernel(object):
    def __init__(self, l=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling

    def __call__(self, xy, batch, tensor=False):
        xy = tf.reshape(xy, (1 ,self.l, self.l))
        xy = tf.tile(xy, [batch]+[1,1])
        sigma = tf.random_uniform([batch]+[1,1], minval=0, maxval=1) * (self.sig_max - self.sig_min) + self.sig_min
        '''
        sigma = tf.expand_dims(sigma, -1)
        '''
        xy = tf.multiply(xy, 1/(2. * sigma ** 2))
        kernel = tf.exp(xy)
        sum = tf.reduce_sum(kernel, [1, 2])
        sum = tf.reshape(sum, (-1,1,1))
        '''
        sum = tf.expand_dims(sum, -1)       
        sum = tf.expand_dims(sum, -1)
        '''
        batch_kernel = kernel / sum
        return batch_kernel


class PCAEncoder(object):
    def __init__(self, weight, mean):
        self.weight = weight  # (10, 441)
        self.mean = mean
        # self.size = self.weight.size()

        # self.weight = Variable(self.weight)
        # self.weight = tf.Variable(self.weight)
    def __call__(self, B, batch_kernel):
        '''
        b, H, W = batch_kernel.shape #[B, l, l]
        batch_kernel = tf.cast(batch_kernel, tf.float32)
        weight = tf.expand_dims(self.weight, 0)
        weight = tf.tile(weight, [B]+[1,1])
        # return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))
        return tf.reshape(tf.matmul(tf.reshape(batch_kernel,(-1, 1, H * W)), weight), (B, -1))
        '''
        b, H, W = batch_kernel.shape  # [B, l, l]
        batch_kernel = tf.cast(batch_kernel, tf.float32)
        batch_kernel = tf.reshape(batch_kernel, (-1, H*W))
        batch_kernel = batch_kernel - self.mean
        weight = tf.transpose(self.weight)
        weight = tf.expand_dims(weight, 0)
        weight = tf.tile(weight, [B] + [1, 1])  # (B, 441, 10)
        return tf.reshape(tf.matmul(tf.reshape(batch_kernel, (-1, 1, H * W)), weight), (B, -1))




def BatchBlur(input, kernel, B, l=15):
    if l % 2 == 1:
        # self.pad = nn.ReflectionPad2d(l // 2)
        pad = tf.pad(input, [[0,0],[0,0],[l // 2,l // 2],[l // 2,l // 2]], mode='REFLECT')
    else:
        # self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        pad = tf.pad(input, [[0,0],[0,0],[l // 2,l // 2 - 1],[l // 2,l // 2 - 1]], mode='REFLECT')

    b, C, H, W = input.shape
    H_p, W_p = pad.shape[-2:]

    # input_CBHW = pad.view((1, C * B, H_p, W_p))
    # B=1, C=C*B, H=H_p, W_p
    input_CBHW = tf.reshape(pad, (1, C * B, H_p, W_p))
    # kernel_var = kernel.contiguous().view((B, 1, l, l)).repeat(1, C, 1, 1).view((B * C, 1, l, l))
    kernel_var = tf.reshape(kernel, (B, 1, l, l))
    kernel_var = tf.tile(kernel_var, [1, C, 1, 1])     # (B, C, l, l)
    kernel_var = tf.reshape(kernel_var, (B * C, 1, l, l))
    kernel_var = tf.transpose(kernel_var, [2, 3, 0, 1])
    # pytorch (out_channels, in_channels/groups, kH, kW):(B * C, 1, l, l)
    # tf: [filter_height, filter_width, in_channels, channel_multiplier], out_channels=in_channels * channel_multiplier
    # return F.conv2d(input_CBHW, kernel_var, groups=B*C).view((B, C, H, W))
    # input(1, B * C, H_p, W_p),  output (1, B * C, H, W)
    return tf.reshape(tf.nn.depthwise_conv2d(input_CBHW, kernel_var, padding='VALID',strides=[1,1,1,1], data_format='NCHW'), (B, C, H, W))


class SRMDPreprocessing(object):
    def __init__(self, scale, pca, mean, random, para_input=10, kernel=21, noise=True, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, rate_cln=0.2, noise_high=0.08):
        self.encoder = PCAEncoder(pca, mean)
        self.kernel_gen = BatchSRKernel(l=kernel, sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling)
        # self.blur = BatchBlur(l=kernel)
        self.para_in = para_input
        self.l = kernel    # kernel size
        self.noise = noise
        self.scale = scale
        self.rate_cln = rate_cln
        self.noise_high = noise_high
        self.random = random

    def __call__(self, xy, xy2, B, kernel=False):
        b_kernels = self.kernel_gen(xy, B, tensor=True)
        xy2 = tf.reshape(xy2, (1, self.l, self.l))
        b_kernles_one = tf.tile(xy2, [B, 1, 1])
        # kernel encode  PCA
        kernel_code = self.encoder(B, b_kernels) # B x self.para_input
        kernel_code_one = self.encoder(B, b_kernles_one)
        # Noisy
        if self.noise:
            # Noise_level = torch.FloatTensor(random_batch_noise(B, self.noise_high, self.rate_cln))
            Noise_level = tf.convert_to_tensor(random_batch_noise(B, self.noise_high, self.rate_cln))
        else:
            Noise_level = tf.zeros((B, 1))

        # Noise_level = tf.Variable(Noise_level)
        re_code = tf.concat([kernel_code, Noise_level * 10], dim=1) if self.noise else kernel_code

        return (kernel_code_one, b_kernels) if kernel else kernel_code_one
