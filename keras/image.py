'''https://github.com/flystarhe/keras/blob/master/keras/preprocessing/image.py
'''
import re
import requests
import numpy as np
from io import BytesIO
from PIL import Image as pil_image


URL_REGEX = re.compile(r'http://|https://|ftp://')


def load_img(path, grayscale=False, target_size=None):
    '''
    Loads an image into PIL format
    Notes: PIL image has format `(width, height, channel)`
           Numpy array has format `(height, width, channel)`

    # Arguments
        path: Path to image file or url
        grayscale: Boolean, whether to load the image as grayscale
        target_size: `None` or tuple of ints `(height, width)`

    # Returns
        A PIL Image instance or `None`

    # Raises
        ..
    '''
    try:
        if URL_REGEX.match(path):
            response = requests.get(path)
            img = pil_image.open(BytesIO(response.content))
        else:
            img = pil_image.open(path)
    except IOError:
        return None

    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')

    if target_size is not None:
        wh_tuple = (target_size[1], target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)

    return img


def img_to_array(img, data_format='channels_last'):
    '''
    Converts a PIL Image instance to a Numpy array
    Notes: PIL image has format `(width, height, channel)`
           Numpy array has format `(height, width, channel)`

    # Arguments
        img: PIL Image instance
        data_format: Image data format

    # Returns
        A 3D Numpy array, as `(height, width, channel)`

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    '''
    if img is None:
        return None

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format:', data_format)

    x = np.asarray(img, dtype=np.float32)

    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape:', x.shape)

    return x


def array_to_img(x, data_format='channels_last', scale=False):
    '''
    Converts a 3D Numpy array to a PIL Image instance
    Notes: PIL image has format `(width, height, channel)`
           Numpy array has format `(height, width, channel)`

    # Arguments
        x: Input Numpy array
        data_format: Image data format
        scale: Whether to rescale image values to be within `[0, 255]`

    # Returns
        A PIL Image instance

    # Raises
        ValueError: if invalid `x` or `data_format` is passed
    '''
    if x.ndim != 3:
        raise ValueError('Image array to have rank 3.', x.shape)

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number:', x.shape[2])


def is_valid_img(path):
    '''
    Valid Image is ok

    # Arguments
        path: Path to image file

    # Returns
        Boolean

    # Raises
        ..
    '''
    try:
        img = pil_image.open(path)
        img.verify()
    except:
        return False

    if min(img.size) < 1:
        return False

    return True