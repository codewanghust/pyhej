import os
import re
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


ref_path = os.path.dirname(__file__)
ref_path = os.path.join(ref_path, '..')
ref_path = os.path.abspath(ref_path)


def_font = ImageFont.truetype(os.path.join(ref_path, 'fonts/DENG.TTF'))
def_color = (0, 0, 0)


URL_REGEX = re.compile(r'http://|https://|ftp://')


def load_img(path, mode=None, target_size=None):
    '''Loads an image into PIL format
    Notes: PIL image has format `(width, height, channel)`
           Numpy array has format `(height, width, channel)`

    # Arguments
        path: Path to image file or url
        mode: String, must in {None, 'L', 'RGB', 'YCbCr'}
        target_size: `None` or tuple of ints `(height, width)`

    # Returns
        A PIL Image instance or `None`

    # Raises
        ..
    '''
    assert mode in {None, 'L', 'RGB', 'YCbCr'}, "mode must in {None, 'L', 'RGB', 'YCbCr'}"

    try:
        if URL_REGEX.match(path):
            response = requests.get(path)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(path)
    except IOError:
        return None

    if mode is not None:
        if img.mode != mode:
            img = img.convert(mode)

    if target_size is not None:
        wh_tuple = (target_size[1], target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)

    return img


def img_to_array(img, data_format='channels_last'):
    '''Converts a PIL Image instance to a Numpy array
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

    if x.ndim == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif x.ndim == 2:
        if data_format == 'channels_first':
            x = x.reshape((1,) + x.shape)
        else:
            x = x.reshape(x.shape + (1,))
    else:
        raise ValueError('Unsupported image shape:', x.shape)

    return x


def array_to_img(x, data_format='channels_last', scale=False):
    '''Converts a 3D Numpy array to a PIL Image instance
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
    if x.ndim == 2:
        x = x.reshape(x.shape + (1,))

    if x.ndim != 3:
        raise ValueError('Image array to have rank 3.', x.shape)

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)

    if scale:
        x = x.astype(np.float32)
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max > 0:
            x /= x_max
        x *= 255

    if x.shape[2] == 3:
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number:', x.shape[2])


def image_new(size, color=None):
    '''
    size: A 2-tuple, containing (width, height) in pixels
    color: What color to use for the image, Default is black
    '''
    if color is None:
        color = def_color

    return Image.new('RGB', size, color)


def draw_text(img, pos, text, font=None, fill=None):
    '''
    img: PIL.Image.Image object
    pos: Top left corner of the text
    text: A text
    '''
    if font is None:
        font = def_font

    if fill is None:
        fill = def_color

    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, file=fill)

    return None


def draw_polygon(img, xys, fill=None, outline=None):
    '''
    img: PIL.Image.Image object
    xys: [(x, y), (x, y), ...] or [x, y, x, y, ...]
    '''
    draw = ImageDraw.Draw(img)
    draw.polygon(xys, fill, outline)

    return None


def draw_rectangle(img, xys, fill=None, outline=None):
    '''
    img: PIL.Image.Image object
    xys: [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]
    '''
    draw = ImageDraw.Draw(img)
    draw.rectangle(xys, fill, outline)

    return None