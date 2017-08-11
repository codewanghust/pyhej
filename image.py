from PIL import Image as pil_image
def load_img(path, grayscale=False, target_size=None):
    '''
    Loads an image into PIL format
    Notes: PIL image has format `(width, height, channel)`
           Numpy array has format `(height, width, channel)`

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale
        target_size: `None` or tuple of ints `(height, width)`

    # Returns
        A PIL Image instance or `None`

    # Raises
        ..
    '''
    try:
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


import numpy as np
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


from PIL import Image as pil_image
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


from PIL import Image as pil_image
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


import multiprocessing
import numpy as np
def get_image_iter(data, target_size=None, batch_size=32, shuffle=False, seed=None, image_gen=None):
    '''
    A batch Iterator

    # Arguments
        data: List of `[(fpath, label), ..]`
        target_size: `None` or tuple of `(height, width)`
        batch_size: Integer, size of a batch
        shuffle: Boolean, whether to shuffle
        seed: `None` or seed for random
        image_gen: `None` or function

    # Returns
        Iterator, `(batch_x, batch_y)`

    # Raises
        ..
    '''
    fpaths = []
    labels = []
    for fpath, label in data:
        fpaths.append(fpath)
        labels.append(label)

    x_shape = target_size + (3,)
    y_shape = (max(labels) + 1,)

    index_iter = get_index_iter(len(fpaths), batch_size, shuffle, seed)

    while 1:
        batch_index, batch_size = next(index_iter)

        batch_x = np.zeros((batch_size,) + x_shape, dtype=np.float32)
        batch_y = np.zeros((batch_size,) + y_shape, dtype=np.float32)

        results = []
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        for index in batch_index:
            results.append(pool.apply_async(get_image_iter_job, (fpaths[index], target_size, image_gen)))
        pool.close()
        pool.join()

        for i, obj in enumerate(results):
            batch_x[i] = obj.get()

        for i, index in enumerate(batch_index):
            batch_y[i, labels[index]] = 1.

        yield (batch_x, batch_y)


def get_image_iter_job(fpath, target_size, image_gen):
    img = load_img(fpath, target_size=target_size)
    arr = img_to_array(img)
    if image_gen is None:
        return arr
    return image_gen(arr)


import numpy as np
def get_index_iter(n, batch_size=1, shuffle=False, seed=None):
    '''
    A batch Iterator

    # Arguments
        n: Integer, total number of samples
        batch_size: Integer, size of a batch
        shuffle: Boolean, whether to shuffle
        seed: Random seed for shuffling

    # Returns
        A Integer List

    # Raises
        ..
    '''
    epochs = 0
    batch_pos = 0
    while 1:
        if batch_pos == 0:
            epochs += 1
            if shuffle:
                if seed is not None:
                    np.random.seed(seed + epochs)
                index_array = np.random.permutation(n)
            else:
                index_array = np.arange(n)

        current_index = batch_pos
        if n > current_index + batch_size:
            current_batch_size = batch_size
            batch_pos = current_index + current_batch_size
        else:
            current_batch_size = n - current_index
            batch_pos = 0

        yield (index_array[current_index:current_index + current_batch_size], current_batch_size)


class ImageDataGenerator(object):
    pass

