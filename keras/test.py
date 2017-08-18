import numpy as np
from .image import load_img, img_to_array
def predict_on_batch(model, data, datagen, target_size=None):
    """
    Predict on batch

    # Arguments
        model: keras model object
        data: a list, as [(fname_image, class_index), ..]
        datagen: a ImageDataGenerator object
        target_size: tuple of ints `(height, width)`

    # Returns
        A list, such as `[ys, ys, ..]`

    # Raises
        ..
    """
    res = []
    for fname, _ in data:
        img = load_img(fname, target_size=target_size)
        arr = img_to_array(img)
        tmp = datagen.standardize(arr)
        res.append(model.predict(np.asarray([tmp]))[0,:])
    return res


import numpy as np
def evaluate_categorical(lys, ly_, topn=1):
    """
    Predict on batch

    # Arguments
        lys: list of the predict
        ly_: list of index, about gold label
        topn: loose degree, default 1

    # Returns
        Score, at `[0,1]`

    # Raises
        ..
    """
    res = []
    for ys, y_ in zip(lys, ly_):
        tmp = np.argsort(ys)[-topn:]
        res.append(y_ in tmp)
    return (sum(res)/len(res), res)

