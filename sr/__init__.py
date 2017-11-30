import time
import math
import numpy as np
from PIL import Image as pil_image
import matplotlib.pyplot as plt


def PSNR(pred, gt):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255. / rmse)


def test(model, img_b, img_gt=None, cuda=False):
    img_b = pil_image.open(img_b).convert('YCbCr')
    y, cb, cr = img_b.split()

    input = ToTensor()(y)
    if cuda:
        input = input.cuda()

    output = model(autograd.Variable(input).view(1, -1, y.size[1], y.size[0]))
    if cuda:
        output = output.cpu()

    out_img_arr = output.data[0].numpy()
    out_img_arr *= 255.0
    out_img_arr = out_img_arr.clip(0, 255)
    out_img_y = pil_image.fromarray(np.uint8(out_img_arr[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, pil_image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, pil_image.BICUBIC)
    out_img = pil_image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    if img_gt:
        img_gt = pil_image.open(img_gt).convert('YCbCr')
        y, cb, cr = img_gt.split()
        psnr = PSNR(out_img_arr[0], np.asarray(y, dtype=np.float32))
    else:
        psnr = None

    return out_img, psnr


def imprint(img_b, img_h, img_gt=None, plan='DNN'):
    if isinstance(img_b, str):
        img_b = pil_image.open(img_b)

    fig = plt.figure(figsize=(12, 4))

    if img_gt:
        img_gt = pil_image.open(img_gt)
        ax = plt.subplot('131')
        ax.imshow(img_gt, cmap='gray')
        ax.set_title('GT')

    ax = plt.subplot('132')
    ax.imshow(img_b, cmap='gray')
    ax.set_title('Input')

    ax = plt.subplot('133')
    ax.imshow(img_h, cmap='gray')
    ax.set_title('Output(PSNR={})'.format(plan))
    plt.show()