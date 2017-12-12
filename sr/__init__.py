import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torchvision.transforms import ToTensor
from pyhej.pillow import image_new, draw_text


def PSNR(pred, gt):
    imdff = pred - gt
    rmse = np.mean(imdff**2)
    if rmse == 0:
        return 100
    return 10 * math.log10(255.**2 / rmse)


def test(model, img_b, img_gt=None, cuda=False):
    '''model of sub_pixel
    '''
    img_b = Image.open(img_b).convert('YCbCr')
    y, cb, cr = img_b.split()

    input = autograd.Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    if cuda:
        input = input.cuda()

    output = model(input)
    if cuda:
        output = output.cpu()

    out_img_y = output.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)

    if img_gt:
        img_gt = Image.open(img_gt).convert('YCbCr')
        y, cb, cr = img_gt.split()
        psnr = PSNR(out_img_y[0], np.asarray(y, dtype=np.float32))
    else:
        psnr = None

    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    return out_img, psnr


def imprint(img_h, img_b, img_gt=None, text=None, filename=None):
    '''
    img_h: A PIL Image
    img_b: A file path
    img_gt: A file path
    '''
    img_b = Image.open(img_b).convert('RGB')

    wid_b, hei_b = img_b.size
    wid_h, hei_h = img_h.size

    if img_gt:
        img_gt = Image.open(img_gt).convert('RGB')
        wid_gt, hei_gt = img_gt.size

        out = image_new((wid_gt+wid_b+wid_h, max(hei_gt, hei_h)), (0, 0, 0))
        out.paste(img_gt, (0, 0))
        out.paste(img_b , (0+wid_gt, (hei_h-hei_b)//2))
        out.paste(img_h , (0+wid_gt+wid_b, 0))
    else:
        out = image_new((wid_b+wid_h, hei_h), (0, 0, 0))
        out.paste(img_b, (0, (hei_h-hei_b)//2))
        out.paste(img_h, (0+wid_b, 0))

    if text:
        draw_text(out, (0, 0), text, fill=(255, 0, 0))

    if filename:
        out.save(filename)
        return filename
    else:
        return out