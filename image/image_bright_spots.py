from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
def detection_bright_spots(image, bright=200, minMax=(10, 1000)):
    '''
    doc: https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
    doc: https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
    image: string of image path
    view:
      image, mask = detection_bright_spots('img.jpg', 210, (300, 3000))
      import imutils
      import matplotlib.pyplot as plt
      plt.figure(figsize=(9, 9))
      plt.imshow(imutils.opencv2matplotlib(image))
      plt.axis('off')
      plt.show()
    '''
    image = cv2.imread(image)
    # 转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 平滑以减少高频噪声,即模糊
    blurred = gray #cv2.GaussianBlur(gray, (11, 11), 0)
    # 揭示模糊图像中最明亮的区域,我们需要应用阈值
    #   像素值`>=200`设置为255(白色)
    #   像素值`<200`被设置为0(黑色)
    thresh = cv2.threshold(blurred, bright, 255, cv2.THRESH_BINARY)[1]
    # 图像中有一些噪点(即小斑点),我们通过执行一系列的腐蚀和膨胀来清理它们
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=4)
    # 在应用我们的腐蚀和膨胀之后,我们仍然希望过滤掉任何剩余的'噪音'区域
    #   一个很好的方法是`connected-component analysis`
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype='uint8')
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # construct the label mask and count the number of pixels
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if minMax[0] < numPixels and numPixels < minMax[1]:
            mask = cv2.add(mask, labelMask)
    # 最后一步是在我们的图像上绘制标记
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = contours.sort_contours(cnts)[0]
    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        #((cX, cY), radius) = cv2.minEnclosingCircle(c)
        #cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
        #cv2.putText(image, '#{}'.format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return mask, image


from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
def detection_bright_spots_2(filename, bright=200, minMax=(8, 200)):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = gray #cv2.medianBlur(gray, 3) #cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold(blurred, bright, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=4)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=4)
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype='uint8')
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if minMax[0] < numPixels and numPixels < minMax[1]:
            mask = cv2.add(mask, labelMask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    if len(cnts) < 1:
        return mask, image
    cnts = contours.sort_contours(cnts)[0]
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
    return mask, image


from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
def detection_bright_spots_3(filename, bright=200, minMax=(8, 200)):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = gray #cv2.medianBlur(gray, 3) #cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold(blurred, bright, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=4)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=4)
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype='uint8')
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if minMax[0] < numPixels and numPixels < minMax[1]:
            mask = cv2.add(mask, labelMask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    if len(cnts) < 1:
        return mask, image
    cnts = contours.sort_contours(cnts)[0]
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
    return mask, image


import os
import cv2
import numpy as np
def save_image(filename, mask, image, save_dir='.'):
    orig = cv2.imread(filename)
    mask = np.dstack((mask, mask, mask))
    save_path = os.path.join(save_dir, os.path.basename(filename))
    cv2.imwrite(save_path, np.hstack((orig, mask, image)))
    return save_path


import numpy as np
import cv2
def arr_to_img(arr):
    x = np.clip(arr, 0, np.max(arr))
    x = x.astype(np.float32)
    x /= np.max(x)
    x *= 255
    return x.astype('uint8')


import scipy.signal as signal
def bbox(arr):
    '''
    arr: 灰度图像
    ex1.
      arr = Image.open('img1.jpg').convert('L')
    ex2.
      arr = cv2.imread('img1.jpg', 0)
    view:
      import matplotlib.pyplot as plt
      plt.figure(figsize=(9, 9))
      plt.imshow(image_suanzi, cmap=plt.cm.gray)
      plt.axis('off')
      plt.show()
    '''
    # Laplace算子
    laplacian = np.array([[0, 1, 0],
                          [1,-4, 1],
                          [0, 1, 0]])
    # 利用signal的convolve计算卷积
    image_suanzi = signal.convolve2d(arr, laplacian, mode='same')
    # 将卷积结果转化成 [0,255]
    image_suanzi = (image_suanzi / float(image_suanzi.max())) * 255
    # 为了使看清边缘检测结果 将大于灰度平均值的灰度变成255
    print(image_suanzi.max(), image_suanzi.mean())
    image_suanzi[image_suanzi < image_suanzi.mean()] = 0
    image_suanzi[image_suanzi > 15] = 255
    return image_suanzi