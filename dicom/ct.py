import SimpleITK as sitk
import dicom
import numpy as np
import cv2


'''
pip install SimpleITK
pip install pydicom
'''


def read_dicominfo(filename):
    try:
        plan = dicom.read_file(filename, force=True)
        temp = {'InstitutionName': plan.InstitutionName,  # 机构
               'PatientID': plan.PatientID,  # 病人
               'PatientName': plan.PatientName,  # 病人
               'StudyDate': plan.StudyDate,  # study
               'StudyTime': plan.StudyTime,  # study
               'Modality': plan.Modality,  # 项目
               'XRayTubeCurrent': plan.XRayTubeCurrent,  # 强度
               'ImagePositionPatient': plan.ImagePositionPatient,  # 位置
               'SeriesDate': plan.SeriesDate,  # series
               'SeriesTime': plan.SeriesTime,  # series
               'InstanceNumber': plan.InstanceNumber}  # 序号
    except Exception as e:
        temp = None
    return temp


def read_dicom(filename):
    '''
    read dicom file
    filename: as 'your/path/filename.dcm'
    '''
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    ## import matplotlib.pyplot as plt
    ## frames, wid, hei = img_array.shape
    ## plt.imshow(img_array[frame_num], plt.cm.bone)
    ## plt.show()
    plan = dicom.read_file(filename, force=True)
    return img_array, plan.Modality, plan.XRayTubeCurrent, plan.ImagePositionPatient, plan.SeriesTime


def img_array_scale(img_array):
    '''
    rescale image values to be within `[0, 255]`
    '''
    img_array = img_array.astype('float')
    img_array = img_array + max(-np.min(img_array), 0)
    img_array_max = np.max(img_array)
    if img_array_max > 0:
        img_array /= img_array_max
    img_array *= 255
    return img_array.astype('uint8')


def img_array_opti(img_array, limit=4.0, scale=True):
    '''
    dicom image array opti
    array: a signed integer array, about x-ray energy
    limit: please help opencv document
    scale: rescale image values to be within `[0, 255]`
    '''
    if scale:
        img_array = img_array_scale(img_array)
    img_array = img_array.astype('uint8')
    array_list = []
    for img in img_array:
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
        array_list.append(clahe.apply(img))
    return np.array(array_list)


'''
a, b, c = img_array[0], img_array_scale(img_array)[0], img_array_opti(img_array)[0]

from PIL import Image as pil_image
tmp = pil_image.fromarray(a, mode='L')
tmp.save('1.a.jpg')
tmp = pil_image.fromarray(b, mode='L')
tmp.save('1.b.jpg')
tmp = pil_image.fromarray(c, mode='L')
tmp.save('1.c.jpg')

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 5))
ax = plt.subplot('131')
ax.imshow(a, plt.cm.bone)
ax.set_title('a')

ax = plt.subplot('132')
ax.imshow(b, plt.cm.bone)
ax.set_title('b')

ax = plt.subplot('133')
ax.imshow(c, plt.cm.bone)
ax.set_title('c')
plt.show()
'''

