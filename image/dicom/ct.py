import SimpleITK as sitk
import dicom
import cv2


'''
pip install SimpleITK
pip install pydicom
'''


def seq2str(seq, mode='{:.9f}'):
    return ','.join([mode.format(i) for i in seq])


def read_dicominfo(filename):
    '''
    read dicom file info
    filename: as 'your/path/filename.dcm'
    '''
    try:
        plan = dicom.read_file(filename, force=True)
        temp = {'PatientID': plan.PatientID,  # 病人
                'InstanceNumber': plan.InstanceNumber,  # 序号
                'XRayTubeCurrent': plan.XRayTubeCurrent,  # 强度
                'ImagePositionPatient': seq2str(plan.ImagePositionPatient),  # 位置
                'Path': filename}
    except Exception as e:
        temp, plan = None, None
    return temp, plan


def read_dicom(filename):
    '''
    read dicom file images
    filename: as 'your/path/filename.dcm'
    ## import matplotlib.pyplot as plt
    ## frames, wid, hei = img_array.shape
    ## plt.imshow(img_array[frame_num], plt.cm.bone)
    ## plt.show()
    '''
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    return img_array