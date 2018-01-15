import SimpleITK as sitk
import dicom
import cv2


'''
pip install SimpleITK
pip install pydicom
'''


def read_dicominfo(filename):
    '''
    read dicom file info
    filename: as 'your/path/filename.dcm'
    '''
    try:
        plan = dicom.read_file(filename, force=True)
        temp = {'PatientID': plan.PatientID,  # 病人
                'InstanceNumber': plan.InstanceNumber,  # 序号
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