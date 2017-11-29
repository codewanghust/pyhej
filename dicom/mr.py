import SimpleITK as sitk
import dicom
import cv2


'''
pip install SimpleITK
pip install pydicom
'''


def seq2str(seq, mode='{:.9f}'):
    return ','.join([mode.format(i) for i in seq])


def read_dicominfo(path):
    try:
        plan = dicom.read_file(path, force=True)
        temp = {'PatientID': plan.PatientID,
                'InstanceNumber': plan.InstanceNumber,
                'MagneticFieldStrength': plan.MagneticFieldStrength,
                'ImagePositionPatient': seq2str(plan.ImagePositionPatient),
                'SeriesDescription': plan.SeriesDescription,
                'Path': path}
    except Exception as e:
        temp = None
    return temp


def read_dicom(path):
    '''
    read dicom file
    path: as 'your/path/filename.dcm'
    '''
    ds = sitk.ReadImage(path)
    img_array = sitk.GetArrayFromImage(ds)
    ## import matplotlib.pyplot as plt
    ## frames, wid, hei = img_array.shape
    ## plt.imshow(img_array[frame_num], plt.cm.bone)
    ## plt.show()
    return img_array