import os
import re
import glob
import h5py
import numpy as np
from scipy.ndimage import imread

root = '/data2/tmps'
data_path = os.path.join(root, 'data/*b.jpg')
save_path = os.path.join(root, 'train.h5')
'''
1. data/xxx_b.jpg => data/xxx_gt.jpg
2. the same Image.size
'''

filenames = glob.glob(data_path)
len(filenames), filenames[:3]


# In[2]:


def_size = 41
def_stride = def_size//2
inputs = []
labels = []

re_sub = re.compile('b.jpg$', re.U)
for i in filenames:
    try:
        img_b_ycbcr = imread(i, mode='YCbCr')
        img_b_y = img_b_ycbcr[:,:,0].astype(float)
        img_b_y = img_b_y.reshape((1,) + img_b_y.shape)
        img_b_y = img_b_y/255.
        img_g_ycbcr = imread(re_sub.sub('gt.jpg', i), mode='YCbCr')
        img_g_y = img_g_ycbcr[:,:,0].astype(float)
        img_g_y = img_g_y.reshape((1,) + img_g_y.shape)
        img_g_y = img_g_y/255.
    except Exception as e:
        continue
    hs = set(tuple(range(0, img_b_y.shape[1]-def_size, def_stride)) + (img_b_y.shape[1]-def_size,))
    ws = set(tuple(range(0, img_b_y.shape[2]-def_size, def_stride)) + (img_b_y.shape[2]-def_size,))
    for h in hs:
        for w in ws:
            inputs.append(img_b_y[:,h:h+def_size,w:w+def_size])
            labels.append(img_g_y[:,h:h+def_size,w:w+def_size])


# In[3]:


hf = h5py.File(save_path, 'w')
hf['data'] = np.array(inputs)
hf['label'] = np.array(labels)
print(hf['data'].shape, hf['label'].shape)
hf.close()


# In[4]:


hf = h5py.File(save_path)
print(hf['data'].shape, hf['label'].shape)
hf.close()

