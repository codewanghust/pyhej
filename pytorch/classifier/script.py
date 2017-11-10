image_path = '/data2/datasets/kaggle-dog-vs-cat/train'
ouput_path = '/data2/datasets/kaggle-dog-vs-cat/tmp'

import os
from glob import glob
import shutil
import random

random.seed(1234)

my_files = {}
for filename in glob(image_path + '/*.jpg'):
    class_name = filename.split('/')[-1].split('.')[-3]
    temp = my_files.get(class_name)
    if temp is None:
        my_files[class_name] = [filename]
    else:
        temp.append(filename)


# random.sample 1000 cat jpgs and 1000 dog jpgs for train
tmp_train = os.path.join(ouput_path, 'train')
if os.path.exists(tmp_train) and os.path.isdir(tmp_train):
    shutil.rmtree(tmp_train)

os.makedirs(tmp_train)
for key, val in my_files.items():
    to_path = os.path.join(tmp_train, key)
    os.makedirs(to_path)
    for filename in random.sample(val, 1000):
        shutil.copy(filename, to_path)


# random.sample 400 cat jpgs and 400 dog jpgs for val
tmp_val = os.path.join(ouput_path, 'val')
if os.path.exists(tmp_val) and os.path.isdir(tmp_val):
    shutil.rmtree(tmp_val)

os.makedirs(tmp_val)
for key, val in my_files.items():
    to_path = os.path.join(tmp_val, key)
    os.makedirs(to_path)
    for filename in random.sample(val, 400):
        shutil.copy(filename, to_path)
