import os
import sys
import argparse


parser = argparse.ArgumentParser(description='Script')
parser.add_argument('--image-path', default='', type=str)
parser.add_argument('--output-path', default='', type=str)
args = parser.parse_args()


if args.image_path:
    image_path = args.image_path
else:
    sys.exit(0)

if args.output_path:
    output_path = args.output_path
else:
    output_path = os.path.join(image_path, '/../tmp')


from glob import glob
import shutil
import random


my_files = {}
for filename in glob(image_path + '/*.jpg'):
    class_name = filename.split('/')[-1].split('.')[-3]
    temp = my_files.get(class_name)
    if temp is None:
        my_files[class_name] = [filename]
    else:
        temp.append(filename)


# random.sample 100, and 80 to train, 20 to val
tmp_train = os.path.join(output_path, 'train')
if os.path.exists(tmp_train) and os.path.isdir(tmp_train):
    shutil.rmtree(tmp_train)


tmp_val = os.path.join(output_path, 'val')
if os.path.exists(tmp_val) and os.path.isdir(tmp_val):
    shutil.rmtree(tmp_val)


for key, val in my_files.items():
    val = random.sample(val, min(1000, len(val)))

    to_path = os.path.join(tmp_train, key)
    os.makedirs(to_path)
    for filename in val[:int(len(val) * 0.8)]:
        shutil.copy(filename, to_path)

    to_path = os.path.join(tmp_val, key)
    os.makedirs(to_path)
    for filename in val[int(len(val) * 0.8):]:
        shutil.copy(filename, to_path)
