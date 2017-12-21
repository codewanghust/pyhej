# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

```
pip install visdom
pip install dominate

git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git gan_pix2pix
cd gan_pix2pix
```

## pix2pix datasets
Download the pix2pix datasets using the following script:
```
bash ./datasets/download_pix2pix_dataset.sh dataset_name
```

- facades: 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade)
- cityscapes: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/)

We provide a python script to generate pix2pix training data in the form of pairs of images {A,B}, where A and B are two different depictions of the same underlying scene. For example, these might be pairs {label map, photo} or {bw image, color image}. Then we can learn to translate A to B or B to A:

- Create folder /path/to/data with subfolders A and B. A and B should each have their own subfolders train, val, test, etc. In /path/to/data/A/train, put training images in style A. In /path/to/data/B/train, put the corresponding images in style B. Repeat same for other data splits (val, test, etc).
- Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., /path/to/data/A/train/1.jpg is considered to correspond to /path/to/data/B/train/1.jpg.

Python Script:
```python
import os
import codecs
import shutil

data_dir = '/data2/datasets/slyx/mr2_sr_x2/dataset_2_3'
save_dir = '/data2/datasets/slyx/mr2_2_3_pix2pix'

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

os.system('mkdir -m 777 -p {}'.format(os.path.join(save_dir, 'low')))
os.system('mkdir -m 777 -p {}'.format(os.path.join(save_dir, 'hig')))
for i in ['train', 'val', 'test']:
    os.system('mkdir -m 777 -p {}'.format(os.path.join(save_dir, 'low', i)))
    os.system('mkdir -m 777 -p {}'.format(os.path.join(save_dir, 'hig', i)))

with codecs.open(os.path.join(data_dir, 'dataset_train.txt'), 'r', 'utf-8') as reader:
    for i, line in enumerate(reader.readlines(), 1):
        file_name = '{:03d}.jpg'.format(i)
        img_a, img_b = line.strip().split(',')
        shutil.copyfile(img_a, os.path.join(save_dir, 'low/train', file_name))
        shutil.copyfile(img_b, os.path.join(save_dir, 'hig/train', file_name))

with codecs.open(os.path.join(data_dir, 'dataset_tests.txt'), 'r', 'utf-8') as reader:
    for i, line in enumerate(reader.readlines(), 1):
        file_name = '{:03d}.jpg'.format(i)
        img_a, img_b = line.strip().split(',')
        shutil.copyfile(img_a, os.path.join(save_dir, 'low/val', file_name))
        shutil.copyfile(img_b, os.path.join(save_dir, 'hig/val', file_name))
        shutil.copyfile(img_a, os.path.join(save_dir, 'low/test', file_name))
        shutil.copyfile(img_b, os.path.join(save_dir, 'hig/test', file_name))

#!tree -L 2 /data2/datasets/slyx/mr2_2_3_pix2pix
```

Once the data is formatted this way, Python Script:
```python
import os
import numpy as np
import cv2

fold_a = '/data2/datasets/slyx/mr2_2_3_pix2pix/low'
fold_b = '/data2/datasets/slyx/mr2_2_3_pix2pix/hig'
fold_ab = '/data2/datasets/slyx/mr2_2_3_pix2pix/data'

for sp in os.listdir(fold_a):
    img_fold_a = os.path.join(fold_a, sp)
    img_fold_b = os.path.join(fold_b, sp)
    img_fold_ab = os.path.join(fold_ab, sp)

    if not os.path.isdir(img_fold_ab):
        os.system('mkdir -m 777 -p {}'.format(img_fold_ab))

    img_list = os.listdir(img_fold_a)
    for file_name in img_list:
        path_a = os.path.join(img_fold_a, file_name)
        path_b = os.path.join(img_fold_b, file_name)
        if os.path.isfile(path_a) and os.path.isfile(path_b):
            path_ab = os.path.join(img_fold_ab, file_name)
            im_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
            im_b = cv2.imread(path_b, cv2.IMREAD_COLOR)
            height, width = im_b.shape[:2]
            im_a = cv2.resize(im_a, (width, height), interpolation=cv2.INTER_CUBIC)
            im_ab = np.concatenate([im_a, im_b], 1)
            cv2.imwrite(path_ab, im_ab)
```

## pix2pix train/test
Download a pix2pix dataset (e.g.facades):
```
bash ./datasets/download_pix2pix_dataset.sh facades
```

Train a model (bash ./scripts/train_pix2pix.sh):
```
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0
```

To view training results and loss plots, run python -m visdom.server and click the URL `http://localhost:8097`. To see more intermediate results, check out `./checkpoints/facades_pix2pix/web/index.html`.

Test the model (bash ./scripts/test_pix2pix.sh):
```
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch
```

The test results will be saved to a html file here: `./results/facades_pix2pix/latest_val/index.html`.

More example scripts can be found at scripts directory. Flags: see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.

### ours dataset
Train:
```
python train.py --dataroot /data2/datasets/slyx/mr2_2_3_pix2pix/data --name mr2_2_3_pix2pix_300_200 --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --display_id 0 --nThreads 8 --niter 300 --niter_decay 200
```
check out `./checkpoints/mr2_2_3_pix2pix/web/index.html`.

Test:
```
python test.py --dataroot /data2/datasets/slyx/mr2_2_3_pix2pix/data --name mr2_2_3_pix2pix_300_200 --model pix2pix --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --norm batch --nThreads 8
```
The test results will be saved to a html file here: `./results/mr2_2_3_pix2pix/latest_val/index.html`.