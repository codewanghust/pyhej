

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch




from PIL import Image
import torchvision.transforms as transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


data_transforms = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


img = pil_loader('/data2/tmps/1109_not_medical_c10/tmp/val/c1/img2062.jpeg')
img = data_transforms(img)
input = img.unsqueeze(0)


import torch
from pyhej.pytorch.classifier.fine_tune import get_model


model = get_model('resnet18', False, True, 10)
checkpoint = torch.load('/data2/tmps/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

input_var = torch.autograd.Variable(input, volatile=True)
output = model(input_var)




#####

import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


