'''https://github.com/pytorch/examples/tree/master/imagenet
'''
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from easydict import EasyDict as edict
from pyhej.pytorch.classifier.tools import ImageFolder, adjust_learning_rate, train, validate, save_checkpoint


cfg = edict()
cfg.data = '/data2/tmps/1114_not_medical_c10/tmp'

cfg.use_cuda = True
cfg.model_name = 'resnet50'
cfg.pretrained = False
cfg.num_classes = 6

cfg.lr = 0.1
cfg.momentum = 0.9
cfg.weight_decay = 1e-4

cfg.start_epoch = 0
cfg.best_accs = [(1, 0), (3, 0)]
cfg.topks = (1, 3)
cfg.resume = ''

cfg.batch_size = 32
cfg.workers = 8
cfg.epochs = 900
cfg.output = '/data2/tmps/1114_not_medical_c10/models/{}_{}'.format(cfg.model_name, time.strftime('%y%m%d'))


# create model
model = models.__dict__[cfg.model_name](pretrained=cfg.pretrained, num_classes=cfg.num_classes)
model = nn.DataParallel(model)
if cfg.use_cuda:
    model = model.cuda()


# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
if cfg.use_cuda:
    criterion = criterion.cuda()


if os.path.isfile(cfg.resume):
    checkpoint = torch.load(cfg.resume)
    cfg.start_epoch = checkpoint['epoch']
    cfg.best_accs = checkpoint['best_accs']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


cudnn.benchmark = True


# Data loading code
traindir, valdir = os.path.join(cfg.data, 'train'), os.path.join(cfg.data, 'val')
train_dataset = ImageFolder(traindir,
    transforms.Compose([
        # new version replace `Scale` with `Resize`, `RandomSizedCrop` with `RandomResizedCrop`
        transforms.Scale((224, 224)),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.4809, 0.4748, 0.4503], std=[0.2344, 0.2301, 0.2304]),
    ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)
val_dataset = ImageFolder(valdir,
    transforms.Compose([
        # new version replace `Scale` with `Resize`, `RandomSizedCrop` with `RandomResizedCrop`
        transforms.Scale((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.4809, 0.4748, 0.4503], std=[0.2344, 0.2301, 0.2304]),
    ]))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)


for epoch in range(cfg.epochs):
    print('{} | Epoch: {} + {}'.format(time.strftime('%y-%m-%d %H:%M:%S'), cfg.start_epoch, epoch))
    adjust_learning_rate(optimizer, epoch, cfg.lr)
    # train for one epoch
    train(train_loader, model, criterion, optimizer, cfg.topks, cfg.use_cuda)
    # evaluate on validation set
    best_accs = validate(val_loader, model, criterion, cfg.topks, cfg.use_cuda)
    # remember best and save checkpoint
    is_best = best_accs[0][1] > cfg.best_accs[0][1]
    if is_best:
        cfg.best_accs = best_accs
    save_checkpoint({
        'epoch': cfg.start_epoch+epoch,
        'best_accs': best_accs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, is_best, cfg.output)


cfg