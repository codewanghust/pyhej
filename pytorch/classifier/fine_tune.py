'''https://github.com/pytorch/examples/tree/master/imagenet
'''
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from .tools import AverageMeter, adjust_learning_rate, save_checkpoint, accuracy, validate


def get_model(name, pretrained, cuda=True, num_class=2):
    if pretrained:
        model = models.__dict__[name](pretrained=True)
    else:
        model = models.__dict__[name]()
    model.fc = nn.Linear(model.fc.in_features, num_class)
    if cuda and torch.cuda.is_available():
        if name.startswith(('alexnet', 'vgg')):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()
    return model


def train(train_loader, model, criterion, optimizer, topk=(1, 5), print_freq=1000):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader, 1):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        # measure accuracy and record loss
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk)

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == len(train_loader):
            print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\n\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5))


def todo(args, topk=(1, 5)):
    args.best_prec1 = 0

    # create model
    model = get_model(args.name, args.pretrained, args.cuda, args.num_class)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.cuda and torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir, valdir = os.path.join(args.data, 'train'), os.path.join(args.data, 'val')

    train_dataset = datasets.ImageFolder(traindir,
        transforms.Compose([
            #transforms.Scale((256, 256)),
            #transforms.RandomSizedCrop(224),
            transforms.Scale((224, 224)),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize((0.5065, 0.5091, 0.4707), (0.2226, 0.2189, 0.2175)),
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True)

    val_dataset = datasets.ImageFolder(valdir,
        transforms.Compose([
            transforms.Scale((224, 224)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize((0.5065, 0.5091, 0.4707), (0.2226, 0.2189, 0.2175)),
        ]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)
        print('\nEpoch: [{0}]'.format(epoch))

        # training
        train(train_loader, model, criterion, optimizer, topk, args.print_freq)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, topk, args.print_freq)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > args.best_prec1
        args.best_prec1 = max(prec1, args.best_prec1)
        save_checkpoint({
            'name' : args.name,
            'epoch': epoch + 1,
            'best_prec1': args.best_prec1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.output)


'''
## train
import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', help='path to dataset')
parser.add_argument('--num-class', default=2, type=int)
parser.add_argument('--name', default='resnet18', type=str, help='model name')
parser.add_argument('--pretrained', dest='pretrained', action='store_true')
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('-b', '--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--print-freq', '-p', default=1000, type=int)
parser.add_argument('--output', default='/tmp', type=str)
args = parser.parse_args(['/data2/tmps/cifar-10/tmp', '--num-class', '10',
    '-j', '8', '--cuda', '-b', '32', '--epochs', '2', '--output', '/data2/tmps/cifar-10'])
print(args)
from pyhej.pytorch.classifier.fine_tune import todo
todo(args, topk=(1, 5))


## test
import torch
from pyhej.pytorch.classifier.fine_tune import get_model
from PIL import Image
import torchvision.transforms as transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

data_transforms = transforms.Compose([
    transforms.Scale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5065, 0.5091, 0.4707), (0.2226, 0.2189, 0.2175)),
])

img = pil_loader('/data2/tmps/1114_not_medical_c10/tmp/val/c4/img2622.jpeg')
img = data_transforms(img)
inputs = img.unsqueeze(0)
inputs_var = torch.autograd.Variable(inputs, volatile=True)

model = get_model('resnet18', False, True, 6)
checkpoint = torch.load('/data2/tmps/1114_not_medical_c10/resnet18-1115/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

outputs = model(inputs_var)
outputs.topk(2, 1)
# import torch.nn as nn
# softmax = nn.Softmax()
# softmax(outputs).topk(2, 1)
'''
