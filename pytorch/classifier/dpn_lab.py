'''https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
'''
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from .dpn import DPN92
from .tools import save_checkpoint


# Training
def train(epoch, net, trainloader, use_cuda, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum()
        total += targets.size(0)

    print('%s | Loss: %.3f | Acc: %.3f (%d/%d)'
            % (time.strftime('%Y-%m-%d %H:%M:%S'), train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# Testing
def test(best_acc, epoch, net, testloader, use_cuda, criterion, filepath):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum()
        total += targets.size(0)

    print('%s | Loss: %.3f | Acc: %.3f (%d/%d)'
            % (time.strftime('%Y-%m-%d %H:%M:%S'), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'net': net.module if use_cuda else net,
        'acc': acc,
        'epoch': epoch,
    }, is_best, filepath)

    return best_acc


def todo(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    best_acc = 0
    start_epoch = 0

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Scale((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5065, 0.5091, 0.4707), (0.2226, 0.2189, 0.2175)),
    ])

    transform_test = transforms.Compose([
        transforms.Scale((32, 32)),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5065, 0.5091, 0.4707), (0.2226, 0.2189, 0.2175)),
    ])

    traindir, valdir = os.path.join(args.data, 'train'), os.path.join(args.data, 'val')

    #trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    trainset = datasets.ImageFolder(traindir, transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    #testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
    testset = datasets.ImageFolder(valdir, transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        # net = VGG('VGG19')
        # net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()

    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    cudnn.benchmark = True

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch, net, trainloader, use_cuda, optimizer, criterion)
        best_acc = test(best_acc, epoch, net, testloader, use_cuda, criterion, args.output)


'''
## train
import argparse
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('data', help='path to dataset')
parser.add_argument('--num-class', default=2, type=int)
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('-b', '--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
parser.add_argument('--output', default='/tmp', type=str)
args = parser.parse_args(['/data2/tmps/cifar-10', '--num-class', '10',
    '-j', '8', '--cuda', '-b', '32', '--epochs', '2', '--output', '/data2/tmps/cifar-10'])
print(args)
from pyhej.pytorch.classifier.dpn_lab import todo
todo(args)


## test
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

data_transforms = transforms.Compose([
    transforms.Scale((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5065, 0.5091, 0.4707), (0.2226, 0.2189, 0.2175)),
])

img = pil_loader('/data2/tmps/1114_not_medical_c10/tmp/val/c4/img2622.jpeg')
img = data_transforms(img)
inputs = img.unsqueeze(0)
inputs_var = torch.autograd.Variable(inputs, volatile=True)

checkpoint = torch.load('/data2/tmps/1114_not_medical_c10/dpn92-1115/model_best.pth.tar')
net = checkpoint['net']
# if use cuda
model = nn.DataParallel(net).cuda()
# if use cpu
model = net.cpu()
model.eval()

outputs = model(inputs_var)
outputs.topk(2, 1)
# import torch.nn as nn
# softmax = nn.Softmax()
# softmax(outputs).topk(2, 1)
'''
