'''https://github.com/pytorch/examples/tree/master/imagenet
'''
import os
import time
import shutil
import codecs
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    '''Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    '''
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        images = []
        for i, target in enumerate(sorted(os.listdir(root))):
            for dirpath, _, filenames in sorted(os.walk(os.path.join(root, target))):
                for filename in filenames:
                    if is_image_file(filename):
                        images.append((os.path.join(dirpath, filename), i))

        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        '''
        filepath, target = self.images[index]
        with open(filepath, 'rb') as f:
            with Image.open(f) as img:
                input = img.convert('RGB')
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.images)


class ImageFile(data.Dataset):
    def __init__(self, filename, transform=None, target_transform=None):
        images = []
        with codecs.open(filename, 'r', 'utf-8') as reader:
            for line in reader.readlines():
                if line.startswith('#'):
                    continue
                filepath, target = line.strip().split(',')
                images.append((filepath, target))

        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        '''
        filepath, target = self.images[index]
        with open(filepath, 'rb') as f:
            with Image.open(f) as img:
                input = img.convert('RGB')
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.images)


def test(model, inputs, topk=1, softmax=None):
    '''test possible k categories
    import torch.nn as nn
    softmax = nn.Softmax()
    '''
    inputs_var = torch.autograd.Variable(inputs, volatile=True)
    outputs = model(inputs_var)
    if softmax:
        outputs = softmax(outputs)
    return outputs.topk(topk, 1)


def eval(model, inputs, targets, topks=(1,), softmax=None):
    '''evaluate model in topks
    import torch.nn as nn
    softmax = nn.Softmax()
    '''
    score, pred = test(model, inputs, max(topks), softmax)
    score = score.t()
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred)).float()

    res = []
    for k in topks:
        score_k = score[:k].sum(0, keepdim=True)
        correct_k = correct[:k].sum(0, keepdim=True)
        res.append((score_k[0].tolist(), correct_k[0].tolist()))
    return res


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.
    dataset = ImageFolder('/your/image/path/')
    get_mean_and_std(dataset)
    '''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, init_lr):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs'''
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, 5)):
    '''Computes the precision@k for the specified values of k'''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, file_path='tmp'):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = os.path.join(file_path, 'checkpoint.pth.tar')
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(file_path, 'model_best.pth.tar'))


def validate(val_loader, model, criterion, topk=(1, 5), print_freq=1000):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader, 1):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        # measure accuracy and record loss
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk)

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == len(val_loader):
            print('Test : [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg
