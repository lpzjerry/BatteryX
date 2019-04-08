from sklearn import metrics
import argparse
import os
import shutil
import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet
from PIL import Image

### Solver
parser = argparse.ArgumentParser(description='PyTorch Xray Training')
parser.add_argument('--epochs', default=15, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=48, type=int,
                    help='mini-batch size (default: 48)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 0.0005)')
parser.add_argument('--print_freq', '-p', default=1, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

### Dataset Processing
def default_loader(path):
    return Image.open(path).convert('RGB')

class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform = None, loader = default_loader):
        images = [] 
        labels = open(label).readlines()
        for line in labels:
            items = line.split() 
            img_name = items.pop(0)
            if os.path.isfile(os.path.join(root, img_name)):
                images.append((img_name, tuple([int(v) for v in items])))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)

best_accu = 0

### Utilities

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        accu = auc(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(accu, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0: 
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        accu = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(accu, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "models/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'models/' + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 7)) * (0.1 ** (epoch // 12))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)

    output = output.cpu().numpy()
    #print(output)
    output = np.where(output > 0.5, 1, 0)
    pred = torch.from_numpy(output).long()
    target = target.cpu().long()
    correct = pred.eq(target)
    correct = correct.numpy()

    res = []
    for k in range(attr_num):
        res.append(1.0*sum(correct[:,k]) / batch_size)
    return sum(res) / attr_num

def mean_accu(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)

    output = output.cpu().numpy()
    output = np.where(output > 0.5, 1, 0)
    pred = output
    target = target.cpu().long()
    target = target.numpy() # new. shape = (48,1)
    positive_accu = sum(pred * target) / sum(target)
    both_zero = sum(np.where(pred>0.5, 0, 1) * np.where(target>0.5, 0, 1))
    negative_accu = (both_zero) / (target.shape[0]-sum(target))
    return 0.5 * (positive_accu[0] + negative_accu[0])

def auc(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)

    output = output.cpu().numpy()
    output = np.where(output > 0.5, 1, 0)
    pred = output
    target = target.cpu().long()
    target = target.numpy()
    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
    return metrics.auc(fpr, tpr)

### Model Training

def main():
    global args, best_accu
    args = parser.parse_args()
    
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize
        ])

    train_dataset = MultiLabelDataset(root='../dataset/',
        label='../list/train.txt', transform=transform_train)
    val_dataset = MultiLabelDataset(root='../dataset/',
        label='../list/test.txt', transform=transform_test)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    # create model
    model = resnet.resnet50(pretrained=True)
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_accu = checkpoint['best_accu']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # iterative training
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        accu = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = accu > best_accu
        best_accu = max(accu, best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_accu': best_accu,
        }, is_best)
    print('Best accuracy: ', best_accu)

if __name__ == '__main__':
    main()
