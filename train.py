import argparse 
import os 
import logging 
import time 
import numpy as np 
import shutil
import setproctitle
from apex import amp 

import torch 
import torch.nn as nn 
import torch.backends.cudnn as cudnn 
import torch.optim 
import torch.utils.data 
import torch.nn.functional as F 
import torchvision.datasets as datasets 
import torchvision.models as models 
from torch.utils.data import Dataset, DataLoader, Sampler

from model import Resnet 
from data import SinglePhase
from data import transforms
from utils import AverageMeter, accuracy_binary



parser = argparse.ArgumentParser(description='MVI Prediction')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start_iter', default=0, type=int, metavar='N',
                    help='manual iter number (useful on restarts')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--scale', default=0.05, type=float, 
                    help='scale factor range for augmentation.')
parser.add_argument('--angle', default=15, type=int, 
                    help='rotation angle range in degrees for augmentation.')

args = parser.parse_args()

date = '0619'


# build directories to save checkpoints and logs
path = os.path.dirname(__file__)
ckpts = os.path.join(path, 'ckpts')
logs = os.path.join(path, 'logs')
if not os.path.exists(ckpts):
    os.mkdir(ckpts)
if not os.path.exists(logs):
    os.mkdir(logs)


# use logging to record
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(os.path.join(logs, date) + '.log', mode='w')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def save_ckpt(state, is_best, name='resnet50'):
    file_name = os.path.join(ckpts, date, name) + '.pth.tar'
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(os.path.dirname(file_name), name + 'best.pth.tar'))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    print('Use GPU: {} for training'.format(args.gpu))

    # creat model
    model = Resnet()
    model = model.cuda(args.gpu)
    logger.info(model)

    # define loss function (criterion) and optimizer 
    criterion = nn.CrossEntropyLoss.cuda(args.gpu)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # use apex to accelerate training speed
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # optionally resume from a checkpoint
    msg = ''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            best_acc = checkpoint['best_acc']
            if args.gpu is not None:
                best_acc = best_acc.to(args.gpu)
            msg = ("=> loaded checkpoint '{}' (iter {})"
                    .format(args.resume, checkpoint['iter']))
        else:
            msg = "=> no checkpoint found as '{}'".format(args.resume)
    else:
        msg = "-----------New training session-------------"

    msg += '\n' + str(args)
    logger.info(msg)


    torch.backends.cudnn.benchmark = True 

    # load the data into the model
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')

    train_dataset = SinglePhase(
        train_dir, 
        image_size=224, 
        transforms=transforms(scale=args.scale, angle=args.angle, flip_prob=0.5)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True
    )

    val_dataset = SinglePhase(
        val_dir, 
        image_size=224, 
        transforms=transforms(scale=args.scale, angle=args.angle, flip_prob=0.5)
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True
    )

    
    # actual train
    for epoch in range(args.start_epoch, args.epochs):
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args)
        
        # set proctitle in top command
        setproctitle.setproctitle('Epoch:[{}/{}]'.format(epoch+1, args.epochs))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc = validate(val_loader, model, criterion, args)

        # record the best accuracy and save checkpoint
        is_best = acc > best_acc
        save_ckpt({
            'epoch': epoch + 1, 
            'state_dict': model.state_dict(), 
            'best_acc': best_acc, 
            'optimizer': optimizer.state_dict()
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    # record the losses and accuracy
    Losses = AverageMeter('Loss', ':.4e')
    Accuracy = AverageMeter('Accuracy', ':6.3f')

    # switch to train mode
    model.train()

    for step, (data, target) in enumerate(train_loader):
        
        data = data.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(data)
        loss = criterion(output, target)

        # measure the accuracy and record loss
        acc, acc_0, acc_1 = accuracy_binary(output, target, logger)
        Losses.update(loss.item(), target.numel())
        Accuracy.update(acc, target.numel())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if step % args.print_freq == 0:
            msg = 'Epoch[{:0>3}/{:0>3}], Loss:{:.6f}, Accuracy:{:.3f}'.format(
                epoch+1, args.epochs, Losses.val, Accuracy.avg)
            logger.info(msg)


def validate(val_loader, model, criterion, args):
    # record the losses and accuracy
    Losses = AverageMeter('Loss', ':.4e')
    Accuracy = AverageMeter('Accuracy', ':6.3f')

    total_num = 0
    total_acc0 = 0
    total_acc1 = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for step, (data, target) in enumerate(val_loader):
            data = data.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(data)
            loss = criterion(output, target)

            # measure the accuracy and record loss
            acc, acc_0, acc_1 = accuracy_binary(output, target, logger)
            Losses.update(loss.item(), target.numel())
            Accuracy.update(acc, target.numel())

            if step % args.print_freq == 0:
                msg = 'Loss:{:.6f}, Accuracy:{:.3f}'.format(Losses.val, Accuracy.avg)
                logger.info(msg)

            total_num += target.numel()
            total_acc0 += acc_0
            total_acc1 += acc_1
        
        logger.info(
            'total number: %d, total acc0: %d, total acc1: %d', 
            total_num, total_acc0, total_acc1)
    
    return Accuracy.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == "__main__":
    main()