import os 
import numpy as np 
import torch 
import time 
import torch.nn.functional as F 


class AverageMeter(object):
    """Computes and stors the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt 
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
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def accuracy_binary(output, target, logger):
    predict = F.softmax(output)
    predict = torch.argmax(predict, dim=1)

    batch_size = target.size(0)

    acc_0 = 0
    acc_1 = 0
    count = 0

    logger.info('predict:%s', predict)
    logger.info('target :%s', target)

    for (a, b) in zip(predict, target):
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        if a == 0 and b == 0:
            acc_0 += 1
            count += 1
        elif a == 1 and b == 1:
            acc_1 += 1
            count += 1

    return float(count), acc_0, acc_1
    