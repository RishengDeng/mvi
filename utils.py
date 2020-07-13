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



def accuracy_binary(output, target, logger, id_num, id_dict={}, id_label = {}, mode='train'):
    # if mode == 'val':
    #     logger.info('output:%s', output)
    predict = F.softmax(output)
    predict = torch.argmax(predict, dim=1)

    batch_size = target.size(0)

    acc_0 = 0
    acc_1 = 0
    count = 0
    # id_dict = {}
    # id_label = {}

    if mode == 'val':
        logger.info('predict:%s', predict)
        logger.info('target :%s', target)

        # if one case is predicted as positive, 
        # the patient will be predicted as positive
        for (a, b, c) in zip(predict, target, id_num):
            a = a.cpu().numpy()
            b = b.cpu().numpy()
            if c not in id_dict:
                id_dict[c] = 0
                id_label[c] = b 
            else:
                id_dict[c] += a
        
        for key in id_dict:
            if id_dict[key] > 0:
                a = 1
            else:
                a = 0
            b = id_label[key]
            if a == 0 and b == 0:
                acc_0 += 1
                count += 1
            elif a == 1 and b == 1:
                acc_1 += 1
                count += 1

        return id_dict, id_label

    if mode == 'train':
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
    