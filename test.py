import os 
import argparse
import logging 
import numpy as np 
from sklearn.metrics import roc_auc_score

import torch 
import torch.nn as nn 
import torch.backends.cudnn as cudnn 
import torch.optim 
import torch.utils.data 
import torch.nn.functional as F 
import torchvision.datasets as datasets 
import torchvision.models as models 
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from model import Resnet18, Resnet50, DilatedResnet, Attention, DRN22, DRN22_test
from data import SinglePhase, transforms
from utils import AverageMeter, accuracy_binary


parser = argparse.ArgumentParser(description='MVI test')

parser.add_argument('data', metavar='DIR', 
                    help='path to dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N', 
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', 
                    help='number of data loads into the model (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', 
                    help='path to the checkpoint (default: none)')
parser.add_argument('--gpu', default=0, type=int, 
                    help='GPU id to use')
parser.add_argument('--scale', default=0.05, type=float, 
                    help='scale factor range for augmentation.')
parser.add_argument('--angle', default=15, type=int, 
                    help='rotation angle range in degrees for augmentation.')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

args = parser.parse_args()


def main():
    torch.cuda.set_device(args.gpu)
    print('Use GPU:  {} to test'.format(args.gpu))

    model = DRN22_test()
    model = model.cuda(args.gpu)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(args.gpu)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    

    if args.resume:
        if os.path.isfile(args.resume):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    test_dir = os.path.join(args.data, 'test')

    test_dataset = SinglePhase(
        test_dir, 
        image_size=224, 
        transforms=transforms(scale=args.scale, angle=args.angle, flip_prob=0.5)
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True 
    )


    model.eval()

    id_label = {}
    id_slice_sum = {}
    id_slice_num = {}

    with torch.no_grad():
        for step, (data, target, id_num) in enumerate(test_loader):
            data = data.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output, vector = model(data)
            # print(vector.shape)
            # print(vector)

            for (a, b, c) in zip(vector, target, id_num):
                a = a.cpu().numpy()
                b = b.cpu().numpy()
                if c not in id_slice_num:
                    id_slice_num[c] = 1
                    id_slice_sum[c] = a 
                    id_label[c] = b 
                else:
                    id_slice_num[c] += 1
                    id_slice_sum[c] += a 

        path = '/home/drs/Desktop/DL_feature'

        for key in id_label:
            average_vector = id_slice_sum[key] / id_slice_num[key]
            # print(average_vector.shape)
            # print(type(average_vector))
            # print(average_vector)
            np.save((os.path.join(path, key) + '_dl.npy'), average_vector)


if __name__ == "__main__":
    main()