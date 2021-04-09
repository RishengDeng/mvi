import os 
import argparse
import logging 
import numpy as np 
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('.')

import torch 
import torch.nn as nn 
import torch.optim 
import torch.utils.data 
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from model import MulRes18Att
from data import TwoPhases

parser = argparse.ArgumentParser(description='Test AUC')

parser.add_argument('--workers', default=4, type=int, metavar='N', 
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', 
                    help='number of data loads into the model (default: 1)')
parser.add_argument('--gpu', default=0, type=int, 
                    help='GPU id to use')
parser.add_argument('--scale', default=0.05, type=float, 
                    help='scale factor range for augmentation.')
parser.add_argument('--angle', default=15, type=int, 
                    help='rotation angle range in degrees for augmentation.')
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

args = parser.parse_args()

date = '0220'

data_path = '/media/drs/extra/Datasets/MVI/'

path = os.path.dirname(__file__)
path = os.path.dirname(path)
logs = os.path.join(path, 'testlogs', date)
print(logs)
if not os.path.exists(logs):
    os.makedirs(logs)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
log_path = os.path.join(logs, '3_datasets_auc') + '.log'
handler = logging.FileHandler(log_path, mode='w')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    torch.cuda.set_device(args.gpu)
    print('Use GPU: {} to test'.format(args.gpu))

    model = MulRes18Att()
    model = model.cuda(args.gpu)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(args.gpu)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    data_list = ['train', 'val', 'EV1_process/bbox_npy']

    resume_path = '/media/drs/extra/Learn/code/mvi/ckpts/0220/art_pv_attention_1'
    for i in range(50):
        resume_tar = resume_path + str(i + 1) + '.pth.tar'

        logger.info('\n\nepoch:{}'.format(i + 1))
        print(resume_tar)

        if os.path.isfile(resume_tar):
            print('==> loading checkpoint {}'.format(resume_tar))
            checkpoint = torch.load(resume_tar)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('==> no checkpoint found at {}'.format(resume_tar))

        for subdata in data_list:
            test_dir = data_path + subdata

            logger.info('\ndataset: {}'.format(subdata))
            
            test_dataset = TwoPhases(
                test_dir,
                image_size=224,
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
            total_0 = 0
            total_acc = 0
            total_num = 0
            total_acc0 = 0
            total_acc1 = 0
            total_0 = 0

            with torch.no_grad():
                for step, (data, target, id_num) in enumerate(test_loader):
                    data = data.cuda(args.gpu, non_blocking=True)
                    art_data = data[:, :3, :, :].cuda(args.gpu, non_blocking=True)
                    pv_data = data[:, 3:, :, :].cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)

                    output = model(art_data, pv_data)
                    probability = F.softmax(output, dim=1)
                    for (a, b, c) in zip(probability, target, id_num):
                        a = a.cpu().numpy()
                        b = b.cpu().numpy()
                        if c not in id_slice_sum:
                            id_slice_num[c] = 1
                            temp = []
                            temp.append(a)
                            id_slice_sum[c] = temp
                            id_label[c] = b 
                        else:
                            id_slice_num[c] += 1
                            temp = id_slice_sum[c]
                            temp.append(a)
                            id_slice_sum[c] = temp 
                
                y_true = []
                y_score = []
                y_predict = []
                y_probability = []
                y_name = []

                for key in id_label:
                    slice_sum = sum(id_slice_sum[key])
                    average_prob = slice_sum / id_slice_num[key]
                    y_true.append(id_label[key])
                    y_score.append(average_prob[1])
                    a = np.argmax(average_prob)
                    b = id_label[key]
                    y_predict.append(a)
                    y_probability.append(id_slice_sum[key])
                    y_name.append(key)
                    if a == 0 and b == 0:
                        total_acc0 += 1
                        total_0 += 1
                    elif a == 1 and b == 1:
                        total_acc1 += 1
                    elif a == 1 and b == 0:
                        total_0 += 1 

                total_num = len(id_label)
                total_1 = total_num - total_0
                y_score = np.array(y_score)
                y_true = np.array(y_true)
                total_acc = total_acc0 + total_acc1
                accuracy = total_acc / total_num

                auc = roc_auc_score(y_true, y_score)
                sensitivity = total_acc1 / total_1
                specificity = total_acc0 / total_0
                print(total_0, total_1, total_acc0, total_acc1)
                # ppv = total_acc1 / (total_0 - total_acc0 + total_acc1)
                # npv = total_acc0 / (total_1 - total_acc1 + total_acc0)
                # f1_score = 2 / ((1 / sensitivity) + (1 / ppv))

                logger.info(
                    'number:{}, acc0:{}, acc1:{}, accuracy:{:.3f}, auc:{:.3f}, sen:{:.3f}, spe:{:.3f}'.format(
                        total_num, total_acc0, total_acc1, accuracy, auc, sensitivity, specificity
                    )
                )

if __name__ == '__main__':
    main()