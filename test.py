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

from model import Resnet18, Resnet50, DilatedResnet, Attention, DRN22, DRN22_test, \
    MulRes18, MulRes18Att
from data import SinglePhase, transforms, TwoPhases
from utils import AverageMeter, accuracy_binary, stack3array


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


date = '0220'


path = os.path.dirname(__file__)
logs = os.path.join(path, 'testlogs', date)
if not os.path.exists(logs):
    os.makedirs(logs)


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
log_path = os.path.join(logs, 'ev1_attention_res18_1') + '.log'
handler = logging.FileHandler(log_path, mode='w')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



def main():
    torch.cuda.set_device(args.gpu)
    print('Use GPU:  {} to test'.format(args.gpu))

    # model = DRN22_test()
    # model = Resnet18()
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
    

    if args.resume:
        resume_path = args.resume + 'art_pv_attention_111' + '.pth.tar'
        if os.path.isfile(resume_path):
            print('==> loading checkpoint {}'.format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # test_dir = os.path.join(args.data, 'bbox_npy', 'dl')
    # test_dir = os.path.join(args.data, 'val')
    test_dir = os.path.join(args.data, 'bbox_npy')

    # test_dataset = SinglePhase(
    test_dataset = TwoPhases(
        test_dir, 
        image_size=224, 
        # transforms=transforms(scale=args.scale, angle=args.angle, flip_prob=0.5)
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
    acc0 = 0
    acc1 = 0
    total_num = 0
    total_acc0 = 0
    total_acc1 = 0
    total_0 = 0

    # with torch.no_grad():
    #     for step, (data, target, id_num) in enumerate(test_loader):
    #         data = data.cuda(args.gpu, non_blocking=True)
    #         target = target.cuda(args.gpu, non_blocking=True)

    #         output, vector = model(data)
    #         # print(vector.shape)
    #         # print(vector)

    #         for (a, b, c) in zip(vector, target, id_num):
    #             a = a.cpu().numpy()
    #             b = b.cpu().numpy()
    #             if c not in id_slice_num:
    #                 id_slice_num[c] = 1
    #                 id_slice_sum[c] = a 
    #                 id_label[c] = b 
    #             else:
    #                 id_slice_num[c] += 1
    #                 id_slice_sum[c] += a 

    #     path = '/home/drs/Desktop/DL_feature'

    #     for key in id_label:
    #         average_vector = id_slice_sum[key] / id_slice_num[key]
    #         # print(average_vector.shape)
    #         # print(type(average_vector))
    #         # print(average_vector)
    #         np.save((os.path.join(path, key) + '_dl.npy'), average_vector)
    with torch.no_grad():
        for step, (data, target, id_num) in enumerate(test_loader):
            data = data.cuda(args.gpu, non_blocking=True)
            art_data = data[:, :3, :, :].cuda(args.gpu, non_blocking=True)
            pv_data = data[:, 3:, :, :].cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            # output = model(data)
            output = model(art_data, pv_data)
            probability = F.softmax(output, dim=1)
            predict = torch.argmax(probability, dim=1)

    #         for (a, b, c) in zip(probability, target, id_num):
    #             a = a.cpu().numpy()
    #             b = b.cpu().numpy()
    #             if c not in id_slice_num:
    #                 id_slice_num[c] = 1
    #                 id_slice_sum[c] = a 
    #                 id_label[c] = b 
    #             else: 
    #                 id_slice_num[c] += 1
    #                 id_slice_sum[c] += a 
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
        y_score_1 = []
        y_pre = []
        y_predict = []
        y_probability = []
        y_name = []

    #     for key in id_label:
    #         average_prob = id_slice_sum[key] / id_slice_num[key]
    #         y_true.append(id_label[key])
    #         y_score.append(average_prob)
    #         y_score_1.append(average_prob[1])
    #         a = np.argmax(average_prob)
    #         y_pre.append(a)
    #         b = id_label[key]
    #         if b == 0:
    #             total_0 += 1
    #             if a == 0:
    #                 total_acc += 1
    #                 acc0 += 1
    #         elif b == 1:
    #             if a == 1:
    #                 total_acc += 1
    #                 acc1 += 1
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
    #     accuracy = total_acc / total_num
        y_score = np.array(y_score)
    #     y_score_1 = np.array(y_score_1)
        y_true = np.array(y_true)
    #     y_pre = np.array(y_pre)
    #     auc = roc_auc_score(y_true, y_score_1)
    #     sensitivity = acc1 / total_1
    #     specificity = acc0 / total_0
        total_acc = total_acc0 + total_acc1
        accuracy = total_acc / total_num

        auc = roc_auc_score(y_true, y_score)
        sensitivity = total_acc1 / total_1
        specificity = total_acc0 / total_0
        print(total_0, total_1, total_acc0, total_acc1)
        ppv = total_acc1 / (total_0 - total_acc0 + total_acc1)
        npv = total_acc0 / (total_1 - total_acc1 + total_acc0)
        f1_score = 2 / ((1 / sensitivity) + (1 / ppv))

        logger.info(
            'number:{}, acc0:{}, acc1:{}, accuracy:{:.3f}, auc:{:.3f}, sen:{:.3f}, spe:{:.3f}, ppv:{:.3f}, npv:{:.3f}, f1:{:.3f}'.format(
                total_num, total_acc0, total_acc1, accuracy, auc, sensitivity, specificity, ppv, npv, f1_score
            )
        )

        txt_path = os.path.splitext(log_path)[0] + '.txt'
        with open(txt_path, 'a+') as f:
            print('name:\n', y_name, file=f)
            print('true:\n', y_true, file=f)
            print('predict:\n', y_predict, file=f)
            print('probability:\n', y_probability, file=f)
            print('\n', file=f)

    #     logger.info(y_score, y_pre, y_true)
        

    



if __name__ == "__main__":
    main()