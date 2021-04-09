import argparse 
import os 
import logging 
import time 
import numpy as np 
import shutil
import setproctitle
from apex import amp 
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

from model import Resnet18, Resnet50, DilatedResnet, Attention, Res50Clinic, \
    DenseNet, AlexNet, LeNet, DRN22, DRN22_test, ResClinic, DRN22Clinic, \
        DRN54Clinic, AttentionClinic, ClinicRes18, ClinicDRN22, ClinicVgg11, \
            ResClinic2, DRN22Clinic2, ResnetLSTM, ResnetLSTMClinic, ResnetLSTMClinic2, \
                Drn22LSTM, Drn22LSTMClinic, Drn22LSTMClinic2, MulRes18, MulRes18Att, \
                    ResRadio, MulRes18AttRadio
from data import transforms, SinglePhase, MultiPhase, PatchMultiPhase, LSTM, \
    TwoPhases, Radiomics, TwoPhasesRadio
from utils import AverageMeter, accuracy_allslice, accuracy_binary, stack3array, accuracy_valid
from liver_classification import ThreeChannel



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

date = '0220'
best_acc = 0
best_auc = 0


# build directories to save checkpoints and logs
path = os.path.dirname(__file__)
ckpts = os.path.join(path, 'ckpts', date)
logs = os.path.join(path, 'logs', date)
if not os.path.exists(ckpts):
    os.mkdir(ckpts)
if not os.path.exists(logs):
    os.mkdir(logs)


# use logging to record
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger_path = os.path.join(logs, 'art_pv_attention_8' + '.log')
handler = logging.FileHandler(logger_path, mode='w')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# show loss and accuracy in tensorboard
writer = SummaryWriter('logs/runs_6/art_pv_attention_8')


def save_full_ckpt(state, name='full_' + 'art_pv_attention_8'):
    file_name = os.path.join(ckpts, name) + '.pkl'
    torch.save(state, file_name)


def save_ckpt(state, is_best, is_best_auc, epoch, name='art_pv_attention_8'):
    file_name = os.path.join(ckpts, name) + str(epoch + 1) + '.pth.tar'
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(os.path.dirname(file_name), name + 'best.pth.tar'))
    if is_best_auc:
        shutil.copyfile(file_name, os.path.join(os.path.dirname(file_name), name + 'bestauc.pth.tar'))


def main():
    global best_acc, best_auc
    # os.environ['CUDA_VISIBLE_DEVICES'] = 'args.gpu' 
    torch.cuda.set_device(args.gpu)
    print('Use GPU: {} for training'.format(args.gpu))

    # creat model
    # model = Resnet18()
    # model = Resnet50()
    # model = DilatedResnet()
    # model = Attention()
    # model = DenseNet()
    # model = AlexNet()
    # model = LeNet()
    # model = DRN22()
    # model = DRN22_test()
    # model = ResClinic()
    # model = DRN22Clinic()
    # model = Res50Clinic()
    # model = DRN54Clinic()
    # model = AttentionClinic()
    # model = ClinicRes18()
    # model = ClinicDRN22()
    # model = ClinicVgg11()
    # model = ResClinic2()
    # model = DRN22Clinic2()
    # model = ResnetLSTM()
    # model = ResnetLSTMClinic()
    # modle = ResnetLSTMClinic2()
    # model = Drn22LSTM()
    # model = Drn22LSTMClinic()
    # model = Drn22LSTMClinic2()
    # model = MulRes18()
    model = MulRes18Att()
    # model = ResRadio()
    # model = MulRes18AttRadio()
    model = model.cuda(args.gpu)
    logger.info(model)

    # define loss function (criterion) and optimizer 
    weight = torch.FloatTensor([1, 3]).cuda(args.gpu)
    criterion = nn.CrossEntropyLoss(weight=weight)
    # criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(args.gpu)
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
    # train_dir = os.path.join(args.data, 'train', 'pv')
    # val_dir = os.path.join(args.data, 'val', 'pv')
    # train_dir = os.path.join(args.data, 'bbox_npy')
    # val_dir = os.path.join(args.data, 'iv_bbox_npy')

    # train_dataset = SinglePhase(
    # train_dataset = PatchMultiPhase(
    # train_dataset = LSTM(
    train_dataset = TwoPhases(
    # train_dataset = ThreeChannel(
    # train_dataset = Radiomics(
    # train_dataset = TwoPhasesRadio(
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

    # val_dataset = SinglePhase(
    # val_dataset = PatchMultiPhase(
    # val_dataset = LSTM(
    val_dataset = TwoPhases(
    # val_dataset = ThreeChannel(
    # val_dataset = Radiomics(
    # val_dataset = TwoPhasesRadio(
        val_dir, 
        image_size=224, 
        # transforms=transforms(scale=args.scale, angle=args.angle, flip_prob=0.5)
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
        acc, auc = validate(val_loader, model, criterion, epoch, args)

        # record the best accuracy and save checkpoint
        is_best = acc > best_acc
        is_best_auc = auc > best_auc
        save_ckpt({
            'epoch': epoch + 1, 
            'state_dict': model.state_dict(), 
            'best_acc': best_acc, 
            'optimizer': optimizer.state_dict()
        }, is_best, is_best_auc, epoch)
        if epoch == args.epochs - 1:
            save_full_ckpt(model)


def train(train_loader, model, criterion, optimizer, epoch, args):
    # record the losses and accuracy
    Losses = AverageMeter('Loss', ':.4e')
    Accuracy = AverageMeter('Accuracy', ':6.3f')

    # switch to train mode
    model.train()

    # for step, (data, target, id_num, clinic) in enumerate(train_loader):
    # for step, (data, target, id_num, art_radio, pv_radio) in enumerate(train_loader):
    for step, (data, target, id_num) in enumerate(train_loader):
    # for step, (data, target) in enumerate(train_loader):
    # for step, (global_data, local_data, target, id_num) in enumerate(train_loader):
        
        # global_data = global_data.cuda(args.gpu, non_blocking=True)
        # local_data = local_data.cuda(args.gpu, non_blocking=True)
        art_data = data[:, :3, :, :]
        pv_data = data[:, 3:, :, :]
        # art_data = stack3array(data[:, :1, :, :])
        # # pv_data = stack3array(data[:, 1:2, :, :])
        art_data = art_data.cuda(args.gpu, non_blocking=True)
        pv_data = pv_data.cuda(args.gpu, non_blocking=True)
        # data = data.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # clinic = clinic.cuda(args.gpu, non_blocking=True)
        # art_radio = art_radio.cuda(args.gpu, non_blocking=True)
        # pv_radio = pv_radio.cuda(args.gpu, non_blocking=True)

        # art_global = stack3array(global_data[:, 0:1, :, :])
        # art_local = stack3array(local_data[:, 0:1, :, :])
        # nc_global = stack3array(global_data[:, 1:2, :, :])
        # nc_local = stack3array(local_data[:, 1:2, :, :])
        # pv_global = stack3array(global_data[:, 2:3, :, :])
        # pv_local = stack3array(local_data[:, 2:3, :, :])

        # output = model(nc_global, nc_local, art_global, art_local, pv_global, pv_local)
        # output= model(data, clinic)
        # output = model(data)
        output = model(art_data, pv_data)
        # output = model(pv_data)
        # output = model(art_data, pv_data, art_radio, pv_radio)
        loss = criterion(output, target)

        # measure the accuracy and record loss
        acc, acc_0, acc_1 = accuracy_binary(output, target, logger, id_num)
        # acc, acc_0, acc_1 = accuracy_valid(output, target, logger)
        Losses.update(loss.item(), target.numel())
        Accuracy.update(acc / target.numel(), target.numel())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()

        if step % args.print_freq == 0:
            msg = 'Epoch[{:0>3}/{:0>3}], Loss:{:.6f}, Accuracy:{:.3f}'.format(
                epoch+1, args.epochs, Losses.avg, Accuracy.avg)
            logger.info(msg)

    writer.add_scalar('train/loss', Losses.avg, epoch+1)
    writer.add_scalar('train/accuracy', Accuracy.avg, epoch+1)


def validate(val_loader, model, criterion, epoch, args):
    # record the losses and accuracy
    Losses = AverageMeter('Loss', ':.4e')
    Accuracy = AverageMeter('Accuracy', ':6.3f')

    total_num = 0
    total_acc0 = 0
    total_acc1 = 0
    total_acc = 0
    total_0 = 0
    count = 0
    # id_dict = {}
    id_label = {}
    id_slice_sum = {}
    id_slice_num = {}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # for step, (data, target, id_num, clinic) in enumerate(val_loader):
        # for step, (data, target, id_num, art_radio, pv_radio) in enumerate(val_loader):
        for step, (data, target, id_num) in enumerate(val_loader):
        # for step, (data, target) in enumerate(val_loader):
        # for step, (global_data, local_data, target, id_num) in enumerate(val_loader):
        
            # global_data = global_data.cuda(args.gpu, non_blocking=True)
            # local_data = local_data.cuda(args.gpu, non_blocking=True)
            art_data = data[:, :3, :, :]
            pv_data = data[:, 3:, :, :]
            # art_data = stack3array(data[:, :1, :, :])
            # pv_data = stack3array(data[:, 1:2, :, :])
            art_data = art_data.cuda(args.gpu, non_blocking=True)
            pv_data = pv_data.cuda(args.gpu, non_blocking=True)
            # data = data.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # clinic = clinic.cuda(args.gpu, non_blocking=True)
            # art_radio = art_radio.cuda(args.gpu, non_blocking=True)
            # pv_radio = pv_radio.cuda(args.gpu, non_blocking=True)
            
            # art_global = stack3array(global_data[:, 0:1, :, :])
            # art_local = stack3array(local_data[:, 0:1, :, :])
            # nc_global = stack3array(global_data[:, 1:2, :, :])
            # nc_local = stack3array(local_data[:, 1:2, :, :])
            # pv_global = stack3array(global_data[:, 2:3, :, :])
            # pv_local = stack3array(local_data[:, 2:3, :, :])

            # output = model(nc_global, nc_local, art_global, art_local, pv_global, pv_local)
            # output = model(data, clinic)
            # output = model(data)
            output = model(art_data, pv_data)
            # output = model(pv_data)
            # output = model(art_data, pv_data, art_radio, pv_radio)
            loss = criterion(output, target)

            # num = len(list(set(id_num)))

            # measure the accuracy and record loss
            # acc, acc_0, acc_1 = accuracy_binary(output, target, logger, id_num, mode='val')
            # acc, acc_0, acc_1 = accuracy_valid(output, target, logger, mode='val')
            # id_dict, id_label = accuracy_binary(output, target, logger, id_num, id_dict, id_label, mode='val')
            # id_label, id_slice_sum, id_slice_num = accuracy_binary(
            #     output, target, logger, id_num, id_label, 
            #     id_slice_sum, id_slice_num, mode='val')
            id_label, id_slice_sum, id_slice_num = accuracy_allslice(
                output, target, logger, id_num, id_label, 
                id_slice_sum, id_slice_num)
            Losses.update(loss.item(), target.numel())
            # Accuracy.update(acc / num, num)
            # Accuracy.update(acc / target.numel(), target.numel())

            if step % args.print_freq == 0:
                msg = 'Loss:{:.6f}, Accuracy:{:.3f}'.format(Losses.avg, Accuracy.avg)
                logger.info(msg)

            # total_num += target.numel()
            # total_num += num
            # total_acc0 += acc_0
            # total_acc1 += acc_1
            # total_acc += acc

        # for key in id_dict:
        #     if id_dict[key] > 0:
        #         a = 1
        #     else:
        #         a = 0
        #     b = id_label[key]
        #     if a == 0 and b == 0:
        #         total_acc0 += 1
        #         count += 1
        #     elif a == 1 and b == 1:
        #         total_acc1 += 1
        #         count += 1
        
        y_true = []
        y_score = []
        y_predict = []
        y_probability = []
        y_name = []
        
        # use average probability to evaluate
        # for key in id_label:
        #     average_prob = id_slice_sum[key] / id_slice_num[key]
        #     y_true.append(id_label[key])
        #     y_score.append(average_prob[1])
        #     a = np.argmax(average_prob)
        #     b = id_label[key]
        #     y_predict.append(a)
        #     y_probability.append(average_prob)
        #     y_name.append(key)
        #     if a == 0 and b == 0:
        #         total_acc0 += 1
        #         count += 1
        #         total_0 += 1
        #     elif a == 1 and b == 1:
        #         total_acc1 += 1
        #         count += 1 
        #     elif a == 1 and b == 0:
        #         total_0 += 1
        for key in id_label:
            slice_sum = sum(id_slice_sum[key])
            average_prob = slice_sum / id_slice_num[key]
            y_true.append(id_label[key])
            y_score.append(average_prob[1])
            a = np.argmax(average_prob)
            # a = np.argmax(id_slice_sum[key])
            # a = np.sum(a)
            # if a > 0:
            #     a = 1
            b = id_label[key]
            y_predict.append(a)
            y_probability.append(id_slice_sum[key])
            y_name.append(key)
            if a == 0 and b == 0:
                total_acc0 += 1
                count += 1
                total_0 += 1
            elif a == 1 and b == 1:
                total_acc1 += 1
                count += 1 
            elif a == 1 and b == 0:
                total_0 += 1
            

        # calculate accuracy and auc
        total_num = len(id_label)
        total_1 = total_num - total_0
        accuracy = count / total_num 
        y_score = np.array(y_score)
        y_true = np.array(y_true)
        auc = roc_auc_score(y_true, y_score)

        # calculate sensitivity and specificity
        sensitivity = total_acc1 / total_1
        specificity = total_acc0 / total_0
        
        logger.info(
            'total number: {}, total acc0: {}, total acc1: {}, Accuracy: {:.3f}, AUC: {}'.format(
            total_num, total_acc0, total_acc1, accuracy, auc))
        # logger.info(
        #     'total number: {}, total acc0: {}, total acc1: {}, Accuracy: {:.3f}'.format(
        #     total_num, total_acc0, total_acc1, Accuracy.avg))
        

        txt_path = logger_path.split('.')[0] + 'txt'
        with open(txt_path, 'a+') as f:
            print('Epoch:{}/{}'.format(epoch + 1, args.epochs), file=f)
            print('name:\n', y_name, file=f)
            print('true:\n', y_true, file=f)
            print('predict:\n', y_predict, file=f)
            print('probability:\n', y_probability, file=f)
            print('\n', file=f)
    
    
    writer.add_scalar('validate/loss', Losses.avg, epoch+1)
    writer.add_scalar('validate/accuracy', accuracy, epoch+1)
    # writer.add_scalar('validate/accuracy', total_acc / total_num , epoch+1)
    writer.add_scalar('validate/auc', auc, epoch+1)
    writer.add_scalar('validate/sensitivity', sensitivity, epoch+1)
    writer.add_scalar('validate/specificity', specificity, epoch+1)
    
    # return Accuracy.avg
    return accuracy, auc


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == "__main__":
    main()