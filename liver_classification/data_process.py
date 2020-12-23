import numpy as np 
import shutil 
import os 
import random


def rename(path):
    for phase in os.listdir(path):
        phase_path = os.path.join(path, phase)
        for case in os.listdir(phase_path):
            case_path = os.path.join(phase_path, case)
            os.rename(case_path, case_path.split('.')[0] + '_' + phase + '.jpg')


def divide(path):
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    
    length = len(os.listdir(path))
    idx = 0
    case_list = os.listdir(path)
    random.shuffle(case_list)

    for case in case_list:
        case_path = os.path.join(path, case)
        if os.path.isdir(case_path):
            continue
        if idx < length * 0.25:
            shutil.move(case_path, os.path.join(val_path, case))
        else:
            shutil.move(case_path, os.path.join(train_path, case))
        idx = idx + 1

def movefile(path):
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    
    for phase in os.listdir(path):
        phase_path = os.path.join(path, phase)
        train = os.path.join(phase_path, 'train')
        val = os.path.join(phase_path, 'val')
        for case in os.listdir(train):
            case_path = os.path.join(train, case)
            shutil.move(case_path, os.path.join(train_path, case))
        for case in os.listdir(val):
            case_path = os.path.join(val, case)
            shutil.move(case_path, os.path.join(val_path, case))




if __name__ == "__main__":
    path = '/media/drs/extra/Datasets/4class_liver'
    # rename(path)
    # for phase in os.listdir(path):
    #     phase_path = os.path.join(path, phase)
    #     divide(phase_path)
    movefile(path)
