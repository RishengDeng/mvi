import os 
import numpy as np 
import shutil 
import xlrd


def readxlsx(xlsx_path, sheet_id):
    wb = xlrd.open_workbook(filename=xlsx_path)
    sheet = wb.sheet_by_index(sheet_id)
    id = sheet.col_values(0)
    return id 


def builddir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def divide(path, id_list, train_path, val_path):
    for item in os.listdir(path):
        if eval(item.split('_')[0]) in id_list:
            shutil.copy(os.path.join(path, item), os.path.join(val_path, item))
        else:
            shutil.copy(os.path.join(path, item), os.path.join(train_path, item))


if __name__ == "__main__":
    path = '/media/drs/extra/Datasets/MVI'
    xlsx_path = os.path.join(path, 'divide.xlsx')
    # sub0 = readxlsx(xlsx_path, 0)
    # sub1 = readxlsx(xlsx_path, 1)
    # sub2 = readxlsx(xlsx_path, 2)
    # sub3 = readxlsx(xlsx_path, 3)
    sub = readxlsx(xlsx_path, 4)

    # builddir(os.path.join(path, 'train0'))
    # builddir(os.path.join(path, 'val0'))
    # builddir(os.path.join(path, 'train1'))
    # builddir(os.path.join(path, 'val1'))
    # builddir(os.path.join(path, 'train2'))
    # builddir(os.path.join(path, 'val2'))
    # builddir(os.path.join(path, 'train3'))
    # builddir(os.path.join(path, 'val3'))
    builddir(os.path.join(path, 'train'))
    builddir(os.path.join(path, 'val'))

    npy_path = os.path.join(path, 'bbox_npy')

    # divide(npy_path, sub0, os.path.join(path, 'train0'), os.path.join(path, 'val0'))
    # divide(npy_path, sub1, os.path.join(path, 'train1'), os.path.join(path, 'val1'))
    # divide(npy_path, sub2, os.path.join(path, 'train2'), os.path.join(path, 'val2'))
    # divide(npy_path, sub3, os.path.join(path, 'train3'), os.path.join(path, 'val3'))
    divide(npy_path, sub, os.path.join(path, 'train'), os.path.join(path, 'val'))