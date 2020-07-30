import shutil 
import numpy as np
import os 
import xlrd 

path = '/media/drs/extra/Datasets/mvi_data/art_npy'

# build the directories to divide data
# label0_path = os.path.join(path, 'label0')
# label1_path = os.path.join(path, 'label1')
train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'val')

# pathes = [label0_path, label1_path, train_path, val_path]
pathes = [train_path, val_path]
for i in pathes:
    if not os.path.exists(i):
        os.mkdir(i)


# # separate label0 and label1
# for item in os.listdir(path):
#     print(item)
#     item_path = os.path.join(path, item)
#     if item.endswith('1.npy'):
#         shutil.move(item_path, os.path.join(label1_path, item))
#     elif item.endswith('0.npy'):
#         shutil.move(item_path, os.path.join(label0_path, item))


# # divide images into two subsets
# count0 = 0
# count1 = 0

# label0_list = os.listdir(label0_path)
# label0_list.sort()
# label1_list = os.listdir(label1_path)
# label1_list.sort()

# for item in label0_list:
#     print(item)
#     if count0 < 534:
#         shutil.move(os.path.join(label0_path, item), os.path.join(train_path, item))
#     else:
#         shutil.move(os.path.join(label0_path, item), os.path.join(val_path, item))
#     count0 += 1

# for item in label1_list:
#     print(item)
#     if count1 < 156:
#         shutil.move(os.path.join(label1_path, item), os.path.join(val_path, item))
#     else:
#         shutil.move(os.path.join(label1_path, item), os.path.join(train_path, item))
#     count1 += 1


# devide the date through excel file
excel_file = os.path.join(os.path.dirname(path), 'dataset_divide.xlsx')
wb = xlrd.open_workbook(filename=excel_file)
sheet = wb.sheet_by_index(0)

# get the train and val list
train_col = sheet.col_values(13)
train_id = train_col[1:]
val_col = sheet.col_values(15)
val_id = []
for item in val_col:
    if isinstance(item, str):
        continue
    val_id.append(item)

for item in os.listdir(path):
    if item == 'train' or item == 'val':
        continue
    if eval(item.split('_')[0]) in train_id:
        shutil.move(os.path.join(path, item), os.path.join(train_path, item))
    elif eval(item.split('_')[0]) in val_id:
        shutil.move(os.path.join(path, item), os.path.join(val_path, item))