import xlrd 
import xlwt 
from xlutils.copy import copy
import random
import numpy as np 

path = '/media/drs/extra/Datasets/mvi_data/dataset_divide.xlsx'

# open a excel
data = xlrd.open_workbook(path)
table = data.sheet_by_index(0)

# read the case ids and labels
cases = table.col_values(5)
labels = table.col_values(6)
# print(cases[0], labels[0])

label0 = []
label1 = []

# devide label0 and label1
for i in range(len(cases)):
    if labels[i] == 0:
        label0.append(cases[i])
    elif labels[i] == 1:
        label1.append(cases[i])

# random shuffle the data
random.shuffle(label0)
random.shuffle(label1)

# open a new workbook and sheet to write data
workbook = copy(data)
sheet = workbook.get_sheet(0)

# seperate 70% data to training and 30% to validation
for i in range(int(np.ceil(len(label0) * 0.7))):
    sheet.write(i, 8, label0[i])
for i in range(int(np.ceil(len(label1) * 0.7))):
    sheet.write(i, 9, label1[i])
for i in range(int(np.floor(len(label0) * 0.3))):
    sheet.write(i, 10, label0[int(np.ceil(len(label0) * 0.7)) + i])
for i in range(int(np.floor(len(label1) * 0.3))):
    sheet.write(i, 11, label1[int(np.ceil(len(label1) * 0.7)) + i])

# save data
workbook.save(path)