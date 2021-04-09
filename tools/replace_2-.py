import os
import numpy as np 
import glob

path = '/media/drs/extra/Datasets/MVI/bbox_npy'

case_list = glob.glob(path + '/*/*')

for case in case_list:
    oldname = os.path.basename(case)
    oldname_1 = os.path.splitext(oldname)[0]
    # print(oldname_1)
    newname = oldname.replace('_', '-', 2)
    # print(newname)
    case_path = os.path.dirname(case)
    new_path = os.path.join(case_path, newname)
    # print(new_path)
    os.rename(case, new_path)