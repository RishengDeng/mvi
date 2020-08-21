import os 
import numpy as np 
import glob 


path = '/media/drs/extra/Datasets/mvi_data/Replace'
case_list = glob.glob(path + '/*/*')

phase_list = ['NC', 'ART', 'PV', 'DL']

for case in case_list:
    flag = 0
    for i in os.listdir(case):
        for phase in phase_list:
            # if i.split('_')[-2] == phase:
            #     os.rename(os.path.join(case, i), os.path.join(case, i.replace('_tumormask.nrrd', '.nrrd')))
            if i.split('_')[-1] == phase + '.nrrd':
                flag += 1
    if flag != 4:
        print(case)