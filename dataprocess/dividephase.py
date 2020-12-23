import os 
import shutil
import numpy as np 

path = '/media/drs/extra/Datasets/MVI/val/'

phase_path = []

art_path = path + 'art/'
nc_path = path + 'nc/'
pv_path = path + 'pv/'
dl_path = path + 'dl/'

phase_path.append(art_path)
phase_path.append(nc_path)
phase_path.append(pv_path)
phase_path.append(dl_path)


for phase in phase_path:
    if not os.path.exists(phase):
        os.makedirs(phase)


for item in os.listdir(path):
    item_path = os.path.join(path, item)
    if os.path.isdir(item_path):
        continue
    if item.split('_')[-2] == 'art':
        shutil.move(item_path, art_path + item)
    elif item.split('_')[-2] == 'nc':
        shutil.move(item_path, nc_path + item)
    elif item.split('_')[-2] == 'pv':
        shutil.move(item_path, pv_path + item)
    elif item.split('_')[-2] == 'dl':
        shutil.move(item_path, dl_path + item)
    