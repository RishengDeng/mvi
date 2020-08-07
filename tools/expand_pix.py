import nrrd 
import os 
import cv2 
import numpy as np 
from skimage import measure

path = '/home/drs/Desktop/mvi_data/MVI/2081487/4832691'

idx = -1
for item in sorted(os.listdir(path)):
    idx += 1
    if item.endswith('ART.nrrd'):
        break 

mask_data, mask_head = nrrd.read(os.path.join(path, sorted(os.listdir(path))[idx]))


mask_slice = list(set(np.nonzero(mask_data)[-1]))
largest_area = 0
largest_slice = 0
for i in mask_slice:
    img_labeled = measure.label(mask_data[..., i], connectivity=2)
    prop = measure.regionprops(img_labeled)
    area = prop[0].area 
    if area > largest_area:
        largest_slice = i
        largest_area =area 


kernel_size = 10    # (kernel_size / 2) is the pixel you want to expand
kernel = np.ones((kernel_size, kernel_size))

new_mask = np.zeros((mask_data.shape[0], mask_data.shape[1]))
new_mask = cv2.dilate(mask_data[..., largest_slice], kernel, iterations=1)

