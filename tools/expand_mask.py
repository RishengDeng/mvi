import os 
import numpy as np 
import nrrd 
import cv2 
import scipy

path = '/home/drs/Desktop/mvi_data/MVI/2081487/4832691'

idx = -1
for item in sorted(os.listdir(path)):
    idx += 1
    if item.endswith('ART.nrrd'):
        break

mask_data, mask_head = nrrd.read(os.path.join(path, sorted(os.listdir(path))[idx]))

mask_slice = list(set(np.nonzero(mask_data)[-1]))

origin_spacing = mask_head['space directions']
new_spacing = [1, 1, origin_spacing[2, 2]]
resize_factor = origin_spacing / new_spacing
new_real_shape = mask_data.shape * resize_factor
new_shape = np.round(new_real_shape)
real_resize_factor = new_shape / mask_data.shape
real_resize_factor_list = [real_resize_factor[0, 0], real_resize_factor[1, 1], real_resize_factor[2, 2]]
new_mask_data = scipy.ndimage.interpolation.zoom(mask_data, real_resize_factor_list, order=0)

kernel_size = 12
kernel = np.ones((kernel_size, kernel_size))

for i in mask_slice:
    new_mask_data[:, :, i] = cv2.dilate(new_mask_data[:, :, i], kernel, iterations=1)    

convert_resize_factor = np.array(mask_data.shape) / np.array(new_mask_data.shape)
expand_mask_data = scipy.ndimage.interpolation.zoom(new_mask_data, convert_resize_factor, order=0)


nrrd_name = path.split('/')[-2] + '_' + path.split('/')[-1] + '_ARTmask_' + str(kernel_size / 2) + 'mm.nrrd'

nrrd.write(os.path.join('/home/drs/Desktop/temp/', nrrd_name), expand_mask_data, mask_head)