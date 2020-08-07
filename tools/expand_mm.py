import nrrd 
import os 
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


ratio = mask_head['space directions'][0][0]


mm = 10 # the mm you want to expand 
pix = mm / ratio 
pix = int(np.round(pix))

new_mask = np.zeros((mask_data.shape[0], mask_data.shape[1]))

mask_nonzero = np.nonzero(mask_data[..., largest_slice])
mask_idx = list(zip(mask_nonzero[0], mask_nonzero[1]))
for i in range(len(mask_idx)):
    x, y = mask_idx[i]
    for m in range(pix * 2 + 1):
        for n in range(pix * 2 + 1):
            if (0 < x-pix+m < mask_data.shape[0] and 0 < y-pix+n < mask_data.shape[1]):
                new_mask[x-pix+m, y-pix+n] = 1

print(np.nonzero(new_mask))
print(np.nonzero(mask_data[..., largest_slice]))