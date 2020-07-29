import os 
import numpy as np 
import nrrd 
import cv2 
import SimpleITK as sitk 
import shutil 
import nrrd 
import csv 
import logging 
import glob 
import imageio
import nibabel as nib 
from PIL import Image 
from skimage import measure, io 


# set logging information
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# path = '/home/drs/Desktop/mvi_data/MVI'
path = '/media/drs/extra/Datasets/mvi_data/MVI'


# read mvi csv file
csv_file = open(os.path.join(os.path.dirname(path), 'mvi.csv'), 'r')
csv_reader = csv.reader(csv_file)
mvi = {}
for item in csv_reader:
    if csv_reader.line_num == 1:
        continue
    mvi[item[0]] = item[1]
csv_file.close()

logger.info('the csv file has been loaded')


# build a directory to save images
image_dir = os.path.join(os.path.dirname(path), 'dl_images_30')
if not os.path.exists(image_dir):
    os.mkdir(image_dir)


# build a directory to save npy
npy_dir = os.path.join(os.path.dirname(path), 'dl_npy_30')
if not os.path.exists(npy_dir):
    os.mkdir(npy_dir)


# get the case path 
case_list = glob.glob(path + '/*/*')
logger.info(case_list)
for case_path in case_list:
    logger.info(case_path)


    # # get LAVA phase
    # LAVA_list = glob.glob(case_path + '/*LAVA*')
    # LAVA_list.sort()

    # get the art nrrd file
    idx = -1
    item_list = os.listdir(case_path)
    item_list.sort()
    for item in item_list:
        idx += 1
        if item.split('_')[-1] == 'DL.nrrd':
            break
    assert idx > 0

    # get the art and mask and liver path
    art_image_path = case_path + '/' + item_list[idx - 1]
    art_mask_path = case_path + '/' + item_list[idx]
    art_liver_path = case_path + '/' + item_list[idx + 1]

    # get art image array
    art_image = nib.load(art_image_path)
    art_array = art_image.get_data()    #channel last

    # get art mask array
    art_mask_array, _ = nrrd.read(art_mask_path)
    art_mask_slice = list(set(np.nonzero(art_mask_array)[-1]))

    # use the maximum connectivity method to find the largest slice
    largest_area = 0
    largest_slice = 0
    for idx in art_mask_slice:
        img_labeled = measure.label(art_mask_array[:, :, idx], connectivity=2)
        prop = measure.regionprops(img_labeled)
        area = prop[0].area
        if area > largest_area:
            largest_area = area 
            largest_slice = idx 

    # define the kernel to expand the mask
    kernel_size = 30
    kernel = np.ones((kernel_size, kernel_size))
    
    for i in [largest_slice - 2, largest_slice - 1, largest_slice, largest_slice + 1, largest_slice + 2]:
        # omit the mask with no data
        if np.sum(art_mask_array[:, :, i]) == 0:
            continue
        # use cv2.dilate to expand the mask by 10 pixel
        new_mask_array = np.zeros((512, 512))
        new_mask_array = cv2.dilate(art_mask_array[:, :, i], kernel, iterations=1)

        # get the dilated ring of the tumor
        ring_array = art_mask_array[:, :, i] - new_mask_array

        # multiply the mask with art array
        roi_array = np.multiply(new_mask_array, art_array[:, :, i])
        # roi_array = np.multiply(art_mask_array[:, :, i], art_array[:, :, i])

        # get the mask coordinate
        mask = (roi_array > 0)

        # get the lower and upper bound
        lower = np.percentile(roi_array[mask], 0.5)
        upper = np.percentile(roi_array[mask], 99.5)

        # cut the array 
        roi_array[mask & (roi_array < lower)] = lower
        roi_array[mask & (roi_array > upper)] = upper 

        # get mean and std of the image
        mean = roi_array[mask].mean()
        std = roi_array[mask].std()

        # do normalize
        roi_array = roi_array.astype(dtype=np.float32)
        roi_array[mask] = (roi_array[mask] - mean) / std 


        # convert the new array to image and save it to file
        id_num = case_path.split('/')[-2]
        image_name = id_num + '_' + str(i) + '_dl_' + mvi[id_num] + '.jpg'
        npy_name = id_num + '_' + str(i) + '_dl_' + mvi[id_num] + '.npy'
        imageio.imwrite(os.path.join(image_dir, image_name), roi_array)
        # io.imsave(os.path.join(image_dir, image_name), roi_array)
        np.save(os.path.join(npy_dir, npy_name), roi_array)