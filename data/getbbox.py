import os 
import numpy as np
import nrrd 
import cv2 
import shutil 
import csv 
import glob 
import logging 
import imageio
import scipy 
import nibabel as nib 
from skimage import measure, io 
import SimpleITK as sitk 

# set logging information
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# path = '/home/drs/Desktop/mvi_data/MVI'
path = '/media/drs/extra/Datasets/mvi_data/MVI'


# get the data label from csv file
csv_file = open(os.path.join(os.path.dirname(path), 'mvi.csv'), 'r')
csv_reader = csv.reader(csv_file)
mvi = {}
for item in csv_reader:
    if csv_reader.line_num == 1:
        continue
    mvi[item[0]] = item[1]
csv_file.close()

logger.info('Load the csv file')


# build the directory to save image
image_dir = os.path.join(os.path.dirname(path), 'art_bbox_image')
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

# build the directory to save npy
npy_dir = os.path.join(os.path.dirname(path), 'art_bbox_npy')
if not os.path.exists(npy_dir):
    os.mkdir(npy_dir)


# get the case path
case_list = glob.glob(path + '/*/*')
logger.info(case_list)
for case_path in case_list:
    logger.info(case_path)

    # get the art nrrd file
    idx = -1
    item_list = os.listdir(case_path)
    item_list.sort()
    for item in item_list:
        idx += 1
        if item.split('_')[-1] == 'ART.nrrd':
            break
    assert idx > 0

    # get the path of art, mask and liver
    art_image_path = case_path + '/' + item_list[idx - 1]
    art_mask_path = case_path + '/' + item_list[idx]
    art_liver_path = case_path + '/' + item_list[idx + 1]

    # get art image array
    art_image = nib.load(art_image_path)
    art_array = art_image.get_data()

    # get art mask array and mask head
    art_mask_array, art_mask_head = nrrd.read(art_mask_path)
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
            # get the bounding box
            bbox = prop[0].bbox
    
    art_bbox_mask = np.zeros((art_mask_array.shape[0], art_mask_array.shape[1]))
    for i in range(art_mask_array.shape[0]):
        for j in range(art_mask_array.shape[1]):
            if i >= bbox[0] and i <= bbox[2] and j >= bbox[1] and j <= bbox[3]:
                art_bbox_mask[i, j] = 1

    for i in art_mask_slice:
        art_slice_array = art_array[:, :, i]
        # get the mean and std of one slice
        mean = art_slice_array.mean()
        std = art_slice_array.std()

        # get the lower and upper bound 
        # mean+-3*std get 99.73% data
        lower = np.percentile(art_slice_array, 0.14)
        upper = np.percentile(art_slice_array, 99.86)

        # truncate the array
        art_slice_array[art_slice_array < lower] = lower 
        art_slice_array[art_slice_array > upper] = upper 

        # do normalization
        art_slice_array = art_slice_array.astype(dtype=np.float32)
        art_slice_array = (art_slice_array - mean) / std 

        # multiply the bbox and art array
        roi_array = np.multiply(art_slice_array, art_bbox_mask)        

        # conver the new array to image and save it to file
        id_num = case_path.split('/')[-2]
        image_name = id_num + '_' + str(i) + '_art_' + mvi[id_num] + '.jpg'
        npy_name = id_num + '_' + str(i) + '_art_' + mvi[id_num] + '.npy'
        imageio.imwrite(os.path.join(image_dir, image_name), roi_array)
        np.save(os.path.join(npy_dir, npy_name), roi_array)