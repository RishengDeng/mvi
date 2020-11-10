import os 
import numpy as np 
import nrrd 
import cv2 
import shutil 
import glob 
import csv 
import logging 
import imageio 
import nibabel as nib 
from PIL import Image 
from skimage import measure, io 


pwd = os.path.dirname(__file__)
path = '/media/drs/extra/Datasets/mvi_new'


# set logging information 
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.StreamHandler()
# handler = logging.FileHandler(os.path.join(pwd, 'errors') + '.log', mode='w')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# get the case path 
# data_path = os.path.join(path, 'mvi_Training')
data_path = os.path.join(path, 'mvi_IV')
case_list = glob.glob(data_path + '/*/*')
for case_path in case_list:
    # print(case_path)

    # get mask file
    idx = -1
    item_list = sorted(os.listdir(case_path))
    for item in item_list:
        idx += 1
        if item.split('_')[-1] == 'ART.nrrd':
            art_mask_slice = idx
        elif item.split('_')[-1] == 'NC.nrrd':
            nc_mask_slice = idx 
        elif item.split('_')[-1] == 'PV.nrrd':
            pv_mask_slice = idx 
        elif item.split('_')[-1] == 'DL.nrrd':
            dl_mask_slice = idx 

    # get three phase image and mask path
    art_image_path = case_path + '/' + item_list[art_mask_slice - 1]
    art_mask_path = case_path + '/' + item_list[art_mask_slice]
    nc_image_path = case_path + '/' + item_list[nc_mask_slice - 1]
    nc_mask_path = case_path + '/' + item_list[nc_mask_slice]
    pv_image_path = case_path + '/' + item_list[pv_mask_slice - 1]
    pv_mask_path = case_path + '/' + item_list[pv_mask_slice]
    dl_image_path = case_path + '/' + item_list[dl_mask_slice - 1]
    dl_mask_path = case_path + '/' + item_list[dl_mask_slice]

    # get art mask array
    art_mask_array, _ = nrrd.read(art_mask_path)
    art_mask_slice = list(set(np.nonzero(art_mask_array)[-1]))

    # use the maximum connectivity method to find the largest slice
    largest_area = 0
    largest_slice = 0
    for i in art_mask_slice:
        img_labeled = measure.label(art_mask_array[:, :, i], connectivity=2)
        prop = measure.regionprops(img_labeled)
        area = prop[0].area
        if area > largest_area:
            largest_area = area 
            largest_slice = i 
            # get the bounding box
            bbox = prop[0].bbox

    for i in [largest_slice-2, largest_slice-1, largest_slice, largest_slice+1, largest_slice+2]:
        # omit the mask with no data
        if np.sum(art_mask_array[:, :, i]) == 0:
            logger.info(case_path)