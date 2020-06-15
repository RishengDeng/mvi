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
import nibabel as nib 
from PIL import Image 
from skimage import measure


# set logging information
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


path = '/home/drs/Desktop/mvi_data/MVI'


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
image_dir = os.path.join(os.path.dirname(path), 'art')
if not os.path.exists(image_dir):
    os.mkdir(image_dir)


# get the case path 
case_list = glob.glob('/home/drs/Desktop/mvi_data/MVI/*/*')
for case_path in case_list:
    logger.info(case_path)


    # get LAVA phase
    LAVA_list = glob.glob(case_path + '/*LAVA*')
    LAVA_list.sort()

    # get the art phase
    art_image_path = LAVA_list[3]
    art_mask_path = LAVA_list[4]
    art_liver_path = LAVA_list[5]