import os 
import numpy as np
import nrrd 
import cv2 
import shutil 
import csv 
import glob 
import logging 
import imageio
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


path = '/home/drs/Desktop/mvi_data/MVI'


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
image_dir = os.path.join(os.path.dirname(path), 'image')
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

# build the directory to save npy
npy_dir = os.path.join(os.path.dirname(path), 'npy')
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

    # get art mask array
    art_mask_array, _ = nrrd.read(art_mask_path)
    art_mask_slice = list(set(np.nonzero(art_mask_array)[-1]))

    # use the maximum connectivity method to find the largest slice
    largest_area = 0
    largest_slice = 0
    for idx in  art_mask_slice:
        img_labeled = measure.label(art_mask_array[:, :, idx], connectivity=2)
        prop = measure.regionprops(img_labeled)
        area = prop[0].area 
        if area > largest_area:
            largest_area = area 
            largest_slice = idx 

    # read nii file
    art_image = sitk.ReadImage(art_image_path)

    # get inputsize and inputspacing
    inputsize = art_image.GetSize()
    inputspacing = art_image.GetSpacing()

    # define outsize and outspaceing
    outspacing = [1, 1, inputspacing[2]]
    outsize = [0, 0, 0]

    # calculate outsize
    transform = sitk.Transform()
    transform.SetIdentity()
    outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
    outsize[2] = inputsize[2]

    # use resampler to resample the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(art_image.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(art_image.GetDirection())
    resampler.SetSize(outsize)
    art_image_new = resampler.Execute(art_image)

    # get art array data
    art_array = sitk.GetArrayFromImage(art_image_new)   
    art_array = np.transpose(art_array, [2, 1, 0])  # (z, y, x) -> (x, y, z)