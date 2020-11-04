import os 
import numpy as np 
import nrrd 
import cv2 
import SimpleITK as sitk 
import shutil 
import csv
import logging 
import glob 
import imageio 
import xlrd 
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

# load the changed label case
changed_path = os.path.join(os.path.dirname(path), 'label_change.xlsx')
wb = xlrd.open_workbook(filename=changed_path)
sheet = wb.sheet_by_index(0)
changed_case = sheet.col_values(6)

# build a directory to save images
image_dir = os.path.join(os.path.dirname(path), 'patch_image')
if not os.path.exists(image_dir):
    os.mkdir(image_dir)


# build a directory to save npy
npy_dir = os.path.join(os.path.dirname(path), 'patch_npy')
if not os.path.exists(npy_dir):
    os.mkdir(npy_dir)


# get the case path 
case_list = glob.glob(path + '/*/*')
logger.info(case_list)
for case_path in case_list:
    logger.info(case_path)

    # ignore the mistake case
    if eval(case_path.split('/')[-2]) in changed_case:
        continue

    # get the art nrrd file
    idx = -1
    item_list = os.listdir(case_path)
    item_list.sort()
    for item in item_list:
        idx += 1
        if item.split('_')[-1] == 'ART.nrrd':
            art_mask_slice = idx
        elif item.split('_')[-1] == 'NC.nrrd':
            nc_mask_slice = idx 
        elif item.split('_')[-1] == 'PV.nrrd':
            pv_mask_slice = idx 

    # get three phase image and mask path
    art_image_path = case_path + '/' + item_list[art_mask_slice - 1]
    art_mask_path = case_path + '/' + item_list[art_mask_slice]
    nc_image_path = case_path + '/' + item_list[nc_mask_slice - 1]
    nc_mask_path = case_path + '/' + item_list[nc_mask_slice]
    pv_image_path = case_path + '/' + item_list[pv_mask_slice - 1]
    pv_mask_path = case_path + '/' + item_list[pv_mask_slice]

    # get art image array
    art_array = nib.load(art_image_path).get_data()     # channel last
    nc_array = nib.load(nc_image_path).get_data()     # channel last
    pv_array = nib.load(pv_image_path).get_data()     # channel last

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
            # get the bounding box
            bbox = prop[0].bbox
    
    # expand the bounding box by 20%
    art_bbox_mask = np.zeros((art_mask_array.shape[0], art_mask_array.shape[1]))
    length = int((bbox[2] - bbox[0]) * 0.1)
    height = int((bbox[3] - bbox[1]) * 0.1)
    art_bbox_mask[(bbox[0] - length): (bbox[2] + length + 1), (bbox[1] - height): (bbox[3] + height + 1)] = 1
    
    for i in [largest_slice]:

        art_slice_array = art_array[:, :, i]
        nc_slice_array = nc_array[:, :, i]
        pv_slice_array = pv_array[:, :, i]

        # get mean and std of one slice
        art_mean = art_slice_array.mean()
        art_std = art_slice_array.std()
        nc_mean = nc_slice_array.mean()
        nc_std = nc_slice_array.std()
        pv_mean = pv_slice_array.mean()
        pv_std = pv_slice_array.std()

        # get the lower and upper bound 
        # mean+-3*std get 99.73% data
        art_lower = np.percentile(art_slice_array, 0.14)
        art_upper = np.percentile(art_slice_array, 99.86)
        nc_lower = np.percentile(nc_slice_array, 0.14)
        nc_upper = np.percentile(nc_slice_array, 99.86)
        pv_lower = np.percentile(pv_slice_array, 0.14)
        pv_upper = np.percentile(pv_slice_array, 99.86)

        # truncate the array
        art_slice_array[art_slice_array < art_lower] = art_lower 
        art_slice_array[art_slice_array > art_upper] = art_upper
        nc_slice_array[nc_slice_array < nc_lower] = nc_lower
        nc_slice_array[nc_slice_array > nc_upper] = nc_upper
        pv_slice_array[pv_slice_array < pv_lower] = pv_lower
        pv_slice_array[pv_slice_array > pv_upper] = pv_upper

        # do normalization
        art_slice_array = art_slice_array.astype(dtype=np.float32)
        art_slice_array = (art_slice_array - art_mean) / art_std 
        nc_slice_array = nc_slice_array.astype(dtype=np.float32)
        nc_slice_array = (nc_slice_array - nc_mean) / nc_std 
        pv_slice_array = pv_slice_array.astype(dtype=np.float32)
        pv_slice_array = (pv_slice_array - pv_mean) / pv_std

        # get the roi area
        art_roi_array = np.multiply(art_slice_array, art_bbox_mask)
        nc_roi_array = np.multiply(nc_slice_array, art_bbox_mask)
        pv_roi_array = np.multiply(pv_slice_array, art_bbox_mask)
        
        # get bbox shape
        bbox_roi_shape0 = bbox[2] + 2 * length + 1 - bbox[0]
        bbox_roi_shape1 = bbox[3] + 2 * height + 1 - bbox[1]

        # get bbox roi area
        art_bbox_roi = art_roi_array[art_bbox_mask.astype('int') == 1]
        art_bbox_roi = art_bbox_roi.reshape(bbox_roi_shape0, bbox_roi_shape1)
        art_bbox_roi = cv2.resize(art_bbox_roi, (224, 224))
        nc_bbox_roi = nc_roi_array[art_bbox_mask.astype('int') == 1]
        nc_bbox_roi = nc_bbox_roi.reshape(bbox_roi_shape0, bbox_roi_shape1)
        nc_bbox_roi = cv2.resize(nc_bbox_roi, (224, 224))
        pv_bbox_roi = pv_roi_array[art_bbox_mask.astype('int') == 1]
        pv_bbox_roi = pv_bbox_roi.reshape(bbox_roi_shape0, bbox_roi_shape1)
        pv_bbox_roi = cv2.resize(pv_bbox_roi, (224, 224))

        # set the name of image and npy
        id_num = case_path.split('/')[-2]
        art_image_name = id_num + '_' + str(i) + '_art_' + mvi[id_num] + '.jpg'
        art_npy_name = id_num + '_' + str(i) + '_art_' + mvi[id_num] + '.npy'
        nc_image_name = id_num + '_' + str(i) + '_nc_' + mvi[id_num] + '.jpg'
        nc_npy_name = id_num + '_' + str(i) + '_nc_' + mvi[id_num] + '.npy'
        pv_image_name = id_num + '_' + str(i) + '_pv_' + mvi[id_num] + '.jpg'
        pv_npy_name = id_num + '_' + str(i) + '_pv_' + mvi[id_num] + '.npy'
        
        # save the array to file
        imageio.imwrite(os.path.join(image_dir, art_image_name), art_bbox_roi)
        imageio.imwrite(os.path.join(image_dir, nc_image_name), nc_bbox_roi)
        imageio.imwrite(os.path.join(image_dir, pv_image_name), pv_bbox_roi)
        np.save(os.path.join(npy_dir, art_npy_name), art_bbox_roi)
        np.save(os.path.join(npy_dir, nc_npy_name), nc_bbox_roi)
        np.save(os.path.join(npy_dir, pv_npy_name), pv_bbox_roi)
        
