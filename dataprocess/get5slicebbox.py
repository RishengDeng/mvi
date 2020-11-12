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


# pwd = os.path.dirname(__file__)
# path = '/media/drs/extra/Datasets/mvi_new'


# # set logging information 
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
# # handler = logging.StreamHandler()
# handler = logging.FileHandler(os.path.join(pwd, 'errors') + '.log', mode='w')
# formatter = logging.Formatter('%(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)


# # read mvi csv file 
# csv_file = open(os.path.join(path, 'mvi_train.csv'), 'r')
# csv_reader = csv.reader(csv_file)
# mvi = {}
# for item in csv_reader:
#     if csv_reader.line_num == 1:
#         continue
#     mvi[item[1]] = item[2]
# csv_file.close()
# print('mvi label has been loaded\n')


# # build the directory to save image and npy
# image_dir = os.path.join(path, 'bbox_image')
# npy_dir = os.path.join(path, 'bbox_npy')
# if not os.path.exists(image_dir):
#     os.mkdir(image_dir)
# if not os.path.exists(npy_dir):
#     os.mkdir(npy_dir)


# # get the case path 
# data_path = os.path.join(path, 'mvi_Training')
# # data_path = os.path.join(path, 'mvi_IV')
# case_list = glob.glob(data_path + '/*/*')
# for case_path in case_list:
#     print(case_path)

#     # get mask file
#     idx = -1
#     item_list = sorted(os.listdir(case_path))
#     for item in item_list:
#         idx += 1
#         if item.split('_')[-1] == 'ART.nrrd':
#             art_mask_slice = idx
#         elif item.split('_')[-1] == 'NC.nrrd':
#             nc_mask_slice = idx 
#         elif item.split('_')[-1] == 'PV.nrrd':
#             pv_mask_slice = idx 
#         elif item.split('_')[-1] == 'DL.nrrd':
#             dl_mask_slice = idx 

#     # get three phase image and mask path
#     art_image_path = case_path + '/' + item_list[art_mask_slice - 1]
#     art_mask_path = case_path + '/' + item_list[art_mask_slice]
#     nc_image_path = case_path + '/' + item_list[nc_mask_slice - 1]
#     nc_mask_path = case_path + '/' + item_list[nc_mask_slice]
#     pv_image_path = case_path + '/' + item_list[pv_mask_slice - 1]
#     pv_mask_path = case_path + '/' + item_list[pv_mask_slice]
#     dl_image_path = case_path + '/' + item_list[dl_mask_slice - 1]
#     dl_mask_path = case_path + '/' + item_list[dl_mask_slice]

#     # get image array
#     art_array = nib.load(art_image_path).get_data()     # channel last
#     nc_array = nib.load(nc_image_path).get_data()     # channel last
#     pv_array = nib.load(pv_image_path).get_data()     # channel last
#     dl_array = nib.load(dl_image_path).get_data()     # channel last

#     # get mask array
#     art_mask_array, _ = nrrd.read(art_mask_path)
#     art_mask_slice = list(set(np.nonzero(art_mask_array)[-1]))
#     nc_mask_array, _ = nrrd.read(nc_mask_path)
#     nc_mask_slice = list(set(np.nonzero(nc_mask_array)[-1]))
#     pv_mask_array, _ = nrrd.read(pv_mask_path)
#     pv_mask_slice = list(set(np.nonzero(pv_mask_array)[-1]))
#     dl_mask_array, _ = nrrd.read(dl_mask_path)
#     dl_mask_slice = list(set(np.nonzero(dl_mask_array)[-1]))

#     # use the maximum connectivity method to find the largest slice
#     largest_area = 0
#     largest_slice = 0
#     for i in art_mask_slice:
#         img_labeled = measure.label(art_mask_array[:, :, i], connectivity=2)
#         prop = measure.regionprops(img_labeled)
#         area = prop[0].area
#         if area > largest_area:
#             largest_area = area 
#             largest_slice = i 
#             # get the bounding box
#             bbox = prop[0].bbox
    
#     art_bbox_mask = np.zeros((art_mask_array.shape[0], art_mask_array.shape[1]))
#     length = int((bbox[2] - bbox[0]) * 0.1)
#     height = int((bbox[3] - bbox[1]) * 0.1)
#     art_bbox_mask[(bbox[0] - length): (bbox[2] + length + 1), (bbox[1] - height): (bbox[3] + height + 1)] = 1
    
#     for i in [largest_slice-2, largest_slice-1, largest_slice, largest_slice+1, largest_slice+2]:
#         art_slice_array = art_array[:, :, i]
#         nc_slice_array = nc_array[:, :, i]
#         pv_slice_array = pv_array[:, :, i]
#         dl_slice_array = dl_array[:, :, i]

#         # get mean and std of one slice
#         art_mean = art_slice_array.mean()
#         art_std = art_slice_array.std()
#         nc_mean = nc_slice_array.mean()
#         nc_std = nc_slice_array.std()
#         pv_mean = pv_slice_array.mean()
#         pv_std = pv_slice_array.std()
#         dl_mean = dl_slice_array.mean()
#         dl_std = dl_slice_array.std()

#         # get the lower and upper bound 
#         # mean+-3*std get 99.73% data
#         art_lower = np.percentile(art_slice_array, 0.14)
#         art_upper = np.percentile(art_slice_array, 99.86)
#         nc_lower = np.percentile(nc_slice_array, 0.14)
#         nc_upper = np.percentile(nc_slice_array, 99.86)
#         pv_lower = np.percentile(pv_slice_array, 0.14)
#         pv_upper = np.percentile(pv_slice_array, 99.86)

#         # truncate the array
#         art_slice_array[art_slice_array < art_lower] = art_lower 
#         art_slice_array[art_slice_array > art_upper] = art_upper
#         nc_slice_array[nc_slice_array < nc_lower] = nc_lower
#         nc_slice_array[nc_slice_array > nc_upper] = nc_upper
#         pv_slice_array[pv_slice_array < pv_lower] = pv_lower
#         pv_slice_array[pv_slice_array > pv_upper] = pv_upper

#         # do normalization
#         art_slice_array = art_slice_array.astype(dtype=np.float32)
#         art_slice_array = (art_slice_array - art_mean) / art_std 
#         nc_slice_array = nc_slice_array.astype(dtype=np.float32)
#         nc_slice_array = (nc_slice_array - nc_mean) / nc_std 
#         pv_slice_array = pv_slice_array.astype(dtype=np.float32)
#         pv_slice_array = (pv_slice_array - pv_mean) / pv_std

#         # get the roi area
#         art_roi_array = np.multiply(art_slice_array, art_bbox_mask)
#         nc_roi_array = np.multiply(nc_slice_array, art_bbox_mask)
#         pv_roi_array = np.multiply(pv_slice_array, art_bbox_mask)
        
#         # get bbox shape
#         bbox_roi_shape0 = bbox[2] + 2 * length + 1 - bbox[0]
#         bbox_roi_shape1 = bbox[3] + 2 * height + 1 - bbox[1]

#         # get bbox roi area
#         art_bbox_roi = art_roi_array[art_bbox_mask.astype('int') == 1]
#         art_bbox_roi = art_bbox_roi.reshape(bbox_roi_shape0, bbox_roi_shape1)
#         art_bbox_roi = cv2.resize(art_bbox_roi, (224, 224))
#         nc_bbox_roi = nc_roi_array[art_bbox_mask.astype('int') == 1]
#         nc_bbox_roi = nc_bbox_roi.reshape(bbox_roi_shape0, bbox_roi_shape1)
#         nc_bbox_roi = cv2.resize(nc_bbox_roi, (224, 224))
#         pv_bbox_roi = pv_roi_array[art_bbox_mask.astype('int') == 1]
#         pv_bbox_roi = pv_bbox_roi.reshape(bbox_roi_shape0, bbox_roi_shape1)
#         pv_bbox_roi = cv2.resize(pv_bbox_roi, (224, 224))

#         # set the name of image and npy



def setLog(pwd):
    logger = logging .getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging .FileHandler(os.path.join(pwd, 'errors') + '.log', mode='w')
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)



def readCSV(csv_path):
    mvi = {}
    csv_file = open(csv_path, 'r')
    csv_reader = csv.reader(csv_file)
    for item in csv_reader:
        if csv_reader.line_num == 1:
            continue
        mvi[item[1]] = item[2]
    csv_file.close()
    print('mvi label has been loaded\n')
    return mvi 



def bddir(path):
    image_dir = os.path.join(path, 'iv_bbox_image')
    npy_dir = os.path.join(path, 'iv_bbox_npy')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)

    return image_dir, npy_dir



def getImgMask(case_path, item_list, slice_num):
    image_path = case_path + '/' + item_list[slice_num - 1]
    mask_path = case_path + '/' + item_list[slice_num]
    image_array = nib.load(image_path).get_data()
    mask_array, _ = nrrd.read(mask_path)
    mask_list = list(set(np.nonzero(mask_array)[-1]))

    return image_array, mask_array, mask_list



def saveImg(image_array, mask_array, mask_list, test_num, image_dir, npy_dir, phase):
    largest_area = 0
    largest_slice = 0
    for i in mask_list:
        img_labeled = measure.label(mask_array[:, :, i], connectivity=2)
        prop = measure.regionprops(img_labeled)
        area = prop[0].area
        if area > largest_area:
            largest_area = area 
            largest_slice = i 
            bbox = prop[0].bbox 
    
    bbox_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]))
    length = int((bbox[2] - bbox[0]) * 0.1)
    height = int((bbox[3] - bbox[1]) * 0.1)
    bbox_mask[(bbox[0] - length): (bbox[2] + length + 1), (bbox[1] - height): (bbox[3] + height + 1)] = 1

    for i in [largest_slice-2, largest_slice-1, largest_slice, largest_slice+1, largest_slice+2]:
        slice_array = image_array[:, :, i]
        mean = slice_array.mean()
        std = slice_array.std()
        lower = np.percentile(slice_array, 0.14)
        upper = np.percentile(slice_array, 99.86)
        slice_array[slice_array < lower] = lower 
        slice_array[slice_array > upper] = upper 
        slice_array = slice_array.astype(dtype=np.float32)
        slice_array = (slice_array - mean) / std 

        roi_array = np.multiply(slice_array, bbox_mask)
        bbox_shape0 = bbox[2] + 2 * length + 1 - bbox[0]
        bbox_shape1 = bbox[3] + 2 * height + 1 - bbox[1]
        
        bbox_array = roi_array[bbox_mask.astype('int') == 1]
        bbox_array = bbox_array.reshape(bbox_shape0, bbox_shape1)
        bbox_array = cv2.resize(bbox_array, (224, 224))
        
        image_name = test_num + '_' + str(i) + phase + mvi[test_num] + '.jpg'
        npy_name = test_num + '_' + str(i) + phase + mvi[test_num] + '.npy'

        imageio.imwrite(os.path.join(image_dir, image_name), bbox_array)
        np.save(os.path.join(npy_dir, npy_name), bbox_array)


    


if __name__ == "__main__":
    pwd = os.path.dirname(__file__)
    path = '/media/drs/extra/Datasets/mvi_new'

    setLog(pwd)

    csv_path = os.path.join(path, 'mvi_train.csv')
    mvi = readCSV(csv_path)

    image_dir, npy_dir = bddir(path)

    # data_path = os.path.join(path, 'mvi_Training')
    data_path = os.path.join(path, 'mvi_IV')
    case_list = glob.glob(data_path + '/*/*')

    for case_path in case_list:
        print(case_path)

        idx = -1
        item_list = sorted(os.listdir(case_path))
        for item in item_list:
            idx += 1
            if item.split('_')[-2] + '_' +item.split('_')[-1]== 'ART_tumormask.nrrd':
                art_mask_slice = idx
            elif item.split('_')[-2] + '_'  + item.split('_')[-1]== 'NC_tumormask.nrrd':
                nc_mask_slice = idx 
            elif item.split('_')[-2] + '_'  + item.split('_')[-1]== 'PV_tumormask.nrrd':
                pv_mask_slice = idx 
            elif item.split('_')[-2] + '_'  + item.split('_')[-1]== 'DL_tumormask.nrrd':
                dl_mask_slice = idx 
            
        art_image_array, art_mask_array, art_mask_list = getImgMask(case_path, item_list, art_mask_slice)
        nc_image_array, nc_mask_array, nc_mask_list = getImgMask(case_path, item_list, nc_mask_slice)
        pv_image_array, pv_mask_array, pv_mask_list = getImgMask(case_path, item_list, pv_mask_slice)
        dl_image_array, dl_mask_array, dl_mask_list = getImgMask(case_path, item_list, dl_mask_slice)

        test_num = case_path.split('/')[-1]

        saveImg(art_image_array, art_mask_array, art_mask_list, test_num, image_dir, npy_dir, '_art_')
        saveImg(nc_image_array, nc_mask_array, nc_mask_list, test_num, image_dir, npy_dir, '_nc_')
        saveImg(pv_image_array, pv_mask_array, pv_mask_list, test_num, image_dir, npy_dir, '_pv_')
        saveImg(dl_image_array, dl_mask_array, dl_mask_list, test_num, image_dir, npy_dir, '_dl_')