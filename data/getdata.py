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
image_dir = os.path.join(os.path.dirname(path), 'art_image')
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

# build the directory to save npy
npy_dir = os.path.join(os.path.dirname(path), 'art_npy')
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

    # get art mask array and mask head
    art_mask_array, art_mask_head = nrrd.read(art_mask_path)
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

    # get the origin spacing of mask and set new spacing
    art_origin_mask_spacing = art_mask_head['space directions']
    art_new_sapcing = [1, 1, art_origin_mask_spacing[2, 2]]

    # calculate the resize factor and new mask shape
    resize_factor = art_origin_mask_spacing / art_new_sapcing
    art_mask_new_real_shape = art_mask_array.shape * resize_factor
    art_mask_new_shape = np.round(art_mask_new_real_shape)
    real_resize_factor = art_mask_new_shape / art_mask_array.shape 
    real_resize_factor_list = [real_resize_factor[0, 0], real_resize_factor[1, 1], real_resize_factor[2, 2]]

    # do resample of mask array
    art_mask_new_array = scipy.ndimage.interpolation.zoom(art_mask_array, real_resize_factor_list)

    # read nii file and get inputsize and inputspacing
    art_image = sitk.ReadImage(art_image_path)
    inputsize = art_image.GetSize()
    inputspacing = art_image.GetSpacing()

    # define outsize and outspaceing
    outspacing = [1, 1, inputspacing[2]]
    outsize = [0, 0, 0]

    # calculate outsize
    transform = sitk.Transform()
    transform.SetIdentity()
    outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.499)
    outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.499)
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

    # define the kernel to expand the mask
    kernel_size = 6
    kernel = np.ones((kernel_size, kernel_size))

    for i in [largest_slice-2, largest_slice-1, largest_slice, largest_slice+1, largest_slice+2]:
        # omit the mask with no data
        if np.sum(art_mask_new_array[:, :, i]) == 0:
            continue
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

        # expand the mask 
        print(art_mask_new_array.shape)
        expand_mask_array = np.zeros((art_mask_new_array.shape[0], art_mask_new_array.shape[1]))
        expand_mask_array = cv2.dilate(art_mask_new_array[:, :, i], kernel, iterations=1)

        # multiply the mask and art array
        roi_array = np.multiply(art_slice_array, art_mask_new_array[:, :, i])
        # roi_array = np.multiply(art_slice_array, expand_mask_array)

        # conver the new array to image and save it to file
        id_num = case_path.split('/')[-2]
        image_name = id_num + '_' + str(i) + '_art_' + mvi[id_num] + '.jpg'
        npy_name = id_num + '_' + str(i) + '_art_' + mvi[id_num] + '.npy'
        imageio.imwrite(os.path.join(image_dir, image_name), roi_array)
        np.save(os.path.join(npy_dir, npy_name), roi_array)