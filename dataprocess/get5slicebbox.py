import os 
import numpy as np 
import nrrd 
import cv2 
import shutil 
import glob 
import csv 
import xlrd
import logging 
import imageio 
import scipy 
import SimpleITK as sitk
import nibabel as nib 
from PIL import Image 
from skimage import measure, io 



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
        mvi[item[0]] = item[1]
    csv_file.close()
    print('mvi label has been loaded\n')
    return mvi 


def readxls(xls_path, sheet_id, id_col, label_col):
    mvi = {}
    wb = xlrd.open_workbook(filename=xls_path)
    sheet = wb.sheet_by_index(sheet_id)
    id = sheet.col_values(id_col)
    label = sheet.col_values(label_col)
    for i in range(1, len(id)):
        mvi[id[i]] = label[i]
    print('mvi label has been loaded\n')
    return mvi 


def bddir(path):
    image_dir = os.path.join(path, 'bbox_image')
    npy_dir = os.path.join(path, 'bbox_npy')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)

    return image_dir, npy_dir


def resample(origin_spacing, array):
    # origin_spacing = np.zeros((3, 3))
    # origin_spacing[0, 0] = origin[0]
    # origin_spacing[1, 1] = origin[1]
    # origin_spacing[2, 2] = origin[2]
    # print(origin_spacing)
    origin_spacing = np.array(origin_spacing)
    new_spacing = [1, 1, origin_spacing[2]]

    # calculate the resize factor and new array shape
    resize_factor = origin_spacing / new_spacing
    new_real_shape = array.shape * resize_factor
    # print(new_real_shape.shape)
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / array.shape
    # real_resize_factor_list = [real_resize_factor[0, 0], real_resize_factor[1, 1], real_resize_factor[2, 2]]
    
    # do resample 
    new_array = scipy.ndimage.interpolation.zoom(array, real_resize_factor, order=1)

    return new_array


def getImgMask(case_path, item_list, slice_num):
    image_path = case_path + '/' + item_list[slice_num - 1]
    mask_path = case_path + '/' + item_list[slice_num]
    image_array = nib.load(image_path).get_data()
    mask_array, mask_head = nrrd.read(mask_path)
    mask_list = list(set(np.nonzero(mask_array)[-1]))


    origin_mask_spacing = mask_head['space directions']
    origin_mask_spacing = [origin_mask_spacing[0, 0], origin_mask_spacing[1, 1], origin_mask_spacing[2, 2]]
    new_mask_array = resample(origin_mask_spacing, mask_array)

    # read nii file and get inputsize and inputspacing 
    image = sitk.ReadImage(image_path)
    # inputsize = image.GetSize()
    inputspacing = image.GetSpacing()
    # origin_image_spacing = [inputspacing[2], inputspacing[1], inputspacing[0]]
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.transpose(image_array, [2, 1, 0])
    new_image_array = resample(inputspacing, image_array)


    # # define outsize and outspacing
    # outspacing = [1, 1, inputsize[2]]
    # outsize = [0, 0, 0]
    # # calculate outsize
    # transform = sitk.Transform()
    # transform.SetIdentity()
    # outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.499)
    # outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.499)
    # outsize[2] = inputsize[2]
    # # use resampler to resample the image
    # resampler = sitk.ResampleImageFilter()
    # resampler.SetTransform(transform)
    # resampler.SetInterpolator(sitk.sitkBSpline)
    # resampler.SetOutputOrigin(image.GetOrigin())
    # resampler.SetOutputSpacing(outspacing)
    # resampler.SetOutputDirection(image.GetDirection())
    # resampler.SetSize(outsize)
    # new_image = resampler.Execute(image)
    # # get image array data
    # image_array = sitk.GetArrayFromImage(new_image)
    # image_array = np.transpose(image_array, [2, 1, 0])  # (z, y, x) -> (x, y, z)
    # print(image_array.shape, np.sum(image_array))
    # print(np.sum(image_array[:, :, 46]))


    return new_image_array, new_mask_array, mask_list
    # return image_array, mask_array, mask_list



def saveImg(image_array, mask_array, mask_list, test_num, image_dir, npy_dir, phase):
    largest_area = 0
    largest_slice = 0
    for i in mask_list:
        img_labeled = measure.label(mask_array[:, :, i], connectivity=2)
        prop = measure.regionprops(img_labeled)
        if len(prop) != 0:
            area = prop[0].area
            if area > largest_area:
                largest_area = area 
                largest_slice = i 
                bbox = prop[0].bbox 

    bbox_mask = np.zeros((image_array.shape[0], image_array.shape[1]))
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
        
        image_name = test_num + '_' + str(i) + phase + str(int(mvi[test_num])) + '.jpg'
        npy_name = test_num + '_' + str(i) + phase + str(int(mvi[test_num])) + '.npy'

        imageio.imwrite(os.path.join(image_dir, image_name), bbox_array)
        np.save(os.path.join(npy_dir, npy_name), bbox_array)


    


if __name__ == "__main__":
    pwd = os.path.dirname(__file__)
    path = '/media/drs/extra/Datasets/MVI'

    setLog(pwd)

    csv_path = os.path.join(path, 'label.csv')
    xls_path = os.path.join(path, 'Clinical_binary_2.xlsx')
    # mvi = readCSV(csv_path)
    mvi = readxls(xls_path, 4, 0, 1)

    image_dir, npy_dir = bddir(path)

    data_path = os.path.join(path, 'EV3')
    # data_path = os.path.join(path, 'mvi_IV')
    # case_list = glob.glob(data_path + '/*/*')
    case_list = glob.glob(data_path + '/*')

    for case_path in case_list:
        print(case_path)

        idx = -1
        item_list = sorted(os.listdir(case_path))
        for item in item_list:
            idx += 1
            # if item.split('_')[-2] + '_' +item.split('_')[-1]== 'ART_tumormask.nrrd':
            if item.split('_')[-1] == 'ART.nrrd':
                art_mask_slice = idx
            # elif item.split('_')[-2] + '_'  + item.split('_')[-1]== 'NC_tumormask.nrrd':
            elif item.split('_')[-1] == 'NC.nrrd':
                nc_mask_slice = idx 
            # elif item.split('_')[-2] + '_'  + item.split('_')[-1]== 'PV_tumormask.nrrd':
            elif item.split('_')[-1] == 'PV.nrrd':
                pv_mask_slice = idx 
            # elif item.split('_')[-2] + '_'  + item.split('_')[-1]== 'DL_tumormask.nrrd':
            elif item.split('_')[-1] == 'DL.nrrd':
                dl_mask_slice = idx 
            
        art_image_array, art_mask_array, art_mask_list = getImgMask(case_path, item_list, art_mask_slice)
        nc_image_array, nc_mask_array, nc_mask_list = getImgMask(case_path, item_list, nc_mask_slice)
        pv_image_array, pv_mask_array, pv_mask_list = getImgMask(case_path, item_list, pv_mask_slice)
        dl_image_array, dl_mask_array, dl_mask_list = getImgMask(case_path, item_list, dl_mask_slice)

        # test_num = case_path.split('/')[-2]
        test_num = case_path.split('/')[-1]

        saveImg(art_image_array, art_mask_array, art_mask_list, test_num, image_dir, npy_dir, '_art_')
        saveImg(nc_image_array, nc_mask_array, nc_mask_list, test_num, image_dir, npy_dir, '_nc_')
        saveImg(pv_image_array, pv_mask_array, pv_mask_list, test_num, image_dir, npy_dir, '_pv_')
        saveImg(dl_image_array, dl_mask_array, dl_mask_list, test_num, image_dir, npy_dir, '_dl_')