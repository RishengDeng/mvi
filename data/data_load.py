import numpy as np 
import os 
import torch.nn as nn 
from PIL import Image 
import torch 
import xlrd
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from .transform import pad_data, resize_data


class SinglePhase1(Dataset):
    def __init__(self, root_path, image_size=224, transforms=None):
        infos = []
        array_list = os.listdir(root_path)
        array_list.sort()
        
        for i in range(int(len(array_list) / 3)):
            
            # load the npy file
            array0 = np.load(os.path.join(root_path, array_list[i * 3]))
            array1 = np.load(os.path.join(root_path, array_list[i * 3 + 1]))
            array2 = np.load(os.path.join(root_path, array_list[i * 3 + 2]))
            
            # merge three arrays into one
            stack_array = np.array((array0, array1, array2))
            stack_array = np.transpose(stack_array, [1, 2, 0])

            # pad the data to square and resize to image size
            stack_array = pad_data(stack_array)
            stack_array = resize_data(stack_array, image_size)
            
            # classify two kinds of data
            array_name = array_list[i * 3]
            if array_name.endswith('0.npy'):
                infos.append((stack_array, 0))
            elif array_name.endswith('1.npy'):
                infos.append((stack_array, 1))
        
        self.infos = infos 
        self.transforms = transforms
            

    def __getitem__(self, index):
        array_data, label = self.infos[index]
        if self.transforms is not None:
            array_data = self.transforms(array_data)

        # convert the data to c*h*w and change it to tensor
        array_data = np.transpose(array_data, [2, 0, 1])
        tensor_data = torch.from_numpy(array_data.astype(np.float32))
        return tensor_data, label 


    def __len__(self):
        return len(self.infos)



class SinglePhase2(Dataset):
    def __init__(self, root_path, image_size=224, transforms=None):
        infos = []
        array_list = os.listdir(root_path)
        array_list.sort()
        
        for i in range(len(array_list)):
            
            # load the npy file
            array = np.load(os.path.join(root_path, array_list[i]))
            
            # merge three arrays into one
            stack_array = np.array((array, array, array))
            stack_array = np.transpose(stack_array, [1, 2, 0])

            # pad the data to square and resize to image size
            stack_array = pad_data(stack_array)
            stack_array = resize_data(stack_array, image_size)
            
            # classify two kinds of data
            array_name = array_list[i]
            id_num = array_name.split('_')[0]
            if array_name.endswith('0.npy'):
                infos.append((stack_array, 0, id_num))
            elif array_name.endswith('1.npy'):
                infos.append((stack_array, 1, id_num))
            
        self.infos = infos 
        self.transforms = transforms
            

    def __getitem__(self, index):
        array_data, label, id_num = self.infos[index]
        if self.transforms is not None:
            array_data = self.transforms(array_data)

        # convert the data to c*h*w and change it to tensor
        array_data = np.transpose(array_data, [2, 0, 1])
        tensor_data = torch.from_numpy(array_data.astype(np.float32))
        return tensor_data, label, id_num


    def __len__(self):
        return len(self.infos)



class SinglePhase3(Dataset):
    def __init__(self, root_path, image_size=224, transforms=None):
        infos = []
        array_list = os.listdir(root_path)
        array_list.sort()
        
        for i in range(len(array_list) - 2):
            # make sure three slices belong to the same patient
            if not (array_list[i].split('_')[0] == array_list[i + 1].split('_')[0] == array_list[i + 2].split('_')[0]):
                continue

            # load the npy file
            array0 = np.load(os.path.join(root_path, array_list[i]))
            array1 = np.load(os.path.join(root_path, array_list[i + 1]))
            array2 = np.load(os.path.join(root_path, array_list[i + 2]))
            
            # merge three arrays into one
            stack_array = np.array((array0, array1, array2))
            stack_array = np.transpose(stack_array, [1, 2, 0])

            # pad the data to square and resize to image size
            stack_array = pad_data(stack_array)
            stack_array = resize_data(stack_array, image_size)
            
            # classify two kinds of data
            array_name = array_list[i]
            id_num = array_name.split('_')[0]
            if array_name.endswith('0.npy'):
                infos.append((stack_array, 0, id_num))
            elif array_name.endswith('1.npy'):
                infos.append((stack_array, 1, id_num))
        
        self.infos = infos 
        self.transforms = transforms
            

    def __getitem__(self, index):
        array_data, label, id_num = self.infos[index]
        if self.transforms is not None:
            array_data = self.transforms(array_data)

        # convert the data to c*h*w and change it to tensor
        array_data = np.transpose(array_data, [2, 0, 1])
        tensor_data = torch.from_numpy(array_data.astype(np.float32))
        return tensor_data, label, id_num


    def __len__(self):
        return len(self.infos)



# get the three nearby slice and add clinic data
class SinglePhase5(Dataset):
    def __init__(self, root_path, image_size=224, transforms=None):
        infos = []
        array_list = os.listdir(root_path)
        array_list.sort()

        # get the clinic vector through excel
        clinic = {}
        clinic_path = os.path.join(os.path.dirname(os.path.dirname(root_path)), 'clinic_vector.xlsx')
        wb = xlrd.open_workbook(filename=clinic_path)
        sheet = wb.sheet_by_index(0)
        for i in range(1, sheet.nrows):
            clinic[str(int(sheet.cell_value(i, 0)))] = sheet.row_values(i)[2:-1]
        
        for i in range(len(array_list) - 2):
            # make sure three slices belong to the same patient
            if not (array_list[i].split('_')[0] == array_list[i + 1].split('_')[0] == array_list[i + 2].split('_')[0]):
                continue

            # load the npy file
            array0 = np.load(os.path.join(root_path, array_list[i]))
            array1 = np.load(os.path.join(root_path, array_list[i + 1]))
            array2 = np.load(os.path.join(root_path, array_list[i + 2]))
            
            # merge three arrays into one
            stack_array = np.array((array0, array1, array2))
            stack_array = np.transpose(stack_array, [1, 2, 0])

            # pad the data to square and resize to image size
            stack_array = pad_data(stack_array)
            stack_array = resize_data(stack_array, image_size)
            
            # classify two kinds of data
            array_name = array_list[i]
            id_num = array_name.split('_')[0]
            clinic_vector = clinic[id_num]
            if array_name.endswith('0.npy'):
                infos.append((stack_array, 0, id_num, clinic_vector))
            elif array_name.endswith('1.npy'):
                infos.append((stack_array, 1, id_num, clinic_vector))
        
        self.infos = infos 
        self.transforms = transforms
            

    def __getitem__(self, index):
        array_data, label, id_num, clinic_vector = self.infos[index]
        if self.transforms is not None:
            array_data = self.transforms(array_data)

        # convert the data to c*h*w and change it to tensor
        array_data = np.transpose(array_data, [2, 0, 1])
        tensor_data = torch.from_numpy(array_data.astype(np.float32))
        clinic_vector = torch.from_numpy(np.array(clinic_vector).astype(np.float32))
        return tensor_data, label, id_num, clinic_vector


    def __len__(self):
        return len(self.infos)


# copy one slice three times and add clinic data
class SinglePhase4(Dataset):
    def __init__(self, root_path, image_size=224, transforms=None):
        infos = []
        array_list = os.listdir(root_path)
        array_list.sort()

        # get the clinic vector through excel
        clinic = {}
        clinic_path = os.path.join(os.path.dirname(os.path.dirname(root_path)), 'clinic_vector.xlsx')
        wb = xlrd.open_workbook(filename=clinic_path)
        sheet = wb.sheet_by_index(0)
        for i in range(1, sheet.nrows):
            clinic[str(int(sheet.cell_value(i, 0)))] = sheet.row_values(i)[2:-1]
        
        for i in range(len(array_list)):
            
            # load the npy file
            array = np.load(os.path.join(root_path, array_list[i]))
            
            # merge three arrays into one
            stack_array = np.array((array, array, array))
            stack_array = np.transpose(stack_array, [1, 2, 0])

            # pad the data to square and resize to image size
            stack_array = pad_data(stack_array)
            stack_array = resize_data(stack_array, image_size)
            
            # classify two kinds of data
            array_name = array_list[i]
            id_num = array_name.split('_')[0]
            clinic_vector = clinic[id_num]
            if array_name.endswith('0.npy'):
                infos.append((stack_array, 0, id_num, clinic_vector))
            elif array_name.endswith('1.npy'):
                infos.append((stack_array, 1, id_num, clinic_vector))
            
        self.infos = infos 
        self.transforms = transforms
            

    def __getitem__(self, index):
        array_data, label, id_num, clinic_vector = self.infos[index]
        if self.transforms is not None:
            array_data = self.transforms(array_data)

        # convert the data to c*h*w and change it to tensor
        array_data = np.transpose(array_data, [2, 0, 1])
        tensor_data = torch.from_numpy(array_data.astype(np.float32))
        clinic_vector = torch.from_numpy(np.array(clinic_vector).astype(np.float32))
        return tensor_data, label, id_num, clinic_vector


    def __len__(self):
        return len(self.infos)


# random data 
class SinglePhase(Dataset):
    def __init__(self, root_path, image_size=224, transforms=None):
        infos = []
        array_list = os.listdir(root_path)
        array_list.sort()

        # get the clinic vector through excel
        clinic = {}
        clinic_path = os.path.join(os.path.dirname(os.path.dirname(root_path)), 'clinic_vector.xlsx')
        wb = xlrd.open_workbook(filename=clinic_path)
        sheet = wb.sheet_by_index(0)
        for i in range(1, sheet.nrows):
            clinic[str(int(sheet.cell_value(i, 0)))] = sheet.row_values(i)[2:-1]
        
        for i in range(len(array_list)):
            
            # load the npy file
            array = np.load(os.path.join(root_path, array_list[i]))
            
            # merge three arrays into one
            stack_array = np.array((array, array, array))
            stack_array = np.transpose(stack_array, [1, 2, 0])

            # pad the data to square and resize to image size
            stack_array = pad_data(stack_array)
            stack_array = resize_data(stack_array, image_size)
            
            # classify two kinds of data
            array_name = array_list[i]
            id_num = array_name.split('_')[0]
            clinic_vector = clinic[id_num]
            if array_name.endswith('0.npy'):
                infos.append((stack_array, 0, id_num, clinic_vector))
            elif array_name.endswith('1.npy'):
                infos.append((stack_array, 1, id_num, clinic_vector))
            
        self.infos = infos 
        self.transforms = transforms
            

    def __getitem__(self, index):
        array_data, label, id_num, clinic_vector = self.infos[index]
        if self.transforms is not None:
            array_data = self.transforms(array_data)

        # convert the data to c*h*w and change it to tensor
        array_data = np.transpose(array_data, [2, 0, 1])
        tensor_data = torch.from_numpy(array_data.astype(np.float32))
        clinic_vector = torch.from_numpy(np.array(clinic_vector).astype(np.float32))
        return tensor_data, label, id_num, clinic_vector


    def __len__(self):
        return len(self.infos)

