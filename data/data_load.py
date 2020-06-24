import numpy as np 
import os 
import torch.nn as nn 
from PIL import Image 
import torch 
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from .transform import pad_data, resize_data


class SinglePhase(Dataset):
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

            # pad the data to square and resize to image size
            stack_array = pad_data(stack_array)
            stack_array = resize_data(stack_array, image_size)
            
            # classify two kinds of data
            array_name = array_list(i * 3)
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
        array_data = np.transpose((2, 0, 1))
        tensor_data = torch.from_numpy(array_data.astype(np.float32))
        return tensor_data, label 


    def __len__(self):
        return len(self.infos)


