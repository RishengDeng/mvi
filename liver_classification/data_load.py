import numpy as np
import os 
import cv2
from PIL import Image
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler




# 4 liver disease class classification 
class ThreeChannel(Dataset):
    def __init__(self, root_path, image_size=224, transforms=None):
        infos = []
        image_list = os.listdir(root_path)

        for image in image_list:
            image_path = os.path.join(root_path, image)
            image_array = Image.open(image_path).convert('RGB')
            image_array = np.asarray(image_array)
            image_array = image_array / 255
            image_array = cv2.resize(image_array, (224, 224))
            if image.endswith('2.jpg'):
                infos.append((image_array, 0))
            else:
                infos.append((image_array, 1))
        
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