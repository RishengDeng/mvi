import numpy as np 
from skimage.transform import rescale, rotate, resize 
from torchvision.transforms import Compose


def transforms(scale=None, angle=None, flip_prob=None):
    transform_list = []

    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))

    return Compose(transform_list)


class Scale(object):
    def __init__(self, scale):
        self.scale = scale 

    def __call__(self, image):
        img_size = image.shape[0]
        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)
        image = rescale(
            image, 
            (scale, scale),
            multichannel=True, 
            mode='constant', 
            anti_aliasing=False, 
        )

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))), ) * 2 + ((0, 0), )
            image = np.pad(image, padding, mode='constant', constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]

        return image 


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle 

    def __call__(self, image):
        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode='constant')
        return image


class HorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if np.random.rand() > self.flip_prob:
            return image
        
        image = np.fliplr(image).copy()
        return image


def pad_data(data):
    a = data.shape[0]
    b = data.shape[1]
    if a == b:
        return data
    diff = (max(a, b) - min(a, b)) / 2.0
    if a > b:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
    else:
        padding = ((int(np.floor(diff)), int(np.ceil(diff))), (0, 0), (0, 0))
    data = np.pad(data, padding, mode='constant', constant_values=0)
    return data


def resize_data(data, size):
    data_shape = data.shape 
    output_shape = (size, size, data_shape[2])
    data = resize(
        data, 
        output_shape=output_shape, 
        order=2, 
        # mode='constant',
        mode='reflect',
        cval=0,
        anti_aliasing=False,
    )
    return data 