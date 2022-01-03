import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, width, height, vmax=255, transform=None, target_transform=None):
        '''
        :param img_dir: The directory where all images are located
        :param width: The desired height of output image's width
        :param height: The desired height of output image's height
        :param vmax: The max possible value of pixel of given image, used to normalize the images
        :param transform: The transformation function for images
        :param target_transform: The transformation function for images targets(labels)
        '''
        self.paths = glob.glob(os.path.join(img_dir, "*.JPEG"))
        self.width = width
        self.height = height
        self.vmax = vmax
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        image = image.resize((self.width, self.height), Image.ANTIALIAS)
        if (image.mode != 'RGB'):
            image = image.convert("RGB")
        image = np.asarray(image)/self.vmax
        label = image.copy()
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class CustomGreyImageDataset(Dataset):
    def __init__(self, img_dir, width, height, vmax=255, transform=None, target_transform=None):
        '''
        :param img_dir: The directory where all images are located
        :param width: The desired height of output image's width
        :param height: The desired height of output image's height
        :param vmax: The max possible value of pixel of given image, used to normalize the images
        :param transform: The transformation function for images
        :param target_transform: The transformation function for images targets(labels)
        '''
        self.paths = glob.glob(os.path.join(img_dir, "*.JPEG"))
        self.width = width
        self.height = height
        self.vmax = vmax
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        image = image.resize((self.width, self.height), Image.ANTIALIAS)
        if (image.mode == 'RGB'):
            image = image.convert("L")
        image = np.asarray(image)/self.vmax
        label = image.copy()
        image = np.expand_dims(image, (0))
        label = np.expand_dims(label, (0))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label