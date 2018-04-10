import os
from torch.utils import data
from PIL import Image
from pathlib import Path
import numpy as np
import torchvision

classes = [
    'background',
    'airplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'motorcycle',
    'person',
    'plant',
    'sheep',
    'sofa',
    'train',
    'monitor',
]

#for now will be set to background
void_class_index = 255

class PascalSegDataset(data.Dataset) :

    def __init__(self, root, split, resize_to, transform=None) :
        self.root = Path(root)

        split = self.root/'ImageSets'/'Segmentation'/(split+'.txt')
        with open(split) as splitfile :
            split = splitfile.readlines()

        split = [s.strip() for s in split]

        self.image_names = split
        self.resize_to = resize_to
        self.transform = transform
        if self.transform is None :
            self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index) :
        imname = self.image_names[index]
        path = self.root/'JPEGImages'/f'{imname}.jpg'
        seg_path = self.root/'SegmentationClass'/f'{imname}.png'
        img = Image.open(path)
        img = img.convert('RGB').resize(self.resize_to)

        label = Image.open(seg_path).resize(self.resize_to)
        label = np.array(label, np.uint8)

        one_hot = np.zeros((label.shape[0], label.shape[1], len(classes)), np.float32)
        for i in range(len(classes)) :
            one_hot[np.where(label == i)] = np.eye(1,len(classes), i)

        one_hot[np.where(label == void_class_index)] = np.eye(1,len(classes), 0)

        one_hot = one_hot.transpose(2,0,1)


        if self.transform is not None :
            img = self.transform(img)

        return img, one_hot

    def __len__(self) :
        return len(self.image_names) 




