import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import json
from PIL import Image
import random
import os
from skimage.transform import resize, rotate


IMG_EXT = ['jpg', 'jpeg', 'png']

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def get_random_parameters(crop_size=224, theta_range=30, size_delta=80):
    # select parameters for rotation
    theta = np.random.randint(-theta_range, theta_range+1)
    # select parameters for scale and crop
    scale_size = np.random.randint(crop_size+1, crop_size+size_delta+1)
    crop_x, crop_y = np.random.randint(0, scale_size-crop_size, 2)
    box = (crop_x, crop_y, crop_x+crop_size, crop_y+crop_size)

    return theta, box, scale_size

def get_flip_flags(hflip_prob=0.5, vflip_prob=0.25):
    hflip = random.random() < hflip_prob
    vflip = random.random() < vflip_prob    
    return hflip, vflip

def resize_crop_image(img, crop_size, resize_ratio=0.15):
    scale = crop_size + int(crop_size * resize_ratio)
    crop_x = crop_y = int(0.5 * (scale - crop_size))
    box = (crop_x, crop_y, crop_x+crop_size, crop_y+crop_size)
    img = img.resize((scale, scale), Image.BILINEAR).crop(box)
    return img

def resize_crop_points(points, crop_size, true_size, resize_ratio=0.15):
    width, height = true_size
    scale = crop_size + int(crop_size * resize_ratio)
    dx = dy = int(0.5 * (scale - crop_size))
    scale_mat  = np.array([
        [scale / width,  0],
        [0, scale / height]])

    transformed_points = {}
    for part, pts in points.items():
        pts = np.array(pts).T
        pts = pts @ scale_mat.T - [dx, dy]
        transformed_points[part] = pts
    return transformed_points

def resize_crop_masks(masks, crop_size, resize_ratio=0.15):

    ms1 = Image.fromarray(masks[0:3].transpose(1,2,0))
    ms2 = Image.fromarray(masks[3:6].transpose(1,2,0))
    ms3 = Image.fromarray(masks[6:9].transpose(1,2,0))

    ms1 = resize_crop_image(ms1, crop_size)
    ms2 = resize_crop_image(ms2, crop_size)
    ms3 = resize_crop_image(ms3, crop_size)

    masks = np.zeros((9, crop_size, crop_size))
    masks[0:3] = np.array(ms1).transpose(2,0,1)
    masks[3:6] = np.array(ms2).transpose(2,0,1)
    masks[6:9] = np.array(ms3).transpose(2,0,1)

    return masks

def augment_image(img, theta, box, scale_size, crop_size=224, 
                  hflip=False, vflip=False):
    #########################################################
    # Apply transforms to the image
    # - resize randomly within a range
    # - rotate randomly within a range
    # - crop to desired size at random (valid) position
    # - randomly choose to horizontally and/or vertically flip
    #########################################################
    img = img.resize((scale_size,scale_size), Image.BILINEAR
    ).rotate(theta, Image.BILINEAR).crop(box)

    if hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return img

def augment_points(points, theta, box, crop_size, scale_size, true_size, hflip, vflip, radius=8.0):
    #########################################################
    # Apply transforms to the points
    # - scale to match the image resize
    # - rotate to match the image rotation
    # - translate to match the image crop
    # - flip coordinates around center to match image if necessary
    # - add small jitter to the point coordinates
    #########################################################
    # calculate parameters for transforming the points
    width, height = true_size
    cos_t = np.cos(theta * np.pi / 180)
    sin_t = np.sin(theta * np.pi / 180)
    dx = box[0]
    dy = box[1]
    dis = scale_size / 2
    scale_mat  = np.array([
        [scale_size / width,  0],
        [0, scale_size / height]])
    rotation_mat = np.array([
        [ cos_t, sin_t], 
        [-sin_t, cos_t]])

    transformed_points = {}
    for part, pts in points.items():
        pts = np.array(pts).T
        # apply the transform to the points
        pts = (pts @ scale_mat.T - [dis,dis]) @ rotation_mat.T + [dis-dx,dis-dy]
        # jitter each point independently within a small radius
        r = np.sqrt(radius * np.random.rand(4))
        phi = 2*np.pi*np.random.rand(4)
        x_jitter = r * np.cos(phi)
        y_jitter = r * np.sin(phi)
        pts[:,0] += x_jitter
        pts[:,1] += y_jitter
        # account for image flipping
        if hflip:
            pts[:,0] = crop_size - pts[:,0]
            if   part[0] == 'l': part = part.replace('l','r')
            elif part[0] == 'r': part = part.replace('r','l')
        if vflip:
            pts[:,1] = crop_size - pts[:,1]
            if   part[0] == 'l': part = part.replace('l','r')
            elif part[0] == 'r': part = part.replace('r','l')
        transformed_points[part] = pts

    return transformed_points

def augment_masks(masks, theta, box, scale_size, crop_size, hflip, vflip):
    ms1 = Image.fromarray(masks[0:3].transpose(1,2,0))
    ms2 = Image.fromarray(masks[3:6].transpose(1,2,0))
    ms3 = Image.fromarray(masks[6:9].transpose(1,2,0))
    
    ms1 = augment_image(ms1, theta, box, scale_size, crop_size, hflip, vflip)
    ms2 = augment_image(ms2, theta, box, scale_size, crop_size, hflip, vflip)
    ms3 = augment_image(ms3, theta, box, scale_size, crop_size, hflip, vflip)

    width, height = box[2]-box[0], box[3]-box[1]
    masks = np.zeros((9, height, width))
    masks[0:3] = np.array(ms1).transpose(2,0,1)
    masks[3:6] = np.array(ms2).transpose(2,0,1)
    masks[6:9] = np.array(ms3).transpose(2,0,1)

    if hflip ^ vflip:
        # swap the part order when flipping. Not that if we flip vertically and
        # horizontally, the part identities don't change
        swap = [0,5,6,7,8,1,2,3,4]
        masks = masks[swap]
    
    masks = np.where(masks > 0.5, 1.0, 0.0)
    return masks

def augment_masks_2(masks, theta, box, scale_size, hflip, vflip):
    # I thought this would be faster, because it does away with the conversions
    # back and forth between pil images. Not so. It's about 20 ms slower on
    # average. It would seem that skimage is slower than PIL
    x1,y1,x2,y2 = box
    scale = (scale_size, scale_size)
    masks = resize(masks.transpose(1,2,0), scale, order=1, mode='reflect')
    masks = rotate(masks, theta, order=1)
    masks = masks[y1:y2, x1:x2, :]
    masks = masks.transpose(2,0,1)

    if hflip:
        masks = masks[:,:,::-1]
    if vflip:
        masks = masks[:,::-1,:]

    if hflip ^ vflip:
        # swap the part order when flipping. Not that if we flip vertically and
        # horizontally, the part identities don't change
        swap = [0,5,6,7,8,1,2,3,4]
        masks = masks[swap]
    
    masks = np.where(masks > 0.5, 1.0, 0.0)
    return masks

class ButterflyDataset(Dataset):

    def __init__(self, root, final_size, augment=True, finetune=False):

        self.root = Path(root)
        self.final_size = final_size
        self.augment = augment
        self.finetune = finetune

        self.imagepaths = []
        self.classes = []
        for class_ in self.root.iterdir():
            if class_.is_dir():
                self.classes.append(class_.parts[-1])
                for img in class_.iterdir():
                    if img.parts[-1].split('.')[-1] in IMG_EXT:
                        self.imagepaths.append(img)
        print('Num images: ', len(self.imagepaths))
        self.classes.sort()
        self.class_to_idx = {c: i for i,c in enumerate(self.classes)}

        self.totensor = transforms.ToTensor()
        # imagenet normalization for finetuning
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        path = self.imagepaths[index]
        target = self.class_to_idx[path.parts[-2]]

        img = Image.open(path).convert('RGB')
        width, height = img.size

        if self.augment:
            hflip, vflip = get_flip_flags()
            theta, box, scale_size = get_random_parameters()
            img = augment_image(img, theta, box, scale_size, hflip, vflip)
        else:
            img = resize_crop_image(img, self.final_size)

        img = self.totensor(img)

        if self.finetune:
            img = self.normalize(img)

        return img, target

    def __len__(self):
        return len(self.imagepaths)


class PoseNormDataset(ButterflyDataset):

    def __init__(self, root, points_file, final_size, augment=True):

        super(PoseNormDataset, self).__init__(root, final_size, augment)
        self.points = json.load(open(points_file))

    def __getitem__(self, index):
        path = self.imagepaths[index]
        point_key = path.parts[-1].split('.')[0]

        img = Image.open(path).convert('RGB')
        width, height = img.size

        points = self.points[point_key]

        if self.augment:
            crop_size = self.final_size
            hflip, vflip = get_flip_flags()
            theta, box, scale_size = get_random_parameters()
            # augment image
            img = augment_image(img, theta, box, scale_size, crop_size, hflip, vflip)
            # augment points
            points = augment_points(points, theta, box, crop_size, scale_size, (width,height), hflip, vflip)
        else:
            img = resize_crop_image(img, self.final_size)
            points = resize_crop_points(points, self.final_size, (width,height))

        img = self.totensor(img)
        return img, points



class SegmentationDataset(ButterflyDataset):

    def __init__(self, root, final_size, augment=True, finetune=True):
        super(SegmentationDataset, self).__init__(
                root, final_size, augment, finetune)

    def __getitem__(self, index):
        
        path = self.imagepaths[index]
        target = self.class_to_idx[path.parts[-2]]

        img = Image.open(path).convert('RGB')
        width, height = img.size

        mask_path = os.path.splitext(path)[0] + '.npy'
        if os.path.exists(mask_path):
            #masks = np.load(mask_path)
            masks = np.load(mask_path).astype('uint8') 
        else:
            #masks = np.zeros((9, img.size[1], img.size[0]), 'float32')
            masks = np.zeros((9, img.size[1], img.size[0]), 'uint8')

        if self.augment:
            hflip, vflip = get_flip_flags()
            theta, box, scale_size = get_random_parameters()
            img = augment_image(img, theta, box, scale_size, self.final_size, hflip, vflip)
            masks = augment_masks(masks, theta, box, scale_size, self.final_size, hflip, vflip)
        else:
            img = resize_crop_image(img, self.final_size)
            masks = resize_crop_masks(masks, self.final_size)

        img = self.totensor(img)
        masks = torch.from_numpy(masks.astype('float32'))

        if self.finetune:
            img = self.normalize(img)

        return img, target, masks, str(path)


class PoseWithSegmentationDataset(ButterflyDataset):

    def __init__(self, root,seg_dir, points_file, final_size, augment=True):

        super(PoseWithSegmentationDataset, self).__init__(root, final_size, augment)
        self.points = json.load(open(points_file))
        self.seg_dir = Path(seg_dir)

    def __getitem__(self, index):
        
        path = self.imagepaths[index]
        target = self.class_to_idx[path.parts[-2]]

        img = Image.open(path).convert('RGB')
        width, height = img.size

        point_key = path.parts[-1].split('.')[0]
        points = self.points[point_key]

        mask_path = self.seg_dir/(point_key+'.npy')
        masks = np.load(mask_path).astype('uint8') 

        crop_size = self.final_size
        if self.augment:
            hflip, vflip = get_flip_flags()
            theta, box, scale_size = get_random_parameters()
            img = augment_image(img, theta, box, scale_size, crop_size, hflip, vflip)
            masks = augment_masks(masks, theta, box, scale_size, crop_size, hflip, vflip)
            points = augment_points(points, theta, box, crop_size, scale_size, (width,height), hflip, vflip)
        else:
            img = resize_crop_image(img, crop_size)
            masks = resize_crop_masks(masks, crop_size)
            points = resize_crop_points(points, crop_size, (width,height))

        img = self.totensor(img)
        masks = torch.from_numpy(masks.astype('float32'))

        if self.finetune:
            img = self.normalize(img)

        return img, points, target, masks, str(path)


