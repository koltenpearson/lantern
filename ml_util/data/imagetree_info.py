import os
from torch.utils import data
from PIL import Image

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)

    return images

class ImagetreeInfo(data.Dataset) :

    def __init__(self, root, transform=None) :
        self.root = root
        self.classes, self.class_to_idx = find_classes(self.root)
        self.imgs = make_dataset(self.root, self.class_to_idx)
        self.transform = transform


    def __getitem__(self, index) :
        path, label = self.imgs[index]
        img = Image.open(path)
        img = img.convert('RGB')
        img.filename = path

        if self.transform is not None :
            img = self.transform(img)

        return img, label

    def __len__(self) :
        return len(self.imgs) 




