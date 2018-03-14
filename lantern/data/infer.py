import os
from torch.utils import data
from PIL import Image
from pathlib import Path



class ImageGlobDataset(data.Dataset) :
    """just loads all files with extension in directory as images"""

    def __init__(self, root, ext, transform=None) :
        self.root = Path(root)
        self.ext = ext
        self.image_paths = list(self.root.glob('**/*.'+ext))
        self.transform = transform


    def __getitem__(self, index) :
        path = self.image_paths[index]
        img = Image.open(path)
        img = img.convert('RGB')

        if self.transform is not None :
            img = self.transform(img)

        return img, str(path)

    def __len__(self) :
        return len(self.image_paths)




