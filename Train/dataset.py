"""
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class GallBladderDataset(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.images=os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img_path=os.path.join(self.image_dir,self.images[index])
        mask_path=os.path.join(self.mask_dir,self.images[index].replace(".jpg","_mask.gif"))
        image=np.array(Image.open(img_path).convert("RGB"))
        mask=np.array(Image.open(mask_path).convert("L",dtypr=np.float32))
        mask[mask==255.0]=1.0

        if self.transform is not None:
            augmentations=self.transform(image=image,mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]

        return image,mask
"""

import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class GallBladderDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = os.listdir(self.image_dir)  # 存储所有图像文件的名字
        self.image_files = sorted(self.image_files)
        self.mask_files = os.listdir(self.mask_dir)  # 存储所有掩码文件的名字
        self.mask_files = sorted(self.mask_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])

        # 加载图像
        image = np.load(img_path).astype(np.float32)

        # 加载掩码数组（从npy文件加载）
        mask = np.load(mask_path).astype(np.float32)

        if self.transform is not None:
            image = self.transform(img=image)
        return image, mask
