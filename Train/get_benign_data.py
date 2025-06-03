import os
import shutil
import random
import numpy as np

# 原始数据文件夹路径
IMAGE_PATH = "brain/t2_flair_selected_320"
MASK_PATH = "brain/mask_selected_320"
SINGLE_MASK_PATH="brain/single_mask"
os.makedirs(SINGLE_MASK_PATH, exist_ok=True)

# 划分后的训练集,验证集和测试集文件夹路径
TRAIN_IMAGE_PATH = "brain/train/images"
TRAIN_MASK_PATH = "brain/train/masks"
VAL_IMAGE_PATH="brain/val/images"
VAL_MASK_PATH="brain/val/masks"
TEST_IMAGE_PATH = "brain/test/images"
TEST_MASK_PATH = "brain/test/masks"

# 创建训练集和测试集文件夹
os.makedirs(TRAIN_IMAGE_PATH, exist_ok=True)
os.makedirs(TRAIN_MASK_PATH, exist_ok=True)   
os.makedirs(VAL_IMAGE_PATH, exist_ok=True)
os.makedirs(VAL_MASK_PATH, exist_ok=True)
os.makedirs(TEST_IMAGE_PATH, exist_ok=True)
os.makedirs(TEST_MASK_PATH, exist_ok=True)

# 获取所有文件列表
image_files = os.listdir(IMAGE_PATH)
image_files=sorted(image_files)
mask_files = os.listdir(MASK_PATH)
mask_files=sorted(mask_files)

#注意！！！！！！！！！！
#此时处理mask文件时，将所有2，3均替换为了0（对应仅保留了胆囊癌的分割，将结石的分割变为了背景）
for mask_file in mask_files:
    mask_npy=np.load(os.path.join(MASK_PATH,mask_file))
    print(f"Before:{np.unique(mask_npy)}")
    if len(np.unique(mask_npy)) == 1 or len(np.unique(mask_npy)) == 2:
        print("single mask or no mask!!")
    else:
        mask_npy[mask_npy!=1]=0
        print(f"After deleting other masks:{np.unique(mask_npy)}")
    np.save(os.path.join(SINGLE_MASK_PATH,mask_file),mask_npy)

single_mask_files=os.listdir(MASK_PATH)
single_mask_files=sorted(single_mask_files)

# 确保两个文件夹中的文件数量一致
assert len(image_files) == len(single_mask_files)

# 随机打乱文件顺序
combined_files = list(zip(image_files, single_mask_files))
random.shuffle(combined_files)
image_files, single_mask_files = zip(*combined_files)

# 计算训练集和测试集划分点6：2：2
total_files = len(image_files)
train_end = int(total_files * 0.6)
val_end=int(total_files*0.8)

# 划分训练集
train_image_files = image_files[:train_end]
train_mask_files = single_mask_files[:train_end]

#划分验证集
val_image_files=image_files[train_end:val_end]
val_mask_files=single_mask_files[train_end:val_end]

# 划分测试集
test_image_files = image_files[val_end:]
test_mask_files = single_mask_files[val_end:]

# 复制文件到训练集、验证集和测试集文件夹
for image_file, mask_file in zip(train_image_files, train_mask_files):
    src_image = os.path.join(IMAGE_PATH, image_file)
    src_mask = os.path.join(SINGLE_MASK_PATH, mask_file)
    dest_image = os.path.join(TRAIN_IMAGE_PATH, image_file)
    dest_mask = os.path.join(TRAIN_MASK_PATH, mask_file)
    shutil.copy(src_image, dest_image)
    shutil.copy(src_mask, dest_mask)

for image_file, mask_file in zip(val_image_files, val_mask_files):
    src_image = os.path.join(IMAGE_PATH, image_file)
    src_mask = os.path.join(SINGLE_MASK_PATH, mask_file)
    dest_image = os.path.join(VAL_IMAGE_PATH, image_file)
    dest_mask = os.path.join(VAL_MASK_PATH, mask_file)
    shutil.copy(src_image, dest_image)
    shutil.copy(src_mask, dest_mask)

for image_file, mask_file in zip(test_image_files, test_mask_files):
    src_image = os.path.join(IMAGE_PATH, image_file)
    src_mask = os.path.join(SINGLE_MASK_PATH, mask_file)
    dest_image = os.path.join(TEST_IMAGE_PATH, image_file)
    dest_mask = os.path.join(TEST_MASK_PATH, mask_file)
    shutil.copy(src_image, dest_image)
    shutil.copy(src_mask, dest_mask)

print("数据集划分完成。")
