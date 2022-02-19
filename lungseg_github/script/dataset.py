from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split
import torch
import torchvision
import numpy as np

def aug():
    data_transforms = {
        'train' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5,],  std = [0.5,])
        ]),
        'test' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5],  std = [0.5])
        ])
    }
    return data_transforms        


image_t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5],  std = [0.5])    
])

mask_t = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


class LungDataset(Dataset):
    def __init__(self, img_mask_list, img_folder, mask_folder, transform = (None,None)): # 'Initialization'
        self.img_mask_list = img_mask_list
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.img_mask_list)

    def __getitem__(self,index): # 'Generates one sample of data'      

        images_names, masks_names = self.img_mask_list[index]
        images = Image.open(self.img_folder +  images_names).convert('L')# grey
        masks = Image.open(self.mask_folder + masks_names).convert('L') # binary

        if self.transform != (None, None):
            images = self.transform[0](images)
            masks = self.transform[1](masks)
        # masks = masks.long()
        # masks = torch.squeeze(masks)
        return images, masks # chua 1 cap

# img_path ='../dataset_lungseg/images/'
# mask_path ='../dataset_lungseg/masks/'
# mask_list = os.listdir(mask_path)
# img_mask_list = [(mask_names.replace('_mask',''), mask_names) for mask_names in mask_list]
# t= LungDataset(img_mask_list, img_path, mask_path, transform = (image_t, mask_t))
# i, m = next(iter(t))
# print(i)
# print(m)


def dataloader():

    mask_path = './dataset_lungseg/masks/'
    img_path = './dataset_lungseg/images/'
    mask_list = os.listdir(mask_path)
    img_mask_list = [(mask_names.replace('_mask',''), mask_names) for mask_names in mask_list]

    train_list, test_list = train_test_split(img_mask_list, test_size = 0.2, random_state = 42) 
    train_list, val_list = train_test_split(train_list,test_size = 0.1, random_state = 42)
    # print(len(train_list), len(val_list), len(test_list)) # 506, 57, 141
    # print(train_list)

    train_set = LungDataset(train_list, img_path, mask_path, transform = (image_t, mask_t))
    val_set = LungDataset(val_list, img_path, mask_path, transform = (image_t, mask_t))
    test_set = LungDataset(test_list, img_path, mask_path, transform = (image_t, mask_t))

    loader ={
        'train' : DataLoader(
            train_set, 
            batch_size= 4,
            shuffle=True
        ),
        'val' : DataLoader(
            val_set, 
            batch_size=4,
            shuffle=True
        ),
        'test' : DataLoader(
            test_set, 
            batch_size=4,
            shuffle=True
        )
    }   
    return loader
