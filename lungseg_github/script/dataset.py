from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split

def aug():
    data_transforms = {
        'train' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5],  std = [0.5])
        ]),
        'test' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms        


class LungDataset(Dataset):
    def __init__(self, img_mask_list, img_folder, mask_folder, transform = None): # 'Initialization'
        self.img_mask_list = img_mask_list
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.img_mask_list)

    def __getitem__(self,index): # 'Generates one sample of data'      

        images_names, masks_names = self.img_mask_list[index]
        images = Image.open(self.img_folder +  images_names).convert('L') # grey
        masks = Image.open(self.mask_folder + masks_names) # binary

        if self.transform is not None:
            images = self.transform(images)
            masks = self.transform(masks)
        
        return images, masks # chua 1 cap


def dataloader(opt):

    mask_list = os.listdir(opt.mask_path)
    img_mask_list = [(mask_names.replace('_mask',''), mask_names) for mask_names in mask_list]

    train_list, test_list = train_test_split(img_mask_list, test_size = 0.2, random_state = 42) 
    train_list, val_list = train_test_split(train_list,test_size = 0.1, random_state = 42)
    # print(len(train_list), len(val_list), len(test_list)) # 506, 57, 141
    # print(train_list)

    train_set = LungDataset(train_list, opt.img_path, opt.mask_path, aug()['train'])
    val_set = LungDataset(val_list, opt.img_path, opt.mask_path, aug()['test'])
    test_set = LungDataset(test_list, opt.img_path, opt.mask_path, aug()['test'])

    loader ={
        'train' : DataLoader(
            train_set, 
            batch_size= opt.BATCH_SIZE,
            shuffle=True
        ),
        'val' : DataLoader(
            val_set, 
            batch_size=opt.BATCH_SIZE,
            shuffle=True
        ),
        'test' : DataLoader(
            test_set, 
            batch_size=opt.BATCH_SIZE,
            shuffle=True
        )
    }   
    return loader
