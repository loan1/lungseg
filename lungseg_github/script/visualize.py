from dataset import *
# from ..main import get_opt
import matplotlib.pyplot as plt
import numpy as np
import argparse
from transform import *

def imshow(input):
    """ Imshow for Tensor"""
    print(input.shape)
    # input = input.numpy(dtype=np.float32).transpose((2, 3, 1, 0))
    input = input.numpy().transpose((1, 2, 0))
    print(input.shape)
    input = input*0.5 + 0.5
    input = np.squeeze(input)
    # print(input)
    input = np.clip(input, 0, 1)
    plt.imshow(input, cmap = 'gray')
    plt.show()

mask_path = '../dataset_lungseg/masks/'
img_path = '../dataset_lungseg/images/'
mask_list = os.listdir(mask_path)
img_mask_list = [(mask_names.replace('_mask',''), mask_names) for mask_names in mask_list]

train_set = LungDataset(img_mask_list, img_path, mask_path, transform = (image_t, mask_t))
# train_set = dataloader()['train']
image, mask = next(iter(train_set))
# print(image.shape)
# print(torch.max(image))
# print(image.min())
# print(mask.max())
# print(mask.min())

plt.subplot(1,2,1)
imshow(image)

plt.subplot(1,2,2)
imshow(mask)

# print(mask)

def plot_acc_loss (loss, val_loss, acc, val_acc):
    """ plot training and validation loss and accuracy """
    plt.figure (figsize = (12, 4))
    plt.subplot (1, 2, 1)
    plt.plot (range (len (loss)), loss, 'b-', label = 'Training')
    plt.plot (range (len (loss)), val_loss, 'bo-', label = 'Validation')
    plt.xlabel ('Epochs')
    plt.ylabel ('Loss')
    plt.title ('Loss')
    plt.legend ()

    plt.subplot (1, 2, 2)
    plt.plot (range (len (acc)), acc, 'b-', label = 'Training')
    plt.plot (range (len (acc)), val_acc, 'bo-', label = 'Validation')
    plt.xlabel ('Epochs')
    plt.ylabel ('accuracy')
    plt.title ('Accuracy')
    plt.legend ()

    plt.show ()

