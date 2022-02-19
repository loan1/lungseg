import matplotlib.pyplot as plt
from dataset import *

# from main import *
import numpy as np
import argparse

def imshow(input, title = None):
    """ Imshow for Tensor"""
    input = input.numpy().transpose((1, 2, 0))
    input = input*[0.5] + [0.5]
    print(input)
    input = np.clip(input, 0, 1)
    plt.imshow(input)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", default = '../model/UNet.pt',type=str)
    parser.add_argument('--img_path', default='../dataset_lungseg/images/', type=str)
    parser.add_argument('--mask_path', default='../dataset_lungseg/masks/', type= str)

    parser.add_argument('--BATCH_SIZE', default=4, type=int)
    parser.add_argument('--num_epochs', default= 100, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)

    opt = parser.parse_args()
    return opt

opt = get_opt()
image, mask = next(iter(dataloader(opt)['train']))

fig = plt.figure(figsize = (20,10))

for idx in np.arange(opt.BATCH_SIZE):
    ax = fig.add_subplot()
    imshow(image[idx], 'Image')
    imshow(mask[idx], 'Mask')