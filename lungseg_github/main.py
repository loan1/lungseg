from script.model import *
from script.dataset import *
from script.trainmodel import *

#importing the libraries
import pandas as pd
import os
import numpy as np
import time
import copy
import warnings
from tqdm import tqdm

#for reading and displaying images
import matplotlib.pyplot as plt

import cv2
from PIL import Image

#Pytorch libraries and modules
import torch

from torchsummary import summary

from torch.nn import Linear, BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader,Dataset

#torchvision for pre-trained models and augmentation
import torchvision

from torchvision import models

#for evaluating model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", default = './model/UNet.pt',type=str)
    parser.add_argument('--img_path', default='./dataset_lungseg/images/', type=str)
    parser.add_argument('--mask_path', default='./dataset_lungseg/masks/', type= str)

    parser.add_argument('--BATCH_SIZE', default=4, type=int)
    parser.add_argument('--num_epochs', default= 100, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)

    opt = parser.parse_args()
    return opt

# img, masks = next(iter(train_loader))
# print(img, masks)
def imshow(input):
    """ Imshow for Tensor"""
    # print(input.shape)
    input = input.numpy().transpose((3,2,0,1))
    # print(input.shape)
    input = input*[0.5] + [0.5]
    input = np.squeeze(input)
    # print(input)
    input = np.clip(input, 0, 1)
    plt.imshow(input, cmap = 'gray')
    plt.show()


if __name__ == '__main__':

    opt = get_opt()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = BCEWithLogitsLoss()
    model = UNet_ResNet.to(device)
    optimizer = Adam(model.parameters(), opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=14, gamma=0.1)
    training_loop(model, optimizer, criterion, scheduler, device, opt.num_epochs, dataloader, opt.CHECKPOINT_PATH)

    image, mask = next(iter(dataloader()['train']))

   

