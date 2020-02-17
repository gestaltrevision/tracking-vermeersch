

from __future__ import division, print_function

import copy
import os
import time
from math import ceil
from statistics import mode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage import io, transform
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

import cv2
import torchvision
from general_utilities import insert_string
from torchvision import datasets, models, transforms

from time import strftime,gmtime




class SlidingWindowData(Dataset):
    """Sliding Window dataset."""

    def __init__(self, root_dir,folder_name,transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.name=folder_name

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,
                                "{0}fr{1}.jpg".format(self.name,idx*3))
 
        image = Image.open(img_name)
   
        if self.transform:
            image = self.transform(image)

        return image



def images_to_predictions(net,images,threshold,device):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    if(device.type=="cuda"):
      preds_tensor=preds_tensor.to("cpu")

    preds = np.squeeze(preds_tensor.numpy())
    percentages=[F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
    preds=[pred if (percentage >threshold) else -1 for pred,percentage in zip(preds,percentages)]

    return preds



# def get_clips_times(df,art_class):
#     class_list=df["class"].values.tolist()
#     time_list=df["time"].values.tolist()
#     try:
#         first_index=next(i for i,art in enumerate(class_list) if art == art_class)
#         last_index=next(i for i,art in enumerate(reversed(class_list)) if art == art_class)

#     except StopIteration:
#         return np.nan,np.nan
    
#     first_time=time_list[first_index]
#     last_time=(time_list[last_index*-1] if not(last_index==0) else time_list[-1-last_index])
#     return first_time,last_time



def time_to_seconds(strftime):
    minutes,seconds=strftime.split(":")
    return float(minutes)*60 + float(seconds)