

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



def video_to_sl(video_path,video_name, frames_folder,skip=3): 

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read() ; assert success == True, "Unable to open the video"  
    frame_count = 0
    while success:
            frame_file = "{0}fr{1}.jpg".format(video_name,frame_count)
            frame_path = os.path.join(video_path, frame_file)
            if(frame_count%skip==0): #Only save one frame per skip frames
                cv2.imwrite(frame_path, frame)
                print("{0} saved successfully".format(frame_file))
            success, frame = cap.read()
            frame_count += 1
            
    cap.release()


