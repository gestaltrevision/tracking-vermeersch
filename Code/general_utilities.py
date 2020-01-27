import glob
import itertools
import json
import ntpath
import os
from shutil import copyfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import cv2
from time import time

import os 

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def get_filename(file_path):
    head, tail = ntpath.split(file_path)
    return tail or ntpath.basename(head)


def insert_string(filename, el, extension):
    name, _ = os.path.splitext(filename)
    return name + str(el) + extension


def get_filename_annotations(annotations_file):
    filenames = []
    with open(annotations_file) as json_file:
        data = json.load(json_file)
        for file_id in data.keys():
            filenames.append(data[file_id]["filename"])
    return filenames

def move_files(filenames, src_folder, dest_folder):
    for file_name in filenames:
        dest_path = os.path.join(dest_folder, file_name)
        src_path = os.path.join(src_folder, file_name)
        copyfile(src_path, dest_path)
    print("Process Finished with exit")

def get_frames(video_file,frames_folder):
    count=0
    cap=cv2.VideoCapture(video_file)
    success,frame=cap.read()
    video_name=get_filename(video_file)
    ext=".jpg"
    while success:
        frame_file=insert_string(video_name,count,ext)
        frame_path=os.path.join(frames_folder,frame_file)
        cv2.imwrite(frame_path,frame)
        success,frame=cap.read()
        count+=1
        print("Frame{0} saved successfully".format(count))
    cap.release()




