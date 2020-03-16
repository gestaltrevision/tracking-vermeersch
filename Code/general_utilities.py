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




def get_subject(json_file):
    with open(json_file) as f:
        participant_info = json.load(f)
    return participant_info["pa_info"]["Name"]
    
def find_participant(participant_target,main_dir):
    
    data_path=os.path.join(main_dir,"Data")
    participant_folders=[os.path.join(data_path,folder) for folder in os.listdir(data_path) if folder!=".vscode"]
    
    return next(participant_path for participant_path in participant_folders
                    if get_subject(os.path.join(participant_path,"participant.json"))==participant_target)

class AttrDisplay:
    """
    Provides an inheritable display overload method that shows
    instances with their class names and a name=value pair for
    each attribute stored on the instance itself (but not attrs
    inherited from its classes). Can be mixed into any class,
    and will work on any instance.
    """
    def gatherAttrs(self):
        attrs = []
        for key in sorted(self.__dict__):
            attrs.append('%s=%s' % (key, getattr(self, key)))
        return ', '.join(attrs)

    def __repr__(self):
        return '[%s: %s]' % (self.__class__.__name__, self.gatherAttrs())