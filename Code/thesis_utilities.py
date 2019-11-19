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
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

import time

def get_filename(file_path):
    head,tail=ntpath.split(file_path)
    return tail or ntpath.basename(head)

def insert_string(filename,el,extension):
    name,_=os.path.splitext(filename)
    return name + str(el) + extension

def get_filename_annotations(annotations_file):
    filenames=[]
    with open(annotations_file) as json_file:
        data = json.load(json_file)
        for file_id in data.keys():
            filenames.append(data[file_id]["filename"])
    return filenames

def video_to_frames(video_file,frames_folder):
    count=0
    cap=cv2.VideoCapture(video_file)
    success,frame=cap.read()
    assert success==True , "Unable to open the video"   #Finish error report

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


def get_test_frames(video_folder,root_name,n_frames=50):
    """ Function a number of frames of one video type to be annotated
            input: video_folder = path to the folder containing all raw videos
            root_name= name of the video without extension (e.g "video_test")
            n_frames= number of frames to be extraced (50 by default)
        output: frames_test= list with filenames of the frames to be annotated

    """
    root_frames=[file for file in os.listdir(video_folder) if root_name in file] 
    rand_frames=np.random.randint(0,len(root_frames),size=n_frames)
    frames_test=[root_frames[ind] for ind in rand_frames]

    return frames_test


def move_files(filenames,src_folder,dest_folder):
    for file_name in filenames:
        dest_path= os.path.join(dest_folder,file_name)
        src_path=os.path.join(src_folder,file_name)
        copyfile(src_path,dest_path)
    print("Process Finished with exit")






def frames_to_video(frames_folder,video_folder,base_name,frames_ext):
    
    img_array = []
    glob_pattern= frames_folder +"/*"+ frames_ext
    frames_list=glob.glob(glob_pattern)
    #Explain lambda function
    frames_list_sorted=sorted(frames_list,key= lambda x : int(x.split(base_name)[1].split(frames_ext)[0])) 

    for filename in frames_list_sorted:
        img = cv2.imread(filename)
        img_array.append(img)
    
    height, width, _ = img.shape
    size = (width,height)
    
    out = cv2.VideoWriter(video_folder,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()





def get_class_dicts(img_dir):

    """Function to map the json annotations into dictionaries that can be used to register
    the dataset according to the detectron2 format
    """
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in imgs_anns.items():   #cambio idx
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        record["image_id"] = idx    #cambio
 
        annos = v["regions"]

        annos_dict=annos[0]  #Convert the list into a dict

    
        objs = []
        anno=annos_dict["shape_attributes"]

  
        px = anno["all_points_x"]
        py = anno["all_points_y"]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = list(itertools.chain.from_iterable(poly))

        obj = {
            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": 0,
            "iscrowd": 0
        }
        objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


#Registration of the dataset


def register_dataset(dataset_name):

    for d in ["train", "val"]:
        DatasetCatalog.register(dataset_name+ "_" + d, lambda d=d: get_class_dicts(dataset_name+"/" + d))
        MetadataCatalog.get(dataset_name+ "_" + d).set(thing_classes=[dataset_name])

    return  MetadataCatalog.get("dataset_name_train")

    # picture_metadata = MetadataCatalog.get("picture_train")


def predictions_from_video(video_file,frames_folder):

    #We create the dir for the destination folder if it is not yet created
    if(not(os.path.isdir(frames_folder))):
        os.makedirs(frames_folder)

    video=cv2.VideoCapture(video_file)
    assert video.isOpened(), "Error opening the video"
    
    begin=time()
    count=0
    while (video.isOpened()):

        ret,frame=video.read()
        if ret==True:
            count+=1
            frame_name="frame{0}.jpg".format(count)
            img_file=os.path.join(imgs_folder,frame_name)
            outputs=predictor(frame)
            v = Visualizer(frame[:, :, ::-1],
                        metadata=picture_metadata, 
                        scale=0.8, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            img_annotated=v.get_image()[:,:,::-1]
            cv2.imwrite(img_file,img_annotated)
            print("Frame {0} saved".format(count))
            
            if cv2.waitKey(0) == 27:
                            break  # esc to quit
            else:
                break   


    video.release()
    end=time()
    total_elapsed=end-begin

    print("Time to process the video = {0}".format(total_elapsed))
