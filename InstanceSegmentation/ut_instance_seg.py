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

from time import time

import os 



class Video (object):
    """docstring for ClassName.
        name=name of the video without extension (eg. fullstream5) """
    def __init__(self,name,path):
        self.name = name 
        self.path=path
    
    def video_to_frames(self, frames_folder):
        #Set-up
        video_path=self.path
        video_name=self.name
        ext = ".jpg"

        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        assert success == True, "Unable to open the video"  # Finish error report
        count = 0

        while success:
            frame_file = insert_string(video_name, count, ext)
            frame_path = os.path.join(frames_folder, frame_file)
            cv2.imwrite(frame_path, frame)
            success, frame = cap.read()
            count += 1
            print("Frame{0} saved successfully".format(count))
        cap.release()

    
    def frames_to_video(self,frames_folder, root_name):

        #Set-up
        video_output=self.name + "_output.avi"
        img_array = []
        # glob_pattern = frames_folder + "/*" + frames_ext

        # frames_list = glob.glob(glob_pattern)

        frames_list = [file for file in os.listdir(frames_folder) if root_name in file]


        #define how to get frames_ext
        _,frames_ext=os.path.splitext(frames_list[0])
        # Explain lambda function
        frames_list_sorted = sorted(frames_list, key=lambda x: int(x.split(root_name)[1].split(frames_ext)[0]))

      
            
        for filename in frames_list_sorted:
            img_path=os.path.join(frames_folder,filename)
            img = cv2.imread(img_path)
            img_array.append(img)

        height, width, _ = img.shape
        size = (width, height)

        out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def predictions_from_video(self, frames_folder):

        # We create the dir for the destination folder if it is not yet created
        if(not(os.path.isdir(frames_folder))):
            os.makedirs(frames_folder)

        video = cv2.VideoCapture(self.path)
        assert video.isOpened(), "Error opening the video"

        begin = time()
        count = 0
        while (video.isOpened()):

            ret, frame = video.read()
            if ret == True:
                count += 1
                frame_name = "frame{0}.jpg".format(count)
                img_file = os.path.join(frames_folder, frame_name)
                outputs = predictor(frame)
                v = Visualizer(frame[:, :, ::-1],
                            metadata=picture_metadata,
                            scale=0.8,
                            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                            )
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                img_annotated = v.get_image()[:, :, ::-1]
                cv2.imwrite(img_file, img_annotated)
                print("Frame {0} saved".format(count))

                if cv2.waitKey(0) == 27:
                    break  # esc to quit
            else:
                break

        video.release()
        end = time()
        total_elapsed = end-begin

        print("Time to process the video = {0}".format(total_elapsed))

    def get_test_frames(self, n_frames=50):

        """ Function a number of frames of one video type to be annotated
            input: video_folder = path to the folder containing all raw videos
            root_name= name of the video without extension (e.g "video_test")
            n_frames= number of frames to be extraced (50 by default)
        output: frames_test= list with filenames of the frames to be annotated
        """ 
        root_frames = [file for file in os.listdir(self.path) if self.name in file]
        rand_frames = np.random.randint(0, len(root_frames), size=n_frames)
        frames_test = [root_frames[ind] for ind in rand_frames]

        return frames_test

pass

def get_class_dicts(img_dir):
    """Function to map the json annotations into dictionaries that can be used to register
    the dataset according to the detectron2 format
    """
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in imgs_anns.items():  # cambio idx
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        record["image_id"] = idx  # cambio

        annos = v["regions"]

        annos_dict = annos[0]  # Convert the list into a dict

        objs = []
        anno = annos_dict["shape_attributes"]

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


# Registration of the dataset


def register_dataset(dataset_name, dataset_path):

    for d in ["train", "val"]:
        DatasetCatalog.register(
            dataset_name + "_" + d, lambda d=d: get_class_dicts(dataset_path+"/" + d))
        MetadataCatalog.get(dataset_name + "_" +
                            d).set(thing_classes=[dataset_name])

    return MetadataCatalog.get(dataset_name + "_train")

    # picture_metadata = MetadataCatalog.get("picture_train")


