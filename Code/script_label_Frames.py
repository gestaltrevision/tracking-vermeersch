from ut_video_seg import get_samples, update_annotations
import pandas as pd
import numpy as np
import cv2 
import os
import shutil
from general_utilities import cd,insert_string,get_filename

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--main_dir", type=str,
                    help="Base directory of annotated data")

parser.add_argument("--video_file", type=str,
                    help="Filename of clip to extract frames")

parser.add_argument("--table_file", type=str,
                    help="File of table with annotations")

parser.add_argument("-s", "--sample", action="store_true",
                    help="Sample frames from clip")

parser.add_argument("-u", "--update", action="store_true",
                    help="Update annotated table , move frames from buffer")

            
if __name__ == "__main__":
    args = parser.parse_args()
    #Define neccesary paths for frames manipulation
    table_path=os.path.join(args.main_dir,args.table_file)
    dest_dir=os.path.join(args.main_dir,"Frames")
    buffer_dir=os.path.join(args.main_dir,"Buffer")
    if (args.sample):
        video_folder= os.path.join(args.main_dir,"Clips",args.video_file)
        get_samples(video_folder,buffer_dir)
    elif(args.update):
        update_annotations(buffer_dir,table_path,dest_dir)
    else:
        print("Need to specify some valid action (Sample/Update)")
    pass

