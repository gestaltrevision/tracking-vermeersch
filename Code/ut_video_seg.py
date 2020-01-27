import pandas as pd
import numpy as np
import cv2 
import os
import shutil
from general_utilities import cd,insert_string,get_filename


def get_samples(video_file,frames_folder,nsamples=30):
    count=0
    cap=cv2.VideoCapture(video_file)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #Only get nsamples (ramdomly sampled)
    samples_id=list(np.random.randint(n_frames, size=nsamples))
    success,frame=cap.read()
    video_name=get_filename(video_file)
    ext=".jpg"
    while (success and (len(samples_id)>0)):
        if(count in samples_id):
            #get frame
            frame_file=insert_string(video_name,count,ext)
            frame_path=os.path.join(frames_folder,frame_file)
            cv2.imwrite(frame_path,frame)
            success,frame=cap.read()
            print("Frame{0} saved successfully".format(count))
            #remove id=count from samples_id
            samples_id.remove(count)
            
        #Udpdate counter
        count+=1
             
    cap.release()

# #Update Table
def update_annotations(buffer_dir,table_path,dest_dir):
    sub_df=create_subtable(buffer_dir)
    #open previous table
    prev_df=pd.read_pickle(table_path)
    #update
    updated_df=pd.concat([prev_df,sub_df])
    #save
    pd.to_pickle(updated_df,table_path)
    #move pictures from buffer to annotated frames folder
    move_buffer(buffer_dir,dest_dir)
    
def create_subtable(buffer_dir):
    frames_id=os.listdir(buffer_dir)
    picture_id=frames_id[0].split("_")[1]
    annotations_dict={}
    annotations_dict["frame_id"]=frames_id
    annotations_dict["class"]=[picture_id for el in range(len(frames_id))]
    sub_df=pd.DataFrame(annotations_dict)
    return sub_df
def move_buffer(source_dir,dest_dir):
    """Move all frames from buffer folder to annotated frames folder"""
    for file in os.listdir(source_dir):
        source=os.path.join(source_dir,file)
        destination=os.path.join(dest_dir,file)
        _ = shutil.move(source, destination)  
    print("Move all files with success")
