import pandas as pd
import numpy as np
import cv2 
import os
import shutil
from general_utilities import cd,insert_string,get_filename
from sklearn.model_selection import train_test_split



def get_samples(video_file,frames_folder,nsamples=30):
    count=0
    cap=cv2.VideoCapture(video_file)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #Only get nsamples (ramdomly sampled)
    samples_id=list(np.random.randint(n_frames, size=nsamples))
    success,frame=cap.read()
    video_name=get_filename(video_file)
    ext=".jpg"
    while success and (len(samples_id) > 0):
        success,frame=cap.read()
        count+=1
        if(count in samples_id):
            #get frame
            frame_file=insert_string(video_name,count,ext)
            frame_path=os.path.join(frames_folder,frame_file)
            cv2.imwrite(frame_path,frame)  
            print("Frame{0} saved successfully".format(count))
            #remove id=count from samples_id
            samples_id.remove(count)
            
    print("Succesfully saved frames")
    cap.release()

# dataset_path=os.path.join(main_dir,"Classification_Frames_Dataset")
def split_dataset(dataset_path):
    picture_labels=[picture for picture in os.listdir(dataset_path) if "Y" in picture]
    for picture in picture_labels:
        Frames_folder=os.path.join(dataset_path,picture)
        Frames=[frame for frame in os.listdir(Frames_folder)] #Filenames of frames from frames_folder
        #Split
        Frames_train, Frames_val= train_test_split(Frames,random_state=42) #split filenames into Train/val
        #Move actual frames using split filenames
        for file in Frames_train: #train
            source=os.path.join(Frames_folder,file)
            dest=os.path.join(dataset_path,"Train",picture)
            shutil.move(source,dest)

        for file in Frames_val: #val
            source=os.path.join(Frames_folder,file)
            dest=os.path.join(dataset_path,"Val",picture)
            shutil.move(source,dest)
        print("Split Frames from artwork {0}".format(picture))
    print("Dataset split with success")



def statistics_dataset(dataset_path):
    """Return mean and std of dataset, this parameters will be used to normalize img in training"""
    mean_arr=[]
    std_arr=[]

    for folder in ["Train","Val"]:
        folder_path=os.path.join(dataset_path,folder)
        labels=[label for label in os.listdir(folder_path)]

        for label in labels:
            label_path=os.path.join(folder_path,label)
            imgs=os.listdir(label_path)

            for img in imgs:
                img_path=os.path.join(label_path,img)
                #read img [0,1]
                image = cv2.imread(img_path).astype(np.float32) / 255
                mean,std=cv2.meanStdDev(image)
                mean_arr.append(mean)
                std_arr.append(std)
            
    dataset_mean =np.mean(np.array(mean_arr, dtype=np.float32),axis=0)
    dataset_std =np.mean(np.array(std_arr, dtype=np.float32),axis=0)

    return dataset_mean,dataset_std

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
