import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
project_path=os.path.dirname(os.getcwd()) 
sys.path.append(os.path.join(project_path,"Code"))

from acc_ut import (filter_nulls, get_pictures_dataset,
                    process_readings_participant, segment_signal,
                    get_har_video_dataset)




parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str,
                    help="Base directory of project")

parser.add_argument("--data_type", type=str,
                    help="Type of data that contains the table (Behaviours,accelerometer)")
parser.add_argument("-bn","--base_name", type=str,
                    help="Prefix for each of the processed tables")
parser.add_argument("--dataset_name" ,type=str,
                    help="name of the current version of dataset")
                    
parser.add_argument("-ns","--n_samples", nargs='?', const=100, type=int, default=100, 
                    help="Number of samples in each sliding window")
parser.add_argument("-np","--n_points", nargs='?', const=50, type=int, default=50, 
                    help=" Data points in each sliding window after downsampling")

def read_metadata(df_path):
  #read df
  df = pd.read_csv(df_path,sep=" ",header= None)
  df.columns = ["video_path","label","frames"]
  return df


if __name__ == "__main__":
    # Sample usage :
    #python process_raw_acc.py --data_dir "C:\Users\jeuux\Desktop\Carrera\MoAI\TFM" --data_type CompleteData --base_name Final_data --dataset_name HAR_Dataset

    args=parser.parse_args()

    participant_path=os.path.join(args.data_dir,"Participants")
    participant_folders=[os.path.join(participant_path,folder) for folder in os.listdir(participant_path)]
    new_samples_number=args.n_points
    samples_per_sl=args.n_samples
    data=[]
    targets=[]
    video_folder_root = "HARClips"
    processed_participants = [participant for participant in os.listdir(participant_path) 
                                if os.path.isdir(os.path.join(participant_path,participant,video_folder_root))]
    #open raw tables to get raw acc values and participant info
    tables_list = []
    for participant_folder in tqdm(participant_folders):
        
        participant=os.path.basename(participant_folder)
        if not(participant in processed_participants):
            folder=os.path.join(participant_folder,args.data_type)
            file=os.path.join(folder,"{0}_raw_{1}.csv".format(args.data_type,participant))
            
            #convert raw acc data into standard form
            raw_df=pd.read_csv(file,decimal=',')
            df=process_readings_participant(raw_df,participant,args.n_samples)
            pictures_df = get_pictures_dataset(df,participant,500)
           
            # pictures_df = filter_nulls(pictures_df)
            HAR_video_df = get_har_video_dataset(df,participant,100)
            tables_list.append(HAR_video_df)

            sensor_comp,labels=segment_signal(df,new_samples_number,samples_per_sl)
    
            np.save(os.path.join(folder,'segments_data'), sensor_comp)
            np.save(os.path.join(folder,'targets_data'), labels)

            #save processed df of each participant into its correspondent folder
            dir_path=os.path.join(folder,"{0}_{1}.pkl".format(args.base_name,participant))
            df.to_pickle(dir_path)
            video_table_file  = os.path.join(folder,f"video_df_{participant}.csv")
            pictures_df.to_csv(video_table_file)

            save processed df of each participant into its correspondent folder
            video_table_file_har  = os.path.join(folder,f"video_df_{participant}_HAR.csv")
            HAR_video_df.to_csv(video_table_file_har)

            print("Succesfully raw data  of participant : {} processed".format(participant))
