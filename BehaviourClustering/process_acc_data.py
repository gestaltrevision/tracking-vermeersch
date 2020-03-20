import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
project_path=os.path.dirname(os.getcwd()) 
sys.path.append(os.path.join(project_path,"Code"))
from acc_ut import create_df_participant,get_participant,get_scaler,\
                univariate_outlier,get_outliers_dataset

import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--main_dir", type=str,
                    help="Base directory of annotated data")

parser.add_argument("--data_type", type=str,
                    help="Type of data that contains the table (Behaviours,accelerometer)")
parser.add_argument("--parameters_file" ,type=str,
                    help="path to json file containing parameter extraction info")
                    
parser.add_argument("-wd","--window_duration", type=int,
                    help="Duration of sliding window interval in secs")
                    
parser.add_argument("-sr","--sample_rate", type=int,
                    help="Time between samples of accelerometer in msecs")
                    
parser.add_argument("-d", "--default", action="store_true",
                    help="default values for window_duration (2 secs) and sample rate(10 msecs)")


            
if __name__ == "__main__":
    #Sample usage with default values
    #python process_acc_data.py --main_dir "C:\Users\jeuux\Desktop\Carrera\MoAI\TFM" --data_type behaviours --parameters_file "C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\AnnotatedData\Accelerometer_Data\Datasets\fc_parameters_basic.json" -d
    args = parser.parse_args()
    #Define neccesary paths for data manipulation
    if args.default==True:
        window_duration=2
        sample_rate=10

    elif args.default==False:
        window_duration=args.window_duration
        sample_rate=args.sample_rate

    acc_path=os.path.join(args.main_dir,"AnnotatedData","Accelerometer_Data")

    #Getting path from acc tables
    participant_path=os.path.join(acc_path,"Participants")
    table_paths=[os.path.join(participant_path,folder,args.data_type,file) 
                    for folder in os.listdir(participant_path)
                    for file in os.listdir(os.path.join(participant_path,folder,args.data_type))
                    if "_data_" in file]
    

    dataset_path=os.path.join(acc_path,"Datasets")

    #get fc_parameters
    with open(args.parameters_file) as f:
        fc_parameters = json.load(f)

    #open raw tables to get raw acc values and participant info
    cols=["id","Recording timestamp","AccX","AccY","AccZ"]
    table_arr=[]
    for file in tqdm(table_paths):
        df=pd.read_csv(file)
        #Extract features 
        table_arr.append(create_df_participant(df,fc_parameters,window_duration,sample_rate))

    #concat tables
    dataset_raw = pd.concat(table_arr)
    dataset_raw=dataset_raw.reset_index()

    #Remove outliers
    components=["AccX","AccY","AccZ"]
    features=["mean","standard_deviation"]
    _,outliers_list=get_outliers_dataset(dataset_raw,components,features)
    dataset_raw["outlier"]=outliers_list
    clean_dataset=dataset_raw.loc[dataset_raw["outlier"]==0].drop(columns="outlier")
  
    #Split train/val
    index_list=np.asarray(clean_dataset.index)
    idx_train, _= train_test_split(index_list,random_state=42) #split index into Train/val
    train_val=[1 if idx in idx_train else 0 for idx in index_list]
    clean_dataset["train"]=train_val #Create column with "1" if entry is a train entry, 0 otherwise
    clean_dataset["participant"]=clean_dataset.apply(lambda row: row["id"].split("_")[0],axis=1)

    #split tables
    train_df=clean_dataset[clean_dataset["train"]==1]
    train_df=train_df.reset_index(drop=True)
    test_df=clean_dataset[clean_dataset["train"]==0]
    test_df=test_df.reset_index(drop=True)
    
    #save tables
    dataset_raw.to_csv(os.path.join(dataset_path,"acc_dataset_raw.csv"),index=False)
    clean_dataset.to_csv(os.path.join(dataset_path,"acc_dataset_clean.csv"),index=False)
    test_df.to_csv(os.path.join(dataset_path,"acc_dataset_test.csv"),index=False)
    train_df.to_csv(os.path.join(dataset_path,"acc_dataset_train.csv"),index=False)

    print("Succesfully created")

