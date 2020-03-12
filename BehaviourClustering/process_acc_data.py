import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from acc_ut import get_psd_values,get_fft_values,get_autocorr_values,get_features
from acc_ut import create_df_participant,get_participant,get_scaler
from pickle import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--main_dir", type=str,
                    help="Base directory of annotated data")
                    
parser.add_argument("--scaler_name", type=str,
                    help="Basename of file to save scaler data")


parser.add_argument("-wd","--window_duration", type=int,
                    help="Duration of sliding window interval in secs")

parser.add_argument("--starts", nargs='+', type=float,
                    help="vector of starts to create sliding windows")

parser.add_argument("-d", "--default", action="store_true",
                    help="default values for window_duration (2 secs) and starts (0s,0.5s)")


            
if __name__ == "__main__":
    #Sample usage with default values
    #python process_acc_data.py --main_dir "C:\Users\jeuux\Desktop\Carrera\MoAI\TFM" --scaler_name "scaler" -d
    args = parser.parse_args()
    #Define neccesary paths for data manipulation
    if args.default==True:
        window_duration=2
        starts=[0,0.5]
    elif args.default==False:
        window_duration=args.window_duration
        starts=args.starts

    acc_path=os.path.join(args.main_dir,"AnnotatedData","Accelerometer_Data")
    #Getting path from acc tables
    participant_path=os.path.join(acc_path,"Participants")
    table_paths=[os.path.join(participant_path,folder,file) 
                    for folder in os.listdir(participant_path)
                    for file in os.listdir(os.path.join(participant_path,folder))
                    if "acc_data_" in file]
    
    acc_table_arr=[];participant_arr=[]
    #open raw tables to get raw acc values and participant info
    for file in table_paths:
        #get participant
        participant_arr.append(get_participant(file,"acc_data_"))
        acc_table_arr.append(pd.read_csv(file)) ##change to already processed tables

    acc_table = pd.concat(acc_table_arr)

    #Get means and standard deviations from full accelerometer table
    cols=["AccX","AccY","AccZ"]
    scaler=get_scaler(acc_table,cols) 
    # save the scaler
    dataset_path=os.path.join(acc_path,"Datasets")
    scaler_path=os.path.join(dataset_path,"{}.pkl".format(args.scaler_name))
    dump(scaler, open(scaler_path, 'wb'))

    table_arr=[]
    for participant,df in tqdm(zip(participant_arr, acc_table_arr)):
        #Normalize
        x = df[cols].values
        x_scaled = scaler.fit_transform(x)
        df = pd.concat((df["time"],pd.DataFrame(x_scaled,columns=cols)),axis=1)
        #Extract features
        table_arr.append(create_df_participant(df[cols],participant,window_duration,starts))

    #concat tables
    complete_df = pd.concat(table_arr)
    
    #Split train/val
    index_list=np.asarray(complete_df.index)
    idx_train, _= train_test_split(index_list,random_state=42) #split index into Train/val
    train_val=[1 if idx in idx_train else 0 for idx in index_list]
    complete_df["train"]=train_val #Create column with "1" if entry is a train entry, 0 otherwise
    complete_df["target"]= np.nan
    
    #split tables
    train_df=complete_df[complete_df["train"]==1]
    train_df=train_df.reset_index(drop=True)
    test_df=complete_df[complete_df["train"]==0]
    test_df=test_df.reset_index(drop=True)
    
    #save tables
    test_df.to_csv(os.path.join(dataset_path,"{0}_test.csv".format("acc_dataset")),index=False)
    train_df.to_csv(os.path.join(dataset_path,"{0}_train.csv".format("acc_dataset")),index=False)

    print("Succesfully created")




