import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
project_path=os.path.dirname(os.getcwd()) 
sys.path.append(os.path.join(project_path,"Code"))
from acc_ut import create_df_participant,univariate_outlier,get_outliers_dataset


import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import argparse


def save_summary_dataset(args,clean_dataset,dataset_path):
    #save name, window_duration, sample rate
    summary_dict=vars(args)
    #save fc parameters
    # summary_dict["fc_parameters"]=fc_parameters

    #basic statistics dataset : n_total samples; rel label/Unlabeled
                                            #    ;rel Movement/Still training set
    stats_dict={}       
    stats_dict["total_samples"] =len(clean_dataset)
    supervised_dataset=clean_dataset[clean_dataset.target!=-1]
    stats_dict["annotated_ratio"]=   len(supervised_dataset) /stats_dict["total_samples"]
    stats_dict["target_ratio"]=   supervised_dataset.target.value_counts(normalize=True)
    summary_dict["stats"]=stats_dict
    
    summary_file=os.path.join(dataset_path,"summary_json")
    #summary to json, save in current folder
    with open(summary_file,'w') as f:
        json.dump(summary_dict,f,indent=4)
    print("Succesfully saved summary of dataset")

def get_fc_parameters(dataset_path):
    """input: folder_path for current dataset
        output: dict with parameter info for feature extraction"""
    fc_parameters_path=next(file for file in os.listdir(dataset_path) if "parameters" in file)
    with open(fc_parameters_path) as f:
        fc_parameters = json.load(f)
    return fc_parameters


def filter_features_dataset(dataset):
    feat_cols=[col for col in dataset.columns if "Acc" in col]
    extracted_features=dataset[feat_cols]
    y=dataset.target
    features_filtered = select_features(extracted_features, y)
    return pd.concat((features_filtered,y),axis=1), features_filtered.columns

parser = argparse.ArgumentParser()

parser.add_argument("--main_dir", type=str,
                    help="Base directory of annotated data")

parser.add_argument("--data_type", type=str,
                    help="Type of data that contains the table (Behaviours,accelerometer)")

parser.add_argument("--dataset_name" ,type=str,
                    help="name of the current version of dataset")
                    
parser.add_argument("-sr","--sample_rate", nargs='?', const=10, type=int, default=10, 
                    help="Time between samples of accelerometer in msecs")

parser.add_argument('-wd','--window_duration', nargs='?', const=2, type=int, default=2,
                    help="Duration of sliding window interval in secs")

                    


from tsfresh import select_features

if __name__ == "__main__":
    #Sample usage with default values
    #python process_acc_data.py --main_dir "C:\Users\jeuux\Desktop\Carrera\MoAI\TFM" --data_type Behaviours -wd 3 --dataset_name Event_Features  
    args = parser.parse_args()
    #Define neccesary paths for data manipulation
    acc_path=os.path.join(args.main_dir,"AnnotatedData","Accelerometer_Data")

    #Getting path from acc tables
    participant_path=os.path.join(acc_path,"Participants")
    table_paths=[os.path.join(participant_path,folder,args.data_type,file) 
                    for folder in os.listdir(participant_path)
                    for file in os.listdir(os.path.join(participant_path,folder,args.data_type))
                    if "_data_" in file]
    
    dataset_path=os.path.join(acc_path,"Datasets",args.dataset_name)
    
    #get fc_parameters
    # fc_parameters=get_fc_parameters(dataset_path)
    
    #open raw tables to get raw acc values and participant info
    cols=["id","Recording timestamp","AccX","AccY","AccZ"]
    table_arr=[]
    for file in tqdm(table_paths):
        df=pd.read_csv(file)
        #Extract features 
        table_arr.append(create_df_participant(df))

    #concat tables
    dataset_raw = pd.concat(table_arr)
    dataset_raw=dataset_raw.reset_index()

    #Remove outliers ((CHANGE HOW WE REMOVE OUTLIERS?))
    components=["AccX","AccY","AccZ"]
    features=["mean","standard_deviation"]
    _,outliers_list=get_outliers_dataset(dataset_raw,components,features)
    dataset_raw["outlier"]=outliers_list
    clean_dataset=dataset_raw.loc[dataset_raw["outlier"]==0].drop(columns="outlier")
    
    data_supervised=clean_dataset[clean_dataset.target!=-1].reset_index(drop=True)
    
    #Split train/val
    idx_train, idx_val= train_test_split(data_supervised.index,random_state=42) #split index into Train/val
    data_supervised["train"]=np.nan
    data_supervised.loc[idx_train,"train"]=1
    data_supervised.loc[idx_val,"train"]=0
    #create participant column
    data_supervised["participant"]=data_supervised.apply(lambda row: row["id"].split("_")[0],axis=1)
    dataset_raw.to_csv(os.path.join(dataset_path,"acc_dataset_raw.csv"),index=False)
    clean_dataset.to_csv(os.path.join(dataset_path,"acc_dataset_semisupervised.csv"),index=False)

    #split tables
    train_df=data_supervised[data_supervised["train"]==1]
    train_df=train_df.reset_index(drop=True)
    test_df=data_supervised[data_supervised["train"]==0]
    test_df=test_df.reset_index(drop=True)
    
   
    ##filer features 
    train_df,filtered_cols=filter_features_dataset(train_df)
    test_dfx=test_df.copy()
    test_df=test_df[filtered_cols]
    target_cols=["target","event_cat"]
    test_df[target_cols]=test_dfx[target_cols]
    

    #save tables
    data_supervised.to_csv(os.path.join(dataset_path,"acc_dataset_supervised.csv"),index=False)
    test_df.to_csv(os.path.join(dataset_path,"acc_dataset_test.csv"),index=False)
    train_df.to_csv(os.path.join(dataset_path,"acc_dataset_train.csv"),index=False)

    #save summary
    save_summary_dataset(args,clean_dataset,dataset_path)
    print("Succesfully created")

