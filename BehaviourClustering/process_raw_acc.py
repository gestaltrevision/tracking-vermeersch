import pandas as pd
import numpy as np
import os
import argparse
import sys
project_path=os.path.dirname(os.getcwd()) 
sys.path.append(os.path.join(project_path,"Code"))
from tqdm import tqdm
from acc_ut import process_readings_participant,segment_signal


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



if __name__ == "__main__":
    # Sample usage :
    #python process_raw_acc.py --data_dir "C:\Users\jeuux\Desktop\Carrera\MoAI\TFM" --data_type CompleteData --base_name Final_data --dataset_name HAR_Dataset
    #
    args=parser.parse_args()

    participant_path=os.path.join(args.data_dir,"Participants")
    participant_folders=[os.path.join(participant_path,folder) for folder in os.listdir(participant_path)]
    new_samples_number=args.n_points
    samples_per_sl=args.n_samples
    # data=[]
    # targets=[]

    #open raw tables to get raw acc values and participant info
    for participant_folder in tqdm(participant_folders):
        participant=os.path.basename(participant_folder)
        folder=os.path.join(participant_folder,args.data_type)
        file=os.path.join(folder,"{0}_raw_{1}.csv".format(args.data_type,participant))
        
        #convert raw acc data into standard form
        raw_df=pd.read_csv(file,decimal=',')
        df=process_readings_participant(raw_df,participant,args.n_samples)
        sensor_comp,labels=segment_signal(df,new_samples_number,samples_per_sl)
        # data.append(sensor_comp)
        # targets.append(labels)
        np.save(os.path.join(os.path.dirname(file),'segments_data'), sensor_comp)
        np.save(os.path.join(os.path.dirname(file),'targets_data'), labels)

        #save processed df of each participant into its correspondent folder
        dir_path=os.path.join(os.path.dirname(file),"{0}_{1}.pkl".format(args.base_name,participant))
        df.to_pickle(dir_path)

        print("Succesfully raw data  of participant : {} processed".format(participant))

    # #save full targets,readings
    # dataset_path=os.path.join(args.data_dir,"Datasets",args.dataset_name)
                               
    # if(not(os.path.isdir(dataset_path))):
    #     os.makedirs(dataset_path)

    # data=np.concatenate(data)
    # np.save(os.path.join(dataset_path,"data"),data)
    # targets=np.concatenate(targets)
    # np.save(os.path.join(dataset_path,"targets"),targets)




