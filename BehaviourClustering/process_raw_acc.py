import pandas as pd
import numpy as np
import os
import argparse
import sys
project_path=os.path.dirname(os.getcwd()) 
sys.path.append(os.path.join(project_path,"Code"))
from tqdm import tqdm
from acc_ut import process_readings_participant,get_events,get_events_index,create_event_col


parser = argparse.ArgumentParser()

parser.add_argument("--main_dir", type=str,
                    help="Base directory of project")

parser.add_argument("--data_type", type=str,
                    help="Type of data that contains the table (Behaviours,accelerometer)")
parser.add_argument("-bn","--base_name", type=str,
                    help="Prefix for each of the processed tables")

parser.add_argument("-ns","--n_samples", nargs='?', const=200, type=int, default=200, 
                    help="Number of accelerometer samples in each sliding window")

parser.add_argument('-fc','--cut_freq', nargs='?', const=10, type=int, default=10,
                    help="Cut-off frequency of filter")
if __name__ == "__main__":
    # Sample usage :
    #python process_raw_acc.py --main_dir "C:\Users\jeuux\Desktop\Carrera\MoAI\TFM" --data_type Acc_Gyros 
    args=parser.parse_args()
    acc_path=os.path.join(args.main_dir,"AnnotatedData","Accelerometer_Data")
    participant_path=os.path.join(acc_path,"Participants")
    participant_folders=[os.path.join(participant_path,folder) for folder in os.listdir(participant_path)]

    #open raw tables to get raw acc values and participant info
    for participant_folder in tqdm(participant_folders):
        participant=os.path.basename(participant_folder)
        folder=os.path.join(participant_folder,args.data_type)
        file=os.path.join(folder,"{0}_raw_{1}.csv".format(args.data_type,participant))

        #convert raw acc data into standard form
        df=process_readings_participant(pd.read_csv(file),args.cut_freq,args.n_samples)

        #save processed df of each participant into its correspondent folder
        dir_path=os.path.join(os.path.dirname(file),"{0}_{1}.csv".format(args.base_name,participant))
        df.to_csv(dir_path,index=False)

        print("Succesfully raw data  of participant : {} processed".format(participant))
    pass


