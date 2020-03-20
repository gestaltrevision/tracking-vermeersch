import pandas as pd
import numpy as np
import os
import argparse
import sys
project_path=os.path.dirname(os.getcwd()) 
sys.path.append(os.path.join(project_path,"Code"))
from tqdm import tqdm
from acc_ut import preprocess_accdata,get_events,get_events_index,create_event_col


parser = argparse.ArgumentParser()

parser.add_argument("--main_dir", type=str,
                    help="Base directory of project")

parser.add_argument("--data_type", type=str,
                    help="Type of data that contains the table (Behaviours,accelerometer)")

parser.add_argument("-wd","--window_duration", type=int,
                    help="Duration of sliding window interval in secs")
                    
parser.add_argument("-sr","--sample_rate", type=int,
                    help="Time between samples of accelerometer in msecs")
                    
parser.add_argument("-d", "--default", action="store_true",
                    help="default values for window_duration (2 secs) and sample rate(10 msecs)")


if __name__ == "__main__":
    
    # Sample usage :
    #python process_raw_acc.py --main_dir "C:\Users\jeuux\Desktop\Carrera\MoAI\TFM" --data_type behaviours -d
    args = parser.parse_args()
    if args.default==True:
        window_duration=2
        sample_rate=10

    elif args.default==False:
        window_duration=args.window_duration
        sample_rate=args.sample_rate
        
    acc_path=os.path.join(args.main_dir,"AnnotatedData","Accelerometer_Data")
    participant_path=os.path.join(acc_path,"Participants")


    participant_folders=[os.path.join(participant_path,folder) for folder in os.listdir(participant_path)]

    #open raw tables to get raw acc values and participant info
    for participant_folder in tqdm(participant_folders):

        participant=os.path.basename(participant_folder)
        folder=os.path.join(participant_folder,args.data_type)

        file=os.path.join(folder,"{0}_raw_{1}.csv".format(args.data_type,participant))

        #convert raw acc data into standard form
        df=preprocess_accdata(pd.read_csv(file),window_duration,sample_rate)

        #save processed df of each participant into its correspondent folder
        dir_path=os.path.join(os.path.dirname(file),"{0}_data_{1}.csv".format(args.data_type,participant))
        df.to_csv(dir_path,index=False)

        print("Succesfully raw data  of participant : {} processed".format(participant))
    pass


