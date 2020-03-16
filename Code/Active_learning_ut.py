##Active learning
from IPython.display import Video,clear_output,display
import os
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from general_utilities import find_participant,AttrDisplay
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip,ffmpeg_resize

        
class VideoClip(object):
    """"""
    def __init__(self,table,query_idx,main_dir,window_duration=2):
        self.table= table
        self.main_dir=main_dir
        self.query_idx=query_idx
        self.window_duration=window_duration
        self.start_time=self.table.loc[query_idx]["start time"]
        self.participant=self.table.loc[query_idx]["participant"]
    
    def extract_data(self):
        labeling_folder=os.path.join(self.main_dir,"AnnotatedData","Accelerometer_Data","labeling_folder")
        data_dir=os.path.join(self.main_dir,"Data")
    
        clip_path=os.path.join(labeling_folder,"clip{0}_{1}.mp4".format(self.participant,self.query_idx))
    
        participant_folder=find_participant(self.participant,self.main_dir)
        video_path=os.path.join(data_dir,participant_folder,"segments","1","fullstream_resized.mp4")
        assert os.path.isfile(video_path),"Resized version does not exit yet"

        #Extract clip
        start_clip=int(self.start_time)
        ffmpeg_extract_subclip(video_path,start_clip,start_clip+self.window_duration, targetname=clip_path)

        self.data_path=clip_path

    def get_data_path(self):
        #Extract clip if clip has not yet been extracted
        if(not(hasattr(self,"data_path"))):
            self.extract_data()
        return self.data_path
    
    def show_data(self,interactive):
        self.get_data_path()   #Extract path to video clip 
        if(interactive):
            display(Video(self.data_path,embed=True))
        else:
            # Create a VideoCapture object and read from input file
            cap = cv2.VideoCapture(self.data_path)
            # Check if video is correctly open
            assert cap.isOpened(), "Not able to open video"
            # Read until video is completed
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    # Display the resulting frame
                    cv2.imshow('Frame',frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    else: 
                        break
            cap.release()
            cv2.destroyAllWindows()

    def remove_data(self):
        os.remove(self.data_path)

    def get_acc_data(self):
        """Method to extract Acc components of window from participant at start_time + window_duration
            output : dataframe with three acc components and time in separate columns"""

        data_path=os.path.join(self.main_dir,"AnnotatedData","Accelerometer_Data","Participants")
        df_path=os.path.join(data_path,self.participant,"acc_data{}.csv".format(self.participant))
        # cols=["time","AccX","AccY","AccZ"] #We want to extract just acceleration components
        df=pd.read_csv(df_path)
        df=df.set_index("time")
        # subset=df.loc[start_time/T:(start_time/T)+N]


        return df.loc[self.start_time,self.start_time+self.window_duration] 

    # def get_features(self):
    #     return a


class VideoOracle(VideoClip):
    """docstring for DataFormat."""

    def get_target(self):
        clear_output(wait=True)
        self.show_data(interactive=True)
        print("Type class of video...")
        self.target= np.array([int(input())], dtype=int)

    def save_target(self,dir_path):
        #get target value if it wasn´t recorded
        if(not(hasattr(self,"target"))):
            self.get_target()
        #save target value
        self.table.loc[self.query_idx,"target"]=self.target
        #write changes in file
        return self.table

    def display_sample(self):
        #extract sample acc data
        df=self.get_acc_data()
        #create grid for plotting
        
        #Plot each Acc component in separate cells

        return a 



# if __name__ == "__main__":
#     #test oracle
# table_path="C:\\Users\\jeuux\\Desktop\\Carrera\\MoAI\\TFM\\AnnotatedData\\Accelerometer_Data\\Datasets\\acc_dataset_test.csv"
# main_dir="C:\\Users\\jeuux\\Desktop\\Carrera\\MoAI\\TFM"
# oracle_test=VideoOracle(table_path,50,main_dir)

#     print(oracle_test)
#     pass



