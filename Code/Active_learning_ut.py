##Active learning
from IPython.display import Video,clear_output,display
import os
import pandas as pd
import numpy as np
from general_utilities import find_participant
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip,ffmpeg_resize

        
class VideoClip(object):
    """"""
    def __init__(self,df,query_idx,metadata,main_dir):
        self.table= df
        self.main_dir=main_dir
        self.query_idx=query_idx
        self.start_time=None
        self.participant=None
        for parameter in metadata:
            self.__setattr__(parameter,self.table.loc[query_idx][parameter])


    def extract_data(self,window_duration=2):
        labeling_folder=os.path.join(self.main_dir,"AnnotatedData","Accelerometer_Data","labeling_folder")
        data_dir=os.path.join(self.main_dir,"Data")
    
        clip_path=os.path.join(labeling_folder,"clip{0}_{1}.mp4".format(self.participant,self.query_idx))
    
        participant_folder=find_participant(self.participant,self.main_dir)
        video_path=os.path.join(data_dir,participant_folder,"segments","1","fullstream_resized.mp4")
        assert os.path.isfile(video_path),"Resized version does not exit yet"

        #Extract clip
        start_clip=int(self.start_time)
        ffmpeg_extract_subclip(video_path,start_clip,start_clip+window_duration, targetname=clip_path)

        self.data_path=clip_path

    def get_data_path(self):
        #Extract clip if clip has not yet been extracted
        if(not(hasattr(self,"data_path"))):
            self.extract_data()
        return self.data_path

    # def remove_clip(self)
    # def show_data()


class VideoOracle(VideoClip):
    """docstring for DataFormat."""
   
#     def show_data(self):
#         #read data
#         #display
    def get_target(self):
        clear_output(wait=True)
        self.get_data_path()   
        display(Video(self.data_path,embed=True))
        print("Type class of video...")
        self.target= np.array([int(input())], dtype=int)

    def save_target(self):
        #get target value if it wasn´t recorded
        if(not(hasattr(self,"target"))):
            self.get_target()
        #save target value
        self.table.loc[self.query_idx,"target"]=self.target
    


if __name__ == "__main__":
    #test oracle
    table_path="C:\\Users\\jeuux\\Desktop\\Carrera\\MoAI\\TFM\\AnnotatedData\\Accelerometer_Data\\Datasets\\acc_dataset_test.csv"
    df=pd.read_csv(table_path)
    df=df.rename(columns={"start time":"start_time"})
    metadata=["participant","start_time"]
    main_dir="C:\\Users\\jeuux\\Desktop\\Carrera\\MoAI\\TFM"
    oracle_test=VideoOracle(df,50,metadata,main_dir)

    oracle_test.get_data_path()

    pass
