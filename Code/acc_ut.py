from __future__ import division, print_function

import json
import os
import re
from functools import reduce
from math import ceil, floor, pow, sqrt
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from numpy import fft
from pyod.models.iforest import IForest
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tsfresh import extract_relevant_features, select_features
from tsfresh.feature_extraction import (EfficientFCParameters,
                                        MinimalFCParameters, extract_features)
from tsfresh.utilities.dataframe_functions import impute

from general_utilities import find_participant


def create_df_participant(df):
    #Drop NaN
    df=df.dropna()
    #targets for each sliding window (id)
    targets=df["target"].groupby(level="id").agg(return_unique).tolist()
    #Reconstruct comp table
    df=df._drop_axis("target",axis=1)
    #create id
    idx_slice=df.index.tolist() #idx_slice contains ids' and samples
    idx_list=[el[0] for el in idx_slice] #get only ids
    #get targets
    cols=df.columns
    flat_cols=["".join(comp) for comp in cols]
    #"Flat" dataframe
    df_feat=pd.DataFrame(df.values,columns=flat_cols)
    df_feat["id"]=idx_list
    
    #Feature extraction
    extracted_features = extract_features(df_feat,
                                            column_id="id",column_sort="Recording timestamp",
                                            default_fc_parameters=MinimalFCParameters())                          

    #Feature filtering
    impute(extracted_features)
    extracted_features["target"]=targets

    return extracted_features.reset_index(drop=False)



def get_sampling_freq(df,samples_per_sl=100):
    diff_times=[]
    for (start,end) in windows(df.index,samples_per_sl,1):
        diff_times.append(df.loc[end,"Recording timestamp"]-df.loc[start,"Recording timestamp"])
    diff_times=np.asarray(diff_times,dtype=np.int32)
    mean_sampling_freq=np.mean(diff_times)
    return mean_sampling_freq

def get_new_samples(df_resampling,df_base,samples_per_sl=100):
    unsampled_freq=get_sampling_freq(df_resampling,samples_per_sl)
    sampled_freq=get_sampling_freq(df_base,samples_per_sl)

    samples_new=(sampled_freq/unsampled_freq)*samples_per_sl
    
    return int(samples_new)

    
def resample_category_readings(df_resampling,df_base,shorting,samples_per_sl):
    ns_new=get_new_samples(df_resampling,df_base,samples_per_sl) #samples giro == time samples acc 
    df_resampled=pd.DataFrame()   
    for comp in ["X","Y","Z"]:
        resample_component=[]
        col="{}{}".format(shorting,comp)

        for (start,end) in windows(df_resampling.index,ns_new,1):
            raw_signal=df_resampling[col][start:end].values
            resample_component.append(signal.resample(raw_signal,samples_per_sl))
        df_resampled[col]=np.hstack(resample_component)
    return df_resampled



def windows(data, size,factor=2):
    start = 0
    while start + (size / factor) < len(data):
        yield int(start), int(start + size)
        start += (size / factor)

def segment_signal(data,new_samples_number,window_size):
    components=[col for col in data.columns if is_component(col)]
    segments = np.empty((0,new_samples_number,len(components)))
    labels = np.empty((0))

    for (start, end) in windows(data.index, window_size):
        subset=data[start:end]
        if(subset["picture"].mode()[0]!="Null"):
            target=subset["target"].mode()[0]
            labels = np.append(labels,target)
            sample=signal.resample(subset[components].values,new_samples_number)#Resample signals
            sample=np.expand_dims(sample,axis=0) #Add batch dimension
            segments = np.vstack([segments,sample])
        
    if(segments.shape[0]!=labels.shape[0]):
        raise ValueError ("the number of samples MUST be equal to number of labels")
        
    return segments, labels

class state_gen:
    """"""
    def __init__(self,pattern_start,pattern_stop,null_state="Null"):
        self.previous = null_state
        self.null_state=null_state
        self.pattern_start=pattern_start
        self.pattern_stop=pattern_stop

    def __call__(self,actual):
        if(not(pd.isnull(actual))):
            #Start condition
            if((re.search(self.pattern_start,actual))!=None):
                self.previous=actual
                            
            #End Condition
            elif((re.search(self.pattern_stop,actual))!=None):
                self.previous=self.null_state
                            
        return self.previous
      

def is_component(col):
    return ("X" in col) | ("Y" in col) | ("Z" in col)

def get_data_subset(df,category,short):
    """
    """
    base_cols=["Recording timestamp","target","picture"]
    components=["X","Y","Z"]
    data_cols=["{} {}".format(category,component) for component in components]
    df=df[data_cols+base_cols]
    columns_renaming={"{} {}".format(category,component):"{}{}".format(short,component)
                            for component in components}

    df=df.rename(columns=columns_renaming)   
    df=df.dropna().sort_index().drop_duplicates()
    return df.reset_index(drop=True)


def process_readings_participant(df,participant,samples_per_sl=100):
    """

    """
    data_cats=["Accelerometer","Gyro","Gaze point 3D"]
    short_cats=["Acc","Gyro","Gaze"]

    ##get targets and valid intervals
    generator_target=state_gen("AG","Behaviour End")
    df["target"]=df["Event"].apply(lambda event: generator_target(event))
    generator_picture=state_gen("_Y.* IntervalStart","_Y.* IntervalEnd")
    df["picture"]=df["Event"].apply(lambda event: generator_picture(event))

    df_dict={}
    df_dict["Gaze"]=get_data_subset(df,"Gaze point 3D","Gaze")
    #cleaning raw data of each cat(remove Nans, resampling)
    for cat,shorting in zip(data_cats,short_cats):
        if(not("Gaze" in shorting)):
            df_raw=get_data_subset(df,cat,shorting)
            df_dict[shorting]=resample_category_readings(df_raw,df_dict["Gaze"],
                                                        shorting,samples_per_sl)
    #merging...
    data_frames=df_dict.values()
    df_merged = reduce(lambda  left,right: pd.merge(left,right,
                                                    left_index=True,
                                                    right_index=True ),
                                                    data_frames)

                                                    
    return df_merged



if __name__ == "__main__":
    table_path=r"C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\AnnotatedData\Accelerometer_Data\Participants\2504e\CompleteData\CompleteData_raw_2504e.csv"
   
    raw_df=pd.read_csv(table_path,decimal=',')
    df=process_readings_participant(raw_df,"pepelui",100)
    sensor_comp,labels=segment_signal(df,50,100)

    print("HI")
