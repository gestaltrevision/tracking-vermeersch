from __future__ import division, print_function

import json
import os
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

#Loading acc_data
def comma_to_float(str_number):
    if(not(pd.isnull(str_number))):
        units,decimals=str_number.split(",")
        factor=pow(10,len(decimals))
        float_number=np.float32(units)+np.float32(decimals)/factor
        return round(float_number,3)
    else:
        return str_number

def get_events(df,event_col="Event"):
    """Get unique behaviours from dataset and create movement label
        Movement label depents on type of behavior
        True=Moving ; False=Still"""
    events=df[event_col].unique().tolist()
    nan_event=next(event for event in events if type(event)!=str)
    events.remove(nan_event)
    #Or-->Orientation Minor (Still); Cp-->Changing Perspective (Movement)
    events_list=[event for event in events if ("Or Mi" in event ) or ("Cp" in event)]

    return events_list

def get_events_index(df,events_list,event_col="Event"):
    col=df[event_col].dropna()
    event_index_dict={event:[] for event in events_list}
    index_list=col.index.tolist()    
    for i,index in enumerate(index_list):
        if(col[index] in events_list):
            #save index and beahaviour
            start=index
            event=col[index]
            #get stop
            try:
                stop=next(index for index in index_list[i:-1] if "00" in col[index])
            except:
                    return ValueError
            #save event, start, stop
            event_index_dict[event].append([start,stop])
    return event_index_dict
    
def create_event_col(df,events_index_dict,event_col="Event"):
    col=df[event_col]
    clean_col=[0]*len(col)
    index=0
    while index < len(col) :
        if (col[index]  in events_index_dict.keys()):
            event=col[index]
            start,stop=next(el for el in events_index_dict[event] if el[0]==index)
            clean_col[start:stop]=[event for _ in range (stop-start)]
            index=stop+1  
        else:
            index+=1   
    return clean_col

def event_to_target(full_event):
    """
    Taxonomy event (Eg. "Or Mi" ,"Cp") Into binary targets
    If event in Changing perspective group (Cp) --> target =1
    If event in Orientation Minor group (Or Mi) --> target = 0
    Rest of the cases --> target =-1 (Will remove these)
    """
    try:
        assert type(full_event)!=int
        if("Cp" in full_event):
            return 1
        elif("Or Mi" in full_event):
            return 0
        else:
            return -1
    except:
        return -1

def create_df_participant(df):
    cols=["id","Recording timestamp",
            "AccX","AccY","AccZ",
            "GyroX","GyroY","GyroZ"]
    df=df.dropna()
    df_targets=df[["id","Event_col"]]
    df_feat=df[cols]
    
    ##Filter data (just get samples from windows with one label) 
    group_targets=df_targets.groupby(df_feat["id"]).aggregate({"Event_col":get_unique_counts }, index=False)

    group_targets["unique_target"]=group_targets.apply(lambda row:type(row["Event_col"])==np.ndarray ,axis=1)
    df_feat_clean=df_feat.set_index("id").loc[group_targets["unique_target"]==False]
    
    group_targets_clean=group_targets[group_targets["unique_target"]==False]
    df_feat_clean=df_feat_clean.reset_index(drop=False)
        
    #Feature extraction
    extracted_features = extract_features(df_feat_clean,
                                            column_id="id",column_sort="Recording timestamp",
                                            default_fc_parameters=EfficientFCParameters())                          

    #Feature filtering
    impute(extracted_features)
    extracted_features["event_target"]=group_targets_clean["Event_col"]
    extracted_features["event_cat"]=extracted_features.event_target.apply(lambda el: create_event_label(el))
    extracted_features["target"] = extracted_features.event_cat.replace({"Or Mi":0 ,"Cp":1})

    return extracted_features.reset_index(drop=False)

##EDA 
def univariate_outlier(df,feat,outliers_fraction=0.05):
    X = df[feat].values.reshape(-1,1)
    # Isolation Forest classifier
    clf= IForest(contamination=outliers_fraction)
    clf.fit(X)
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    return y_pred.tolist()

def get_outliers_dataset(dataset,components,features,outliers_fraction=0.05):
    # components=["AccX","AccY","AccZ"]
    # features=["mean","standard_deviation"]
    outlier_dict={}
    for component in tqdm(components):
        for feat in features:
            feat="{0}__{1}".format(component,feat)
            outlier_dict[feat]=univariate_outlier(dataset,feat,outliers_fraction)
    
    outlier_list=np.any(np.stack(list(outlier_dict.values()),axis=1),axis=1)

    return outlier_dict,outlier_list

def get_series_diff(df,name,samples_per_sl=200):
    #get difference (starting-end times) for each 200 samples
    diff_times=[]
    for idx in range(int(len(df)/samples_per_sl)):
        diff_times.append(df.loc[idx+samples_per_sl,"Recording timestamp"]-df.loc[idx,"Recording timestamp"])
        
    diff_times=np.asarray(diff_times,dtype=np.int32)
    return pd.Series(diff_times,name=name)

def get_samples_giro(df_giro,df_acc,samples_per_sl=200):
    diff_giro=get_series_diff(df_giro,"giro")
    diff_acc=get_series_diff(df_acc,"acc")
    
    mean_sl_giro=diff_giro.describe()["mean"]
    mean_sl_acc=diff_acc.describe()["mean"]
    samples_giro_new=(mean_sl_acc/mean_sl_giro)*samples_per_sl
    
    return int(samples_giro_new)

def butter_lowpass(cutoff, nyq_freq, order=6):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=6):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y
# Filter signal x, result stored to y: 
def filter_noise_gravity(x,fc,sample_rate=100):
    """Function to remove gravity comp from readings and
        filtering the "gravity-free" signal
    input:
            x:signal
            fc:cut-off freq filter
    output:
            filtered_signal 
    """
    cutoff_frequency_gravity = 0.3
    gravity_comp = butter_lowpass_filter(x, cutoff_frequency_gravity, sample_rate/2)
    diff = np.array(x)-np.array(gravity_comp)
    filtered_signal=butter_lowpass_filter(diff,fc,sample_rate/2)
    return filtered_signal

def resample_giro_components(df_giro,df_acc,samples_per_sl):
    ns_giro=get_samples_giro(df_giro,df_acc,samples_per_sl) #samples giro == time samples acc 
    giro_resampled=pd.DataFrame()   
    for comp in ["X","Y","Z"]:
        resample_component=[]
        for i in range(len(df_giro)//ns_giro):
            idx=range(ns_giro*i,ns_giro*(i+1))
            raw_signal=df_giro.loc[idx,"Gyro{}".format(comp)].values
            resample_component.append(signal.resample(raw_signal,samples_per_sl))
        giro_resampled["Gyro{}".format(comp)]=np.hstack(resample_component)
    return giro_resampled

### Process acc_giro
def get_raw_data(df,samples_per_sl):
    """Extracting raw accelerometer and gyroscope data seperately
        in order to preprocess them 
    """
    giro_cols=["Recording timestamp","Gyro X","Gyro Y","Gyro Z","target"]
    df_giro=df[giro_cols]
    df_giro=df_giro.rename(columns={"Gyro X":"GyroX",
                                    "Gyro Y":"GyroY",
                                    "Gyro Z":"GyroZ"})   
                
    df_giro=df_giro.dropna().reset_index(drop=True)

    acc_cols=["Recording timestamp","Accelerometer X","Accelerometer Y","Accelerometer Z","target"]
    df_acc=df[acc_cols]
    df_acc=df_acc.rename(columns={"Accelerometer X":"AccX",
                                    "Accelerometer Y":"AccY",
                                    "Accelerometer Z":"AccZ"})
                                    
    df_acc=df_acc.dropna().reset_index(drop=True)

    ##convert samples from str to float 
    for comp in ["X","Y","Z"]:
        acc_comp,giro_comp="Acc{}".format(comp),"Gyro{}".format(comp)
        df_acc[acc_comp]=df_acc.apply(lambda row :comma_to_float(row[acc_comp]), axis = 1)
        df_giro[giro_comp]=df_giro.apply(lambda row :comma_to_float(row[giro_comp]), axis = 1)
    ##resample
    df_giro=resample_giro_components(df_giro,df_acc,samples_per_sl)
    #merge
    df_raw=pd.merge(df_acc,
                    df_giro,
                    left_index=True,
                    right_index=True,
                    how="left")
    target_data=df_raw[["Recording timestamp","target"]]
    sensors_data=df_raw.drop(columns=["Recording timestamp","target"])
    return target_data, sensors_data

def create_targets_idx(df):
    """input:
           df with Event
            sample_rate(msecs)"""
    #Get available Ground truth 
    events_list=get_events(df)
    events_dict_index=get_events_index(df,events_list)
    df["Event_col"]=create_event_col(df,events_dict_index)
    df["target"]=df.apply(lambda row: event_to_target(row["Event_col"]),axis=1)
    df=df.drop(columns="Event")
    return df

def return_unique(Series):
    unique_values=Series.unique().tolist()
    if len(unique_values)==1:
        return unique_values[0]
    else:
        return -1

def data_to_mult_idx(sensors_data,target_data,participant,samples_per_sl=200):
    n_sliding_windows=floor(len(sensors_data)/samples_per_sl)
    idx_arr=["id{0}_{1}".format(participant,i) for i in range(n_sliding_windows)]
    
    iterables = [idx_arr, range(samples_per_sl)]
    mult_idx=pd.MultiIndex.from_product(iterables, names=['id', 'samples'])
    mult_col=pd.MultiIndex.from_product([["Acc","Gyro"],["X","Y","Z"]], names=['Magnitud', 'Component'])

    df_complete=pd.DataFrame(sensors_data.loc[:len(mult_idx)-1].values,
                            index=mult_idx,
                            columns=mult_col)

    target_data=pd.DataFrame(target_data.loc[:len(mult_idx)-1,"target"].values,
                            index=mult_idx,
                            columns=["target"])
    
    df_complete["target"]=target_data
    return df_complete

def process_readings_participant(df,participant,fc=15,samples_per_sl=200):
    """
    fc=cut-off frequecy for filter
    f=sampling freq (Our sensors have an approx sampling freq of 100 Hz)
    """
    ##get target,valid_ids
    target_df=create_targets_idx(df[["Event","Recording timestamp"]])
    df=pd.merge(df,
                target_df,
                on="Recording timestamp")
    df=df[df["target"]!=-1]
    #Separate raw acc and gyro data
    target_data,sensors_data=get_raw_data(df,samples_per_sl)
    ##Gravity removal and low-pass filter at 15Hz for each comp
    for comp in sensors_data.columns:
        sensors_data[comp]=filter_noise_gravity(sensors_data[comp].values,fc=15)

    #MULTI-INDEX
    df_complete=data_to_mult_idx(sensors_data,target_data,participant,samples_per_sl) 
    ##FILTER IDS
    #get target for each sliding window
    target_series=df_complete["target"].groupby("id").agg(return_unique)
    #keep only "pure" targets (all observations in each sl is the same)
    target_series=target_series[target_series!=-1]
    #filter the dataset, using the previous ids
    df_complete=df_complete.loc[target_series.index]
    
    return df_complete






