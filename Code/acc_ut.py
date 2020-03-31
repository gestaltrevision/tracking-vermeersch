from __future__ import division, print_function

import json
import os
from math import pow
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from numpy import fft
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import welch

from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features

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
def get_target(element,events_dict):
    try:
        return events_dict[element]
    except:
        return -1

from math import ceil,floor
def create_targets_idx(df,n_samples):
    """input:
            window_duration(secs)
            sample_rate(msecs)"""
    #Get available Ground truth 
    events_list=get_events(df)
    events_dict_index=get_events_index(df,events_list)
    df["Event_col"]=create_event_col(df,events_dict_index)
    
    df=df[df["Event_col"]!=0]
    df=df.drop(columns="Event").drop_duplicates().dropna().reset_index(drop=True)

    #Clean raw accelerometer readings
    participant=df.loc[0,"participant"]
    
    #Create array with id's for feature extraction(each window has its own id)
    id_arr=["{0}_{1}".format(participant,idx) for idx in range(floor(len(df)/n_samples)) for _ in range(n_samples)]
    id_series=pd.Series(id_arr,name="id")
    df=pd.concat((df,id_series),axis=1).dropna()
    
    for col in df.columns:
        if(("Gyro" in col) or ("Acc" in col)):
            df[col]=df.apply(lambda row :comma_to_float(row[col]), axis = 1)
    return df


from math import sqrt
def energy_to_rms(abs_energy,n):
    return sqrt((1/n)*abs_energy)

from tsfresh.feature_extraction import extract_features, MinimalFCParameters,EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

def get_median_value(Series):
    diff_values=Series.value_counts()
    median_idx,_=max(zip(diff_values.index,diff_values),key=lambda x:x[1])
    return median_idx

def get_unique_counts(Series):
    diff_values=Series.unique()
    return diff_values
def create_event_label(full_event):
    categories=["0","Cp","Or Mi","Or Ma"]
    try:
        return next(cat for cat in categories if cat in full_event)
    except:
        return full_event

def create_df_participant(df):
    cols=["id","Recording timestamp","AccX","AccY","AccZ"]
    df_targets=df[["id","Event_col"]]
    df_feat=df[cols]
    
    ##Filter data (just get samples from windows with one label) 
    group_targets=df_targets.groupby(df_targets["id"]).aggregate({"Event_col":get_unique_counts }, index=False)

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
    extracted_features["target"] = extracted_features.event_cat.replace({"0":-1, "Or Ma": 0, "Or Mi":1 ,"Cp":2})

    return extracted_features.reset_index(drop=False)





from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_scaler(df,cols):
    #Normalization
    x = df[cols].values
    scaler = StandardScaler().fit(x) #we have mean,std info in scaler
    return scaler

##EDA 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from tqdm import tqdm
from pyod.models.iforest import IForest
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



def get_series_diff(df,name,n_samples=200):
    #get difference (starting-end times) for each 200 samples
    diff_times=[]
    for idx in range(int(len(df)/n_samples)):
        diff_times.append(df.loc[idx+n_samples,"Recording timestamp"]-df.loc[idx,"Recording timestamp"])
        
    diff_times=np.asarray(diff_times,dtype=np.int32)
    return pd.Series(diff_times,name=name)

def get_samples_giro(df_giro,df_acc,n_samples_acc=200):
    diff_giro=get_series_diff(df_giro,"giro")
    diff_acc=get_series_diff(df_acc,"acc")
    
    mean_sl_giro=diff_giro.describe()["mean"]
    mean_sl_acc=diff_acc.describe()["mean"]
    samples_giro_new=(mean_sl_acc/mean_sl_giro)*n_samples_acc
    
    return int(samples_giro_new)

def filtered_dataset_acc(df,sos):
    #get index list
    idx_list=df["id"].unique().tolist()
    df=df.set_index("id",drop=True)

    newX,newY,newZ=[],[],[]
    for idx in idx_list:
        newX.append(signal.sosfilt(sos,df.loc[idx,"AccX"]))
        newY.append(signal.sosfilt(sos,df.loc[idx,"AccY"]))
        newZ.append(signal.sosfilt(sos,df.loc[idx,"AccZ"]))

    df["AccX"]=np.hstack(newX)
    df["AccY"]=np.hstack(newY)
    df["AccZ"]=np.hstack(newZ)
    
    return df.reset_index(drop=False)
    
f=100
fc=10
ws=(fc/f)
#Filter Design
sos = signal.ellip(8, 0.001,40,ws,output='sos')
def filtered_dataset_giro(df,sos):
    #get index list
    idx_list=df["id"].unique().tolist()
    df=df.set_index("id",drop=False)

    newX,newY,newZ=[],[],[]
    for idx in idx_list:
        comp_filtered=signal.sosfilt(sos,df.loc[idx,"GyroX"])
        comp_resampled=signal.resample(comp_filtered,200)
        newX.append(comp_resampled)
        
        comp_filtered=signal.sosfilt(sos,df.loc[idx,"GyroY"])
        comp_resampled=signal.resample(comp_filtered,200)
        newY.append(comp_resampled)
        
        comp_filtered=signal.sosfilt(sos,df.loc[idx,"GyroZ"])
        comp_resampled=signal.resample(comp_filtered,200)
        newZ.append(comp_resampled)
        
    df_new=pd.DataFrame()   
    df_new["GyroX"]=np.hstack(newX)
    df_new["GyroY"]=np.hstack(newY)
    df_new["GyroZ"]=np.hstack(newZ)
    
    return df_new

### Process acc_giro
def get_raw_data(df):
    df_giro=df[["Recording timestamp",
                "Gyro X",
                "Gyro Y",
                "Gyro Z",
                "Event","Participant name"]]
    df_giro=df_giro.rename(columns={"Participant name":"participant",
                                    "Gyro X":"GyroX",
                                    "Gyro Y":"GyroY",
                                    "Gyro Z":"GyroZ"})
    df_giro=df_giro.reset_index(drop=True)

    df_acc=df[["Recording timestamp",
                "Accelerometer X",
                "Accelerometer Y",
                "Accelerometer Z",
                "Event","Participant name"]]
    df_acc=df_acc.rename(columns={"Participant name":"participant",
                                    "Accelerometer X":"AccX",
                                    "Accelerometer Y":"AccY",
                                    "Accelerometer Z":"AccZ"})
    df_acc=df_acc.reset_index(drop=True)

    return df_acc,df_giro

def process_readings_participant(df,fc,n_samples):
    """
    fc=cut-off frequecy for filter
    f=sampling freq (Our sensors have an approx sampling freq of 100 Hz)
    """
    #Separate raw acc and gyro data
    df_acc,df_giro=get_raw_data(df)
    n_samples_giro=get_samples_giro(df_giro,df_acc,n_samples)

    #Create filter to remove noise
    f=100
    filter_sos=signal.ellip(8,0.001,40,fc/f,output='sos')
    #Process acc data
    df_acc=create_targets_idx(df_acc,n_samples)
    df_acc=filtered_dataset_acc(df_acc,filter_sos)
    #Process giro data 
    df_giro=create_targets_idx(df_giro,n_samples_giro)
    df_giro=filtered_dataset_giro(df_giro,filter_sos)
    #Merge
    df_complete=pd.merge(df_acc,
                        df_giro,
                        left_index=True,
                        right_index=True,
                        how="left")

    return df_complete
