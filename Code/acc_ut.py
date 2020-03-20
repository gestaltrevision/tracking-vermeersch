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
    events_list=[event for event in events if ("Or" in event ) or ("Cp" in event)]
    
    events_dict={}
    for event in events_list:
        events_dict[event]= True if "Cp" in event else False
        
    return events_dict
def get_events_index(df,events_dict,event_col="Event"):
    col=df[event_col].dropna()
    event_index_dict={event:[] for event in events_dict.keys()}
    index_list=col.index.tolist()    
    for i,index in enumerate(index_list):
        if(col[index] in events_dict.keys()):
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
            indexes=next(el for el in events_index_dict[event] if el[0]==index)
            start=indexes[0]
            stop=indexes[1]
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
def preprocess_accdata(df,window_duration=2,sample_rate=10):
    """input:
            window_duration(secs)
            sample_rate(msecs)"""

    df=df.drop(columns={"Event value","Participant name.1"})
    df=df.rename(columns={"Participant name":"participant","Accelerometer X":"AccX",
                                    "Accelerometer Y":"AccY","Accelerometer Z":"AccZ"})

    df=df.reset_index(drop=True)

    #Get available Ground truth 
    events_dict=get_events(df)
    events_dict_index=get_events_index(df,events_dict)


    df["Event_col"]=create_event_col(df,events_dict_index)
    df["target"]=df.Event_col.apply(lambda element: get_target(element,events_dict))

    df=df.drop(columns="Event").drop_duplicates().dropna().reset_index(drop=True)

  
    #Clean raw accelerometer readings
    n_samples=int((window_duration*1000)/sample_rate)
    participant=df.loc[0,"participant"]
    #Create array with id's for feature extraction(each window has its own id)
    id_arr=["{0}_{1}".format(participant,idx) for idx in range(floor(len(df)/n_samples)) for _ in range(n_samples)]
    id_series=pd.Series(id_arr,name="id")
    df=pd.concat((df,id_series),axis=1).dropna()
    df["time"]=[sample_rate*count for count in range(len(df))]

    iter_cols=["AccX","AccY","AccZ"]
    for col in iter_cols:
        df[col]=df.apply(lambda row :comma_to_float(row[col]), axis = 1)

    return df



from math import sqrt
def energy_to_rms(abs_energy,n):
    return sqrt((1/n)*abs_energy)

from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

def get_median_value(Series):
    diff_values=Series.value_counts()
    median_idx,_=max(zip(diff_values.index,diff_values),key=lambda x:x[1])
    return median_idx
def create_df_participant(df,fc_parameters,window_duration=2,sample_rate=10):
    cols=["id","Recording timestamp","AccX","AccY","AccZ"]
    dfx=df[["id","Event_col","target"]]
    df=df[cols]
    n_samples=(window_duration*1000)/sample_rate
    extracted_features = extract_features(df, column_id="id",
                                            default_fc_parameters=fc_parameters,
                                            column_sort="Recording timestamp",
                                            column_kind=None, column_value=None)


    
    if("abs_energy" in fc_parameters.keys()):
        energy_cols=[col for col in extracted_features.columns if "energy" in col]
        for col in energy_cols:
            extracted_features[col]=extracted_features.apply(lambda row: energy_to_rms(row[col],n_samples),axis=1)
        
        new_cols={col:"rms_{0}".format( col.split("abs_energy")[0]) for col in energy_cols}
        extracted_features=extracted_features.rename(columns=new_cols)

    group_df=dfx.groupby(dfx["id"]).aggregate({"Event_col":get_median_value,
                                                "target":get_median_value}, index=False)
    #Add participant ,target and event columns
    extracted_features["Event_col"]=group_df["Event_col"]
    extracted_features["target"]=group_df["target"]

    return extracted_features





import ntpath
def get_participant(file_path,common_str):
    path,_ = os.path.splitext(file_path)
    file_name=ntpath.basename(path)

    participant=str.split(file_name,common_str)[1]
    return participant


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