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

# 

from math import ceil,floor
def preprocess_accdata(df,window_duration=2,sample_rate=10):
    """input:
            window_duration(secs)
            sample_rate(msecs)"""

    df=df[["Participant name","Accelerometer X","Accelerometer Y","Accelerometer Z"]]
    df=df.rename(columns={"Participant name":"participant","Accelerometer X":"AccX",
                                    "Accelerometer Y":"AccY","Accelerometer Z":"AccZ"})

    df=df.dropna().drop_duplicates().reset_index(drop=True)

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


def create_df_participant(df,fc_parameters,window_duration=2,sample_rate=10):
    n_samples=(window_duration*1000)/sample_rate
    extracted_features = extract_features(df, column_id="id",
                                            default_fc_parameters=fc_parameters,
                                            column_sort="time",
                                            column_kind=None, column_value=None)
    
    if("abs_energy" in fc_parameters.keys()):
        energy_cols=[col for col in extracted_features.columns if "energy" in col]
        for col in energy_cols:
            extracted_features[col]=extracted_features.apply(lambda row: energy_to_rms(row[col],n_samples),axis=1)
        
        new_cols={col:"rms_{0}".format( col.split("abs_energy")[0]) for col in energy_cols}
        extracted_features=extracted_features.rename(columns=new_cols)
    # extracted_features["id_info"]=df["id"].unique()
    # extracted_features["start time"]=[ index*window_duration for index in range(len(extracted_features))]
    
    return extracted_features



 


#Feature Extraction functions


"""Detect peaks in data based on their amplitude and other features."""


__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.6"
__license__ = "MIT"


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None, title=True):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    title : bool or string, optional (default = True)
        if True, show standard title. If False or empty string, doesn't show
        any title. If string, shows string as title.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None: 
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
            no_ax = True
        else:
            no_ax = False

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        if title:
            if not isinstance(title, str):
                mode = 'Valley detection' if valley else 'Peak detection'
                title = "%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"% \
                        (mode, str(mph), mpd, str(threshold), edge)
            ax.set_title(title)
        # plt.grid()
        if no_ax:
            plt.show()


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values
 
def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values

def get_first_n_peaks(x,y,no_peaks=5):
    x_, y_ = list(x), list(y)
    if (len(x_) >= no_peaks):
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks
    
def get_features(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y

# t_n = 2; N = 200 ; T = t_n / N;  f_s = 1/T ; denominator = 10


# def extract_window_features(subset, T, N, f_s, denominator):
#     percentile = 5
#     features = []
#     for signal_comp in subset.columns:
#         signal = np.asarray(subset[signal_comp],dtype=np.float32)
#         signal_min = np.nanpercentile(signal, percentile)
#         signal_max = np.nanpercentile(signal, 100-percentile)
#         #ijk = (100 - 2*percentile)/10
#         #The value for minimum peak height is denominator of the maximum value of the signal. 
#         mph = signal_min + (signal_max - signal_min)/denominator
  
#         features += get_features(*get_psd_values(signal, T, N, f_s), mph)
#         features += get_features(*get_fft_values(signal, T, N, f_s), mph)
#         features += get_features(*get_autocorr_values(signal, T, N, f_s), mph)
        
#     return np.array(features)

# from math import ceil
# from tqdm import tqdm
# def create_df_participant(df,participant,window_duration,starts):
#     table={}
#     t_n = 2; N = 200 ; T = t_n / N;  f_s = 1/T ; denominator = 10

#     part_arr=[];start_arr=[];feat_arr=[]
#     #We can paralelize this
#     for start in starts:
#         for index in tqdm(range(ceil(len(df)/(window_duration*N)))):
#             start_time=start + (index*window_duration)
#             df=df.set_index("time")
#             subset=df.loc[start_time*T:(start_time/T)+N] #!!!!!!!!!!
#             part_arr.append(participant)
#             start_arr.append(start_time)
#             feat_arr.append(np.asarray(extract_window_features(subset,T,N,f_s,denominator),dtype=np.float32))
            
#         table["participant"]=part_arr
#         table["start time"]=start_arr
        
# #         table["features"]=np.stack(feat_arr,axis=0)

#         feat_cols = [ 'feat'+str(i) for i in range(feat_arr[1].shape[0]) ]
#         feat_df = pd.DataFrame(np.stack(feat_arr,axis=0),columns=feat_cols)

#     return pd.concat([pd.DataFrame(table),feat_df],axis=1)
    
    

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







# def fix_frequency(times,fq_desired=10):  
#     offset=0
#     for i in range(1,len(times)):
#         dif=(times[i]-(times[i-1]-offset))
#         if(dif!=fq_desired):
#             offset+=fq_desired-dif
#         times[i]+=offset
#     return times

# def process_subtable(df):
#     df=df.dropna()
#     time_offset=min(df["Recording timestamp"])
#     df["Recording timestamp"]=df.apply(lambda row: row["Recording timestamp"]-time_offset,axis=1)
#     df["Recording timestamp"]=fix_frequency(np.asarray(df["Recording timestamp"]))
#     return df