import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
import joblib
import itertools

from transformers import LevelSelector
from sklearn.preprocessing import LabelEncoder, RobustScaler

class TSDataset(Dataset):
    """
    Dataset class for Loading Time Series (Ts) signals(Accelerometer + Gyroscope + Gaze coordinates) 
    over different sliding windows
    input:
        folder: path to main dir of dataset (root containing Train and Test Folder)
        scaler: type of scaler (eg. StandardScaler) to be applied to data
            if we are loading the training set(train=True) 
                we define the scaler over training set and we save it
                
            When loading test set, we ommit this value and fit the previously fit scaler over training set
        set_cat= string with name of set of data (Train , Validation or Test)

        "Data_types" = Array with one boolean flag indicating weather the correspondant type of
                        data is used or not.

        Data_types : [Gaze,Acc,Gyro] , Hence if we have [True,True,False] we use all data components
                        except Gyro components
    output:
        samples (n_samples,sequence_length,n_components):
            Array containing the readings of each component over a number of different windows(n_samples)
        targets (n_samples)
            Array containing the target (Binary) over each of the correspondant samples
        
    """
    def __init__(self,folder,scaler,set_cat,level,data_types=[True,True,True],ActiveLearning = False,base_folder = None):
        #Get folder ("Train" or "Test")
        self.data_folder=os.path.join(folder,set_cat)
        assert os.path.isdir(self.data_folder) 
        #Load data    
        self.targets=np.load(os.path.join(self.data_folder,"targets.npy"))
        self.data=np.load(os.path.join(self.data_folder,"data.npy"))

        if(level is not None):
            #Level selector
            self.level_selector=LevelSelector(level=level)
            #filter levels
            self.targets=self.level_selector.transform(self.targets)

        #select components...
        self.data_types=data_types
        self.data=self._select_components()
        #Reshape as (readings/component,components) to scale data
        samples,sequence_length,n_components=self.data.shape
        self.data=self.data.reshape(-1,n_components)
        
        #Scalling/Encoding
        if(set_cat=="Train" and not(ActiveLearning)):
          #scaler
          self.scaler=scaler.fit(self.data)
          #save scaler (in dataset folder)
          joblib.dump(self.scaler,os.path.join(folder,'scaler_train.pkl'))
          #encoder
          self.encoder=LabelEncoder().fit(self.targets)
          joblib.dump(self.encoder,os.path.join(folder,'encoder_train.pkl'))

        if(base_folder ==None):
            base_folder = folder
            
        else:
          #get scaler,encoder file
          scaler_file=next(file for file in os.listdir(base_folder) if "scaler" in file)
          encoder_file=next(file for file in os.listdir(base_folder) if "encoder" in file)
          self.scaler=joblib.load(os.path.join(base_folder,scaler_file))
          self.encoder=joblib.load(os.path.join(base_folder,encoder_file))         

        #Data Filtering and Scaling
        self.data=self.scaler.transform(self.data).reshape(samples,sequence_length,n_components)
        if(level is not None):
            self.data=self.data[self.level_selector.valid_idx]

        #Label encoding...
        self.targets=self.encoder.transform(self.targets)
        self.classes=self.encoder.classes_
        self.num_classes=len(self.classes)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
        return [self.data[idx], self.targets[idx]]

    def _select_components(self):
        assert len(self.data_types)==3 , "Need full combination of data types"

        component_idxs=[[0,1,2],[3,4,5],[6,7,8]] # ids [Gaze,Acc,Gyro]

        idx_selected =[component_idx * component_flag 
                        for i,(component_flag,component_idx) in 
                        enumerate(zip(self.data_types,component_idxs))]

        idx_selected=itertools.chain.from_iterable(idx_selected)
        idx_selected=list(idx_selected)
        
        return self.data[:,:,idx_selected]
    
    def get_class_ratios(self):
        total=len(self.targets) #total instances in dataset

        ratios=[len(self.targets[self.targets==label]) /total
                for label in  range(self.num_classes)]

        return np.array(ratios,dtype=np.float32)

class TSDatasetML(Dataset):
    """
    Dataset class for Loading Time Series (Ts) signals(Accelerometer + Gyroscope + Gaze coordinates) 
    over different sliding windows
    input:
        folder: path to main dir of dataset (root containing Train and Test Folder)
        scaler: type of scaler (eg. StandardScaler) to be applied to data
            if we are loading the training set(train=True) 
                we define the scaler over training set and we save it
                
            When loading test set, we ommit this value and fit the previously fit scaler over training set
        set_cat= string with name of set of data (Train , Validation or Test)

        "Data_types" = Array with one boolean flag indicating weather the correspondant type of
                        data is used or not.

        Data_types : [Gaze,Acc,Gyro] , Hence if we have [True,True,False] we use all data components
                        except Gyro components
    output:
        samples (n_samples,sequence_length,n_components):
            Array containing the readings of each component over a number of different windows(n_samples)
        targets (n_samples)
            Array containing the target (Binary) over each of the correspondant samples
        
    """
    def __init__(self,folder,targets,data,scaler,set_cat,level = None ,data_types=[True,True,True]):
        #Get folder ("Train" or "Test")
        #Load data    
        self.targets = targets
        self.data = data

        if(level is not None):
            #Level selector
            self.level_selector=LevelSelector(level=level)
            #filter levels
            self.targets=self.level_selector.transform(self.targets)

        #select components...
        self.data_types=data_types
        self.data=self._select_components()
        #Reshape as (readings/component,components) to scale data
        samples,sequence_length,n_components=self.data.shape
        self.data=self.data.reshape(-1,n_components)
        
        #Scalling/Encoding
        if(set_cat=="Train"):
          #scaler
          self.scaler=scaler.fit(self.data)
          #save scaler (in dataset folder)
          joblib.dump(self.scaler,os.path.join(folder,'scaler_train.pkl'))
          #encoder
          self.encoder=LabelEncoder().fit(self.targets)
          joblib.dump(self.encoder,os.path.join(folder,'encoder_train.pkl'))

        else:
          #get scaler,encoder file
          scaler_file=next(file for file in os.listdir(folder) if "scaler" in file)
          encoder_file=next(file for file in os.listdir(folder) if "encoder" in file)
          self.scaler=joblib.load(os.path.join(folder,scaler_file))
          self.encoder=joblib.load(os.path.join(folder,encoder_file))         

        #Data Filtering and Scaling
        self.data=self.scaler.transform(self.data).reshape(samples,sequence_length,n_components)
        if(level is not None):
            self.data=self.data[self.level_selector.valid_idx]

        #Label encoding...
        self.targets=self.encoder.transform(self.targets)
        self.classes=self.encoder.classes_
        self.num_classes=len(self.classes)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
        return [self.data[idx], self.targets[idx]]

    def _select_components(self):
        assert len(self.data_types)==3 , "Need full combination of data types"

        component_idxs=[[0,1,2],[3,4,5],[6,7,8]] # ids [Gaze,Acc,Gyro]

        idx_selected =[component_idx * component_flag 
                        for i,(component_flag,component_idx) in 
                        enumerate(zip(self.data_types,component_idxs))]

        idx_selected=itertools.chain.from_iterable(idx_selected)
        idx_selected=list(idx_selected)
        
        return self.data[:,:,idx_selected]
    
    def get_class_ratios(self):
        total=len(self.targets) #total instances in dataset

        ratios=[len(self.targets[self.targets==label]) /total
                for label in  range(self.num_classes)]

        return np.array(ratios,dtype=np.float32)