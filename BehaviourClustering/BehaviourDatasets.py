import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
import joblib
import itertools

from transformers import LevelSelector
from sklearn.preprocessing import LabelEncoder, RobustScaler

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
    def __init__(self,data_folders,scaler,set_cat,level,data_types=[True,True,True],ActiveLearning = False,base_folder = None):
 
        self.targets = self.fetch_data(data_folders,set_cat,"targets")
        self.data = self.fetch_data(data_folders,set_cat,"data")
        self.ActiveLearning = ActiveLearning
        self.base_folder = self.init_base_folder(base_folder,data_folders)
        self.level_selector = None
        self.encoder = self.init_preprocessing_module("encoder",data_folders)
        self.scaler = self.init_preprocessing_module("scaler",data_folders)

        
        #select components...
        self.data_types=data_types
        self.data=self._select_components()
        #Reshape as (readings/component,components) to scale data
        samples,sequence_length,n_components=self.data.shape
        self.data=self.data.reshape(-1,n_components)
        
        # #Scalling/Encoding
        # if(set_cat=="Train" and not(ActiveLearning)):
        #   #scaler
        #   self.scaler=scaler.fit(self.data)
        #   #save scaler (in dataset folder)
        #   joblib.dump(self.scaler,os.path.join(base_folder,'scaler_train.pkl'))
        #   #encoder
        #   self.encoder=LabelEncoder().fit(self.targets)
        #   joblib.dump(self.encoder,os.path.join(base_folder,'encoder_train.pkl'))
   
        # else:
            # #get scaler,encoder file
            # scaler_file=next(file for file in os.listdir(self.base_folder) if "scaler" in file)
            # encoder_file=next(file for file in os.listdir(self.base_folder) if "encoder" in file)
            # self.scaler=joblib.load(os.path.join(self.base_folder,scaler_file))
            # self.encoder=joblib.load(os.path.join(self.base_folder,encoder_file))  
            #  
       


        self.classes = self.encoder.classes_
        self.num_classes=len(self.encoder.classes_)
        assert self.num_classes == 22, print(self.num_classes)
        #filter behaviours by hierarchy
        self.filter_levels(level)
 
        #Data Filtering and Scaling
        self.data=self.scaler.transform(self.data).reshape(samples,sequence_length,n_components)
        if(level is not None):
            self.data=self.data[self.level_selector.valid_idx]

        #Label encoding...
        self.targets = self._encode_targets()
       

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

    def init_preprocessing_module(self,module,data_folders):
        if not(self.ActiveLearning):
            return joblib.load(os.path.join(data_folders,f"{module}.pkl"))
        else:
            return joblib.load(os.path.join(self.base_folder,f"{module}.pkl"))

        
    def init_base_folder(self,base_folder,data_folders):
        if(base_folder ==None):
            base_folder = data_folders[0]
        return base_folder

    def filter_levels(self,level):
        if(level is not None):
            #Level selector
            self.level_selector=LevelSelector(level=level)
            #filter levels
            self.targets=self.level_selector.transform(self.targets)

    def get_class_ratios(self):
        total=len(self.targets) #total instances in dataset

        ratios=[len(self.targets[self.targets==label]) /total
                for label in  range(self.num_classes)]

        return np.array(ratios,dtype=np.float32)

    def fetch_data(self,data_folders,set_cat,data_type):
        data = []
        if(type(data_folders)==list):
            for folder in data_folders:
                data_folder = os.path.join(folder,set_cat)
                assert os.path.isdir(data_folder)
                #Load data    
                data_subset = np.load(os.path.join(data_folder,f"{data_type}.npy"))
                data.append(data_subset)
        else:
            data_folder = os.path.join(data_folders,set_cat)
            assert os.path.isdir(data_folder)
            #Load data    
            data_subset = np.load(os.path.join(data_folder,f"{data_type}.npy"))
            data.append(data_subset)

        return np.concatenate(data)
        
    def check_encodings(self,original_classes):
      for target in set(self.targets):
        if not(target in original_classes):
          return False
        return True

    def _encode_targets(self):
        try:
            return self.encoder.transform(self.targets)
        except:
            #case where we encounter new labeled class not covered on 
            #original dataset
            original_classes = self.encoder.classes_
            actual_classes = set(self.targets)
            unseen_targets = [label for label in actual_classes if not(label in original_classes)]
            invalid_ids =[]
            for unseen_target in unseen_targets:
                invalid_ids.append(np.argwhere(self.targets==unseen_target))
            invalid_ids  = np.concatenate(invalid_ids)
            valid_ids = [id for id in range(len(self.targets)) if not(id in invalid_ids)]
            self.targets = self.targets[valid_ids]
            assert self.check_encodings(original_classes) == True, "Bad encodings"
            self.data = self.data[valid_ids]
            return self.encoder.transform(self.targets)

