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
    def __init__(self,folder,scaler,set_cat,level,data_types=[True,True,True]):
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

import random

class TSDatasetSiamese(TSDataset):
    def __init__(self,folder,scaler,set_cat,level,dataset_length,data_types=[True,True,True]):
        super(TSDatasetSiamese,self).__init__(folder,scaler,set_cat,level,data_types)
        self.dataset_length=dataset_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        label = None
        # ts1 = None
        # ts2 = None
        #ts =="time series"
        if isinstance(index, torch.Tensor):
                index = index.tolist()
                
        # get  sample from same class
        if index % 2 == 1:
            label = 1.0
            #get random index
            idx=random.randint(0,len(self.targets)-1)
            ts1=self.data[idx]
            #get label of random instance
            label_1=self.targets[idx]
            #get random instance with label==label_1
            ind_g = np.squeeze(np.argwhere(self.targets == label_1)) #genuine indexes
            ind_g=int(np.random.choice(ind_g, 1))
            ts2 = self.data[ind_g]
            #debug..
            label_2=self.targets[ind_g]

        # get sample from different class
        else:
            label = 0.0
            #get random index
            idx=random.randint(0,len(self.targets)-1)
            ts1=self.data[idx]
            #get label of random instance
            label_1=self.targets[idx]
            ind_d = np.squeeze(np.argwhere(self.targets != label_1)) #impostor indexes
            #get random instance with label!=label_1
            ind_d=int(np.random.choice(ind_d, 1))
            ts2 = self.data[ind_d]
            #debug 
            label_2=self.targets[ind_d]
       
        # if self.transform:
        #     ts1 = self.transform(ts1)
        #     ts2 = self.transform(ts2)

        return ts1, ts2, torch.from_numpy(np.array([label], dtype=np.float32)),label_1,label_2

if __name__ == "__main__":
    scaler=RobustScaler()
    folder=r"C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\AnnotatedData\Accelerometer_Data\Datasets\HAR_Dataset_raw" #Dataset without label grouping
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyper-parameters
    sequence_length = 50
    hidden_size = 128
    batch_size =128
    num_layers = 2
    num_classes = 2
    n_filters=256
    filter_size=5
    data_types=[True,True,True] # select gaze
    level="Ds"
    n_components=9
    #creating train and valid datasets
    train_dataset= TSDatasetSiamese(folder,scaler,"Train",level,10000,data_types)
    validation_dataset= TSDatasetSiamese(folder,scaler,"Val",level,1000,data_types)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader= DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
    data_loaders=[train_loader,val_loader]

    samples1,samples2,labels=next(iter(train_loader))
    print(len(train_dataset))
    print(samples1.shape)
    print(" ")
    pass


