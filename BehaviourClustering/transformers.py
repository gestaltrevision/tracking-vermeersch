import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import RandomOverSampler
import os 

class LevelSelector( BaseEstimator, TransformerMixin):
    """Custom Transformer that select the group of behaviours inside a level hierarchy"""

    #Class Constructor 
    def __init__( self,level):
        self._level=level
        
    def _split_target_hierarchy(self,target_list):
        """split target hierarchy inside each target 
        (eg. ['AG Cp Av Cl'] --> ['AG', 'Cp', 'Av', 'Cl'] )"""
        return list(map(lambda target: target.split(" "),target_list))
    
    def _filter_levels(self,target,pos):
        target=target[:pos+2]
        return " ".join(target)
    
    def _get_pos(self,y):
        y_first=y[0]
        return next(idx for idx,_ in enumerate(y_first) if self._level==y_first[idx])

    def _get_valid_idx(self,y):
        return [idx for idx,target in enumerate (y) if self._level in target]

    def fit(self,y):
        return self 
    
    def transform(self, y):
        #filter behaviours
        self.valid_idx= self._get_valid_idx(y)
        y = y[self.valid_idx]
        #split hierarchy
        y=self._split_target_hierarchy(y)
        #get pos
        pos=self._get_pos(y)
        #filter levels
        y=list(map(lambda target: self._filter_levels(target,pos) ,y))
        return  y

#encode classes 
class Oversampler(BaseEstimator, TransformerMixin):
    def __init__(self,folder,level,data_types=[True,True,True],set_cat="Train"):
        #Get folder ("Train" or "Test")
        self.data_folder=os.path.join(folder,set_cat)
        assert os.path.isdir(self.data_folder) 
        self.set_cat=set_cat
        #Load data    
        self.y=np.load(os.path.join(self.data_folder,"targets.npy"))
        self.X=np.load(os.path.join(self.data_folder,"data.npy"))
        #Level selection
        if(level is not None):
            #Level selector
            self.level_selector=LevelSelector(level=level)
            #filter levels
            self.y=self.level_selector.transform(self.y)
            self.X=self.X[self.level_selector.valid_idx]  

        #Save shape variables for shape recovery after sampling                
        self.samples,self.sequence_length,self.n_components=self.X.shape

    def fit(self):
        return self

    def transform(self):
        self.X=self.X.reshape(self.samples,self.sequence_length*self.n_components) #"Flatten" data readings
        #Oversample
        ros = RandomOverSampler(random_state=0)
        X, y = ros.fit_resample(self.X, self.y)
        #Recover original dimensions (samples,seq_length,n_components)
        X=X.reshape(-1,self.sequence_length,self.n_components)
      
        return X,y
    
    def save(self,dir_folder):
        X_reshaped,y_reshaped=self.transform()
        #save targets
        np.save(os.path.join(dir_folder,self.set_cat,"data"),X_reshaped)
        #save samples
        np.save(os.path.join(dir_folder,self.set_cat,"targets"),y_reshaped)
        print("Finished saving oversampled dataset correctly")
        

if __name__ == "__main__":
    folder=r"C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\AnnotatedData\Accelerometer_Data\Datasets\HAR_Dataset_raw"
    dir_folder=r"C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\AnnotatedData\Accelerometer_Data\Datasets\HAR_Dataset_RO"
    over=Oversampler(folder,"AG")
    # over.save(dir_folder)
    x,y=over.transform()
    # from collections import Counter
    # print(sorted(Counter().items()))
    print("HI")
    pass
