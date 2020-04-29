import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin


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
    
    def transform( self, y ):
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