import numpy as np
import torch
import os
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7,metric_kind ="MCC", verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            metric_kind(str): The score used to keep track of the validation performance (eg. Loss, Mcc...)
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_kind = metric_kind
        self.metric_value_best = np.Inf
        self.delta = delta

    def _scale_metric(self,metric_value):
        return -metric_value if("loss" in self.metric_kind) else metric_value

    def __call__(self, metric_value, models,training_path):

        score = self._scale_metric(metric_value)
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_value, models,training_path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric_value, models, training_path)
            self.counter = 0

    def save_checkpoint(self, metric_value, models ,training_path):
        '''Saves model when metric improves.'''
        if self.verbose:
            print(f'{self.metric_kind} improved \
                    ({self.metric_value_best:.6f}  --> {metric_value:.6f}).\
                    Saving model ...')
            
        for model_name, model in models.items():
            checkpoint_path = os.path.join(training_path, f"{model_name}_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
        self.metric_value_best = metric_value
