import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from tqdm.notebook import tqdm

from sklearn.preprocessing import RobustScaler,StandardScaler
from pytorchtools import plot_grad_flow,prepare_IMU_batch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,matthews_corrcoef

from torch.optim.lr_scheduler import ReduceLROnPlateau
class Trainer(object):
  def __init__(self,model,criterion,opt_parameters,data_loaders,prepare_batch):
    self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model=model.to(self.device)
    self.criterion=criterion
    self.train_loader, self.valid_loader = data_loaders
    self.optimizer= torch.optim.Adam(
                            self.model.parameters(),
                            **opt_parameters
    )
    self.prepare_batch=prepare_batch
  
  def inference(self,batch):
    self.model.eval()
    with torch.no_grad():
        samples,labels=self.prepare_batch(batch,self.device)
        labels_pred = self.model(samples)
        return labels_pred, labels
  
  def _get_metrics(self,prob_pred,labels,metrics,metrics_dict):
    with torch.no_grad():
      #loss
      metrics["loss"]+= self.criterion(prob_pred,labels).item()
      #transform tensors to numpy arrays
      probabilities=torch.exp(prob_pred)
      labels_pred=torch.argmax(probabilities,dim=1)
      labels_pred=labels_pred.to("cpu").numpy().astype(np.int32)
      labels=labels.to("cpu").numpy().astype(np.int32)
      #get metrics
      for metric_name,metric in metrics_dict.items():
        metrics[metric_name]+=metric(labels_pred,labels)
      return metrics
      
  def train_batch(self,batch,metrics,metrics_dict):
    # Training mode
    self.model.train()
    #get batch data
    samples,labels=self.prepare_batch(batch,self.device)
    # Forward pass
    outputs = self.model(samples)
    loss = self.criterion(outputs, labels)
    assert(not(np.isnan(loss.item()))), "Your model exploded"
    # Backward and optimize
    self.optimizer.zero_grad()
    loss.backward()

    #Gradient Flow to debug
    # plot_grad_flow(model.named_parameters())
    # Gradient clipping(prevent gradient explotion)
    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
    #Optimizer step
    self.optimizer.step()
    
    #get metrics
    return self._get_metrics(outputs,labels,metrics,metrics_dict)

  def evaluate_batch(self,batch,metrics,metrics_dict):
    prob_pred,labels=self.inference(batch)
    return self._get_metrics(prob_pred,labels,metrics,metrics_dict)


  def init_metrics(self,metrics_dict):
    #init metrics
    running_metrics={}
    running_metrics["loss"]=0.0
    for key in metrics_dict.keys():
        running_metrics[key]=0.0

    yield running_metrics

  def log_metrics(self,metrics_train,metrics_eval,epoch,i,log_interval):
    #Outputing results
    for cat,metrics in zip(["Train","Test"],[metrics_train,metrics_eval]):
      log_str=["{} = {} ".format(key,metrics[key]/log_interval)
                                                    for key in metrics.keys()]
      print(f"{cat} results"+",".join(log_str))

    print("-" * 10)
    #Logging results into TensorBoard
    for metric in metrics_train.keys():
      self.writer.add_scalars(metric,
                          {'Train':metrics_train[metric]/log_interval,
                          'Test':metrics_eval[metric]/log_interval},
                            epoch * len(self.train_loader) + i)

  def train(self,num_epochs,training_path,metrics_dict,early_stopping,log_interval=50):
    #Inits
    self.writer = SummaryWriter(training_path)
    scheduler = ReduceLROnPlateau(self.optimizer, 'min')
    #Training /Eval Loop
    for epoch in tqdm(range(num_epochs)):
        #Init running metrics(Train/Test)
        metrics_train=next(self.init_metrics(metrics_dict))
        metrics_eval= next(self.init_metrics(metrics_dict))

        for i, batch in enumerate(self.train_loader):

          metrics_train=self.train_batch(batch,
                                         metrics_train,
                                         metrics_dict)
          
          test_batch=next(iter(self.valid_loader))

          metrics_eval=self.evaluate_batch(test_batch,
                                              metrics_eval,
                                              metrics_dict)
                                              
          
          if (i+1) % log_interval == 0:
            self.log_metrics(metrics_train,
                            metrics_eval,
                            epoch,i,
                            log_interval)
            
            scheduler.step(metrics_eval["loss"]/log_interval)

            #Check if we are overfitting 
            early_stopping(metrics_eval["loss"]/log_interval, 
                            self.model,
                            training_path)
      
            if early_stopping.early_stop:
                print("Early stopping")
                return True
                            
            #reset running metrics
            metrics_train=next(self.init_metrics(metrics_dict))
            metrics_eval= next(self.init_metrics(metrics_dict))

    self.writer.close()
    return True
