import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from tqdm.notebook import tqdm

from sklearn.preprocessing import RobustScaler,StandardScaler
from IMU_ut import plot_grad_flow,prepare_IMU_batch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,matthews_corrcoef

class Trainer(object):
  def __init__(self,model,criterion,learning_rate,data_loaders,device):
    self.model=model.to(device)
    self.criterion=criterion
    self.device=device
    self.train_loader, self.valid_loader = data_loaders
    #Generalize this
    self.optimizer=torch.optim.Adam(
                    self.model.parameters(),
                    lr=learning_rate,
                    #optionals...
                    # momentum=0.9
    )
  
  def inference(self,batch):
    self.model.eval()
    with torch.no_grad():
        samples,labels=prepare_IMU_batch(batch,self.device)
        labels_pred = self.model(samples)
        return labels_pred, labels

  def train_batch(self,batch):
    # Training mode
    self.model.train()
    #get batch data
    samples,labels=prepare_IMU_batch(batch,self.device  )
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
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
    #Optimizer step
    self.optimizer.step()
    
    return loss.item()

  def evaluate_batch(self,batch,metrics,metrics_dict):
    prob_pred,labels=self.inference(batch)
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

  def init_metrics(self,metrics_dict):
    #init metrics
    running_loss = 0.0
    running_metrics={}
    running_metrics["loss"]=0.0

    for key in metrics_dict.keys():
        running_metrics[key]=0.0
    return running_loss, running_metrics

  def log_metrics(self,running_loss,running_metrics,epoch,i,log_interval):
    #running loss
    self.writer.add_scalars('Loss',
                          {'Train':running_loss/log_interval,
                          'Test':running_metrics["loss"]/log_interval},
                            epoch * len(self.train_loader) + i)
    
    print("Current Training Loss {}".format(running_loss / log_interval))

    eval_log_str=["{} = {} ".format(key,running_metrics[key]/log_interval)
                                                    for key in running_metrics.keys()]

    print("Test results:" + ",".join(eval_log_str))

    print("-" * 10)
    
    for metric in running_metrics.keys():
      if(metric!="loss"):
        self.writer.add_scalar(metric, 
                          running_metrics[metric]/log_interval,
                          epoch * len(self.train_loader) + i)

   
  def train(self,num_epochs,training_path,metrics_dict,log_interval=10):
    #Init tensordboard writer for current running
    self.writer = SummaryWriter(training_path)
    best_loss=10e6
    for epoch in tqdm(range(num_epochs)):
        running_loss,running_metrics= self.init_metrics(metrics_dict)
        for i, batch in enumerate(self.train_loader):
            #Train pipeline for current training batch
            running_loss+=self.train_batch(batch)
            #Get validation batch and evaluate current model
            test_batch=next(iter(self.valid_loader))
            running_metrics=self.evaluate_batch(test_batch,
                                                running_metrics,
                                                metrics_dict)
                                                
            #Logging                                     
            if (i+1) % log_interval == 0:
              self.log_metrics(running_loss,
                               running_metrics,
                               epoch,i,
                               log_interval)
              
              if(running_metrics["loss"]<best_loss):
                torch.save(self.model.state_dict(),
                           os.path.join(training_path,"model.pth"))
                              
              #reset running metrics
              running_loss,running_metrics=self.init_metrics(metrics_dict)

    self.writer.close()

