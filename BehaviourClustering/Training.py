import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from tqdm.notebook import tqdm

from pytorchtools import plot_grad_flow
# from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer(object):

  def __init__(self,model,criterion,opt_parameters,data_loaders,prepare_batch,ratios):
    self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model=model.to(self.device)
    self.criterion=criterion
    self.train_loader, self.valid_loader = data_loaders
    self.optimizer= torch.optim.Adam(
                            self.model.parameters(),
                            **opt_parameters
    )
    self.prepare_batch=prepare_batch
    self.ratios=torch.from_numpy(ratios).to(self.device)

  def init_metrics(self,metrics_dict):
    #init metrics
    running_metrics={}
    running_metrics["loss"]=0.0
    if(type(metrics_dict) is dict):
      for key in metrics_dict.keys():
          running_metrics[key]=0.0
          
    return running_metrics

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

  def inference(self,batch):
    self.model.eval()
    with torch.no_grad():
        samples,labels=self.prepare_batch(batch,self.device)
        preds = self.model(samples)
        return preds, labels

  def _prob_to_predictions(self,preds,labels):
    """"From model outputs (Unnormalized probabilities,Tensors) 
        to predictions(Classes,np.array)"""
    probabilities=F.softmax(preds,dim=1)
    labels_pred=torch.argmax(probabilities,dim=1)
    labels_pred=labels_pred.to("cpu").numpy().astype(np.int32)
    labels=labels.to("cpu").numpy().astype(np.int32)
    return labels_pred,labels

  def _get_metrics(self,preds,labels,metrics,metrics_dict):
    with torch.no_grad():
      #loss
      metrics["loss"]+= self.criterion(preds,labels).item()
      #get metrics
      if(type(metrics_dict) is dict):
        #From prob to predictions and from tensor to numpy 
        labels_pred,labels=self._prob_to_predictions(preds,labels)
        for metric_name,metric_info in metrics_dict.items():
          #Metric with optional parameters (eg. F1 Score)
          if(type(metric_info)==list):
            metric_fcn=metric_info[0]
            kwargs=metric_info[1]
            metrics[metric_name]+=metric_fcn(labels_pred,labels,**kwargs)
          #Metric without optional parameters (eg. Balanced Accuracy)
          else:
            metric_fcn=metric_info
            metrics[metric_name]+=metric_fcn(labels_pred,labels)

      return metrics

  def evaluate_set(self,metrics,metrics_dict):
    actual_metrics=self.init_metrics(metrics_dict)
    for iterations,batch in enumerate(self.valid_loader):
      labels_pred,labels=self.inference(batch)
      self._get_metrics(labels_pred,labels,actual_metrics,metrics_dict)

    #mean validation metrics
    for metric in actual_metrics.keys():
      actual_metrics[metric]=actual_metrics[metric]/iterations
      metrics[metric]+=actual_metrics[metric]

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

  def train(self,num_epochs,training_path,metrics_dict,early_stopping,log_interval=10):
    #Inits
    self.writer = SummaryWriter(training_path)
    scheduler = ReduceLROnPlateau(self.optimizer, 'min')
    #Training /Eval Loop
    for epoch in tqdm(range(num_epochs)):
        #Init running metrics(Train/Test)
        metrics_train=self.init_metrics(metrics_dict)
        metrics_eval= self.init_metrics(metrics_dict)

        for i, batch in enumerate(self.train_loader):

          metrics_train=self.train_batch(batch,
                                         metrics_train,
                                         metrics_dict)
          

          metrics_eval=self.evaluate_set(metrics_eval,
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
            metrics_train=self.init_metrics(metrics_dict)
            metrics_eval= self.init_metrics(metrics_dict)

    self.writer.close()
    return True


class TrainerSiamese(Trainer):

  def __init__(self,model,criterion,opt_parameters,data_loaders,prepare_batch,threshold):
    super(TrainerSiamese,self).__init__(model,criterion,opt_parameters,data_loaders,prepare_batch)
    self.threshold=threshold
      
  def _scores_to_preds(self,scores):
    y_pred=np.empty(len(scores))
    cond=(scores < self.threshold)
    y_pred[cond]=0
    y_pred[~cond]=1
    return y_pred

  def _prob_to_predictions(self,preds,labels):
    """"From model outputs (Unnormalized probabilities,Tensors) 
        to predictions(Classes,np.array)"""

    similarity_scores = F.pairwise_distance(*preds).to("cpu").numpy().astype(np.float32)

    labels_pred=self._scores_to_preds(similarity_scores)

    # labels_pred=labels_pred.to("cpu").numpy().astype(np.int32)
    
    labels=labels.to("cpu").numpy().astype(np.int32)

    return labels_pred,labels
