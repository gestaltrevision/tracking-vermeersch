import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LossTracker:
    def __init__(self, loss_names):
        if "total_loss" not in loss_names:
            loss_names.append("total_loss")
        self.losses = {key: 0 for key in loss_names}
        self.loss_weights = {key: 1 for key in loss_names}

    def weight_the_losses(self, exclude_loss=("total_loss")):
        for k, _ in self.losses.items():
            if k not in exclude_loss:
                self.losses[k] *= self.loss_weights[k]

    def get_total_loss(self, exclude_loss=("total_loss")):
        self.losses["total_loss"] = 0
        for k, v in self.losses.items():
            if k not in exclude_loss:
                self.losses["total_loss"] += v

    def set_loss_weights(self, loss_weight_dict):
        for k, _ in self.losses.items():
            if k in loss_weight_dict:
                w = loss_weight_dict[k]
            else:
                w = 1.0
            self.loss_weights[k] = w

    def update(self, loss_weight_dict):
        self.set_loss_weights(loss_weight_dict)
        self.weight_the_losses()
        self.get_total_loss()

class Trainer(object):

    def __init__(self,
                models,
                loss_fcns,
                loss_names,
                optimizers,
                data_loaders,
                prepare_batch,
                device=None,
                metrics_dict=None,
                loss_weights = None,
                lr_schedulers = None):
                
        self.models = models
        self.loss_fcns = loss_fcns
        self.optimizers =  optimizers
        self.train_loader, self.valid_loader = data_loaders
        self.prepare_batch=prepare_batch
        self.device = device
        self.loss_names = loss_names
        self.loss_weights = loss_weights
        self.metrics_dict = metrics_dict
        self.lr_schedulers = lr_schedulers
        self.initialize_device()
        self.initialize_loss_weights()
        self.initialize_loss_tracker()
        self.initialize_lr_schedulers()
        
    def initialize_loss_tracker(self):
        self.loss_tracker = LossTracker(self.loss_names)
        self.losses = self.loss_tracker.losses
        
    def initialize_device(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_loss_weights(self):
        if self.loss_weights is None:
            self.loss_weights = {k: 1 for k in self.loss_names}

    def initialize_lr_schedulers(self):
        if self.lr_schedulers is None:
            self.lr_schedulers = {}

    def init_metrics(self):
        #init metrics
        metrics={}
        if(type(self.metrics_dict) is dict):
            for key in self.metrics_dict.keys():
                metrics[key] = 0
            for loss in self.losses:
                metrics[loss] = 0

        return metrics 

    def zero_losses(self):
        for k in self.losses.keys():
            self.losses[k] = 0

    def zero_grad(self):
        for v in self.models.values():
            v.zero_grad()
        for v in self.optimizers.values():
            v.zero_grad()

    def create_log_str(self,metrics_dict,log_interval):
        return ["{} = {} ".format(metric,metrics_dict[metric]/log_interval)
                                                        for metric in metrics_dict.keys()]
    def log_metrics(self,metrics_train,metrics_eval,epoch,i,log_interval):
        #Printing results
        for cat,metrics_dict in zip(["Train","Test"],[metrics_train,metrics_eval]):
            log_str = self.create_log_str(metrics_dict,log_interval)
        print(f"{cat} results"+",".join(log_str))
        print("-" * 10)
        #Logging results into TensorBoard
        for metric in metrics_train.keys():
            self.writer.add_scalars(metric,
                            {'Train':metrics_train[metric]/log_interval,
                            'Test':metrics_eval[metric]/log_interval},
                            epoch * len(self.train_loader) + i)
        
      
    def _logit_to_predictions(self,preds,labels):
        """"From model outputs (Unnormalized probabilities,Tensors) 
            to predictions(Classes,np.array)"""
        probabilities=F.softmax(preds,dim=1)
        labels_pred=torch.argmax(probabilities,dim=1)
        labels_pred=labels_pred.to("cpu").numpy().astype(np.int32)
        labels=labels.to("cpu").numpy().astype(np.int32)
        return labels_pred,labels

    def _update_metrics(self,logits,labels,metrics):
        #update losses
        for loss in self.losses.items():
            metrics[loss] += self.losses[loss]
        #update metrics 
        if(type(self.metrics_dict) is dict):
            #From prob to predictions and from tensor to numpy 
            labels_pred,labels=self._logit_to_predictions(logits,labels)
            for metric_name,metric_info in self.metrics_dict.items():
            #Metric with optional parameters (eg. F1 Score)
                if(type(metric_info)==list):
                    metric_fcn=metric_info[0]
                    kwargs=metric_info[1]
                    metrics[metric_name] += metric_fcn(labels_pred,labels,**kwargs)
                #Metric without optional parameters (eg. Balanced Accuracy)
                else:
                    metric_fcn=metric_info
                    metrics[metric_name]+=metric_fcn(labels_pred,labels)
        return metrics 

    def set_to_train(self):
        trainable = [self.models, self.loss_fcns]
        for T in trainable:
            for _, v in T.items():
                v.train()

    def set_to_eval(self):
        for _, v in self.models.items():
            v.eval()

    def _compute_loss_and_metrics(self, batch, metrics):
        samples,labels = self.prepare_batch(batch,self.device)
        embeddings = self.compute_embeddings(samples)
        logits = self.get_logits(embeddings)
        self.losses["classifier_loss"] = self.get_classifier_loss(logits, labels)
        self.loss_tracker.update(self.loss_weights)
        updated_metrics = self._update_metrics(logits,labels,metrics)

        return updated_metrics

    def get_classifier_loss(self, logits, labels):
        if logits is not None:
            return self.loss_fcns["classifier_loss"](logits, labels)
        return 0

    def compute_embeddings(self, data):
        trunk_output = self.get_trunk_output(data)
        embeddings = self.get_final_embeddings(trunk_output)
        return embeddings
    
    def get_trunk_output(self, data):
        return self.models["trunk"](data)

    def get_final_embeddings(self, base_output):
        return self.models["embedder"](base_output)
        
    def get_logits(self, embeddings):
        return self.models["classifier"](embeddings)

    def backward(self):
        self.losses["total_loss"].backward()

    def step_optimizers(self):
        for v in self.optimizers.values():
            v.step()

    def step_lr_plateau_schedulers(self, validation_info):
        if self.lr_schedulers is not None:
            for k, v in self.lr_schedulers.items():
                if k.endswith("plateau"):
                    v.step(validation_info)

    def train_batch(self,batch,metrics):
        self.set_to_train()
        self.zero_losses()
        self.zero_grad()
        metrics = self._compute_loss_and_metrics(batch,metrics)
        self.backward()
        # self.clip_gradients()
        self.step_optimizers()
        return metrics

    def train(self,num_epochs,training_path,early_stopping,log_interval=10):
        #Inits
        self.writer = SummaryWriter(training_path)
        #Training /Eval Loop
        for epoch in tqdm(range(num_epochs)):
            #Init running metrics(Train/Test)
            metrics_train=self.init_metrics()
            metrics_eval= self.init_metrics()

            for i, batch in enumerate(self.train_loader):
                metrics_train=self.train_batch(batch,metrics_train)
                metrics_eval=self.evaluate_set(metrics_eval)        
                
                if (i+1) % log_interval == 0:
                    self.log_metrics(metrics_train,metrics_eval,
                                    epoch,i,log_interval)
                    validation_info = metrics_eval["total_loss"]/log_interval
                    self.step_lr_plateau_schedulers(validation_info)
                    #Check if we are overfitting 
                    early_stopping(validation_info, 
                                    self.models,
                                    training_path)
            
                    if early_stopping.early_stop:
                        print("Early stopping")
                        return True
                    #reset running metrics
                    metrics_train=self.init_metrics()
                    metrics_eval= self.init_metrics()

        self.writer.close()
        return True

    def evaluate_set(self,metrics):
        current_eval_metrics=self.init_metrics()
        for iterations,batch in enumerate(self.valid_loader):
            with torch.no_grad():
                self.set_to_eval()
                current_eval_metrics = self._compute_loss_and_metrics(batch,current_eval_metrics)
        #mean validation metrics
        for metric in current_eval_metrics.keys():
            current_eval_metrics[metric] = current_eval_metrics[metric]/iterations
            metrics[metric]+= current_eval_metrics[metric]

        return metrics
