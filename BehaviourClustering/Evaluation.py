import os
import warnings
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn.functional as F
from pytorchtools import to_numpy
from torch.utils.tensorboard import SummaryWriter
from Training_modular import Trainer

warnings.filterwarnings("ignore")

class Evaluator(Trainer):

    def __init__(self,
                models,
                prepare_batch,
                dataset,
                data_loader,
                results_folder,
                temperature = None,
                device=None,
                metrics_dict= None):
                
        self.models = models
        self.prepare_batch=prepare_batch
        self.device = device
        self.dataset=dataset
        self.data_loader=data_loader
        self.n_classes=dataset.num_classes
        self.metrics_dict = metrics_dict
        self.temperature = temperature
        self.results_folder = results_folder
        self.categorical_labels = self.dataset.classes
        self.true_labels = None
        self.predicted_best_probabilities = None
        self.predicted_labels = None
        self.predicted_probabilities = None
        self.initialize_device()
        self.initialize_metric_dict()

    def is_defined(self,argument):
        if(getattr(self,argument)==None):
            raise ValueError

    def init_metrics(self):
        #init metrics
        metrics={}
        for key in self.metrics_dict.keys():
            metrics[key] = []
        metrics["coverages"] = []
        return metrics 
    
    def initialize_writer(self,writer_folder):
      if not os.path.exists(writer_folder):
          os.makedirs(writer_folder)
      return SummaryWriter(writer_folder)

    def to_tensorboard(self,base_dir,title,fig,epoch):
      writer = self.initialize_writer(base_dir)
      writer.add_figure(tag= title,figure = fig, global_step = epoch)

    def save_fig(self,title,ext,fig,epoch):
        # Save
        base_dir  = os.path.join(self.results_folder,title)
        if not(os.path.isdir(base_dir)):
          os.makedirs(base_dir)
        fig_file= f"{title}_v{epoch}.{ext}"
        fig_dir = os.path.join(base_dir,fig_file)
        fig.savefig(fig_dir)
        self.to_tensorboard(base_dir,title,fig,epoch)

    def _normalize_probabilities(self,preds):
        """Network outputs (logits)
        to normalized probabilities"""
        # preds=preds/self.ratios #thresholding
        return F.softmax(preds,dim=1)
    
    def get_valid_ids(self,probabilities,conf_thresh):
        valid_idx = [idx for idx,prob in enumerate(probabilities)
                        if np.max(prob) > conf_thresh]
        return valid_idx

    def compute_coverage(self,filtered_labels):
        return len(filtered_labels)/len(self.true_labels)

    def get_best_probabilities(self,probabilities,labels_pred):
      best_probabilities = [probabilities[batch_id,best_class]
                                  for batch_id,best_class in enumerate(labels_pred)]
      return best_probabilities
    
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def get_logits(self,embeddings):
        """Return logits of model 
        If model has been calibrated, then scale the computed logits
        using Temperature scaling previously computed
        """
        logits  = self.models["classifier"](embeddings)
        try:
          # Scale logits using temperature scaling
          self.is_defined("temperature")
          logits = self.temperature_scale(logits)
          return logits
        except:
          return logits
            
    def _logits_to_predictions(self,logits,labels):
        """"From model outputs (Unnormalized probabilities,Tensors) 
            to predictions(Classes,np.array)"""
        probabilities = self._logit_to_probabilies(logits)
        #Filtering predicted labels with model confidence below threshold
        labels_pred = torch.argmax(probabilities,dim=1)
        labels_pred = to_numpy(labels_pred)
        labels = to_numpy(labels)
        probabilities = to_numpy(probabilities)
        return probabilities, labels_pred, labels

    def inference (self,batch):
        self.set_to_eval()
        with torch.no_grad():
            samples, labels = self.prepare_batch(batch,self.device)
            embeddings = self.compute_embeddings(samples)
            logits = self.get_logits(embeddings)
        return logits, labels

    def _get_set_predictions(self):
        true_labels=[]
        predicted_labels=[]
        predicted_probabilities = []
        predicted_best_probabilities = []

        for batch in self.data_loader:
            logits,labels = self.inference(batch)
            probabilities_pred, labels_pred, labels = self._logits_to_predictions(logits,labels)
            best_probabilities = self.get_best_probabilities(probabilities_pred,labels_pred)
            true_labels.append(labels)
            predicted_labels.append(labels_pred)
            predicted_probabilities.append(probabilities_pred)
            predicted_best_probabilities.append(best_probabilities)

        self.true_labels = np.concatenate(true_labels)
        self.predicted_labels = np.concatenate(predicted_labels)
        self.predicted_probabilities = np.concatenate(predicted_probabilities)
        self.predicted_best_probabilities = np.concatenate(predicted_best_probabilities)

    def compute_set_embeddings(self):
        s, e = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.data_loader)):
                samples, labels = self.prepare_batch(batch,self.device)
                q = self.compute_embeddings(samples)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                if i == 0:
                    all_labels = torch.zeros(len(self.data_loader.dataset), labels.size(1))
                    all_q = torch.zeros(len(self.data_loader.dataset), q.size(1))
                e = s + q.size(0)
                all_q[s:e] = q
                all_labels[s:e] = labels
                s = e
            all_labels = all_labels.cpu().numpy()
            all_q = all_q.cpu().numpy()

        return all_q, all_labels


    def filter_predictions(self,conf_thresh):
        try:
            self.is_defined("predicted_best_probabilities")
        except:
            self._get_set_predictions()
        valid_idx = self.get_valid_ids(self.predicted_best_probabilities,conf_thresh)
        true_labels_filtered = self.true_labels[valid_idx]
        pred_labels_filtered = self.predicted_labels[valid_idx]

        return true_labels_filtered, pred_labels_filtered

    def get_metrics_with_threshold(self,conf_thresh):
        try:
            self.is_defined("predicted_labels")
        except:
            self._get_set_predictions()
        true_labels_filtered, pred_labels_filtered = self.filter_predictions(conf_thresh)
        report = classification_report(true_labels_filtered,pred_labels_filtered)
        return report
       
    def _update_test_metrics(self,true_labels,predicted_labels,metrics):
        for metric_name,metric_info in self.metrics_dict.items():
        #Metric with optional parameters (eg. F1 Score)
            if(type(metric_info)==list):
                metric_fcn=metric_info[0]
                kwargs=metric_info[1]
                metrics[metric_name].append(metric_fcn(true_labels,predicted_labels,**kwargs))
            #Metric without optional parameters (eg. Balanced Accuracy)
            else:
                metric_fcn=metric_info
                metrics[metric_name].append(metric_fcn(true_labels,predicted_labels))
        #update coverages
        coverage  = self.compute_coverage(true_labels)
        metrics["coverages"].append(coverage)
        return metrics

    #plotting
    def plot_confusion_matrix(self,
                             true_labels,
                             predicted_labels,
                             epoch =0, 
                             title="", ext = "png", 
                             save = True, normalize = True):
        try:
            cnf_matrix = confusion_matrix(true_labels, 
                                            predicted_labels,labels=range(self.n_classes))
            if (normalize):
              cnf_matrix= cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
              cnf_matrix  = np.around(cnf_matrix,2)
              
        except:
            cnf_matrix  = np.zeros((self.n_classes,self.n_classes))
            print("Not enough samples to compute confusion matrix")
       
        df = pd.DataFrame(cnf_matrix,self.categorical_labels,self.categorical_labels)
        fig = plt.figure(figsize=(15,10),dpi = 125)

        sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
        sns.heatmap(df,cmap ="Oranges", annot=True) 
        plt.title(title)
        if save:
            self.save_fig(title,ext,fig,epoch)

    def plot_metric(self,
                    xaxis,yaxis,
                    metrics,
                    title="", ext = "png", 
                    save = True):
   
        fig = plt.figure(figsize=(15,10),dpi = 125)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(metrics[xaxis],metrics[yaxis])
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ax.set_title(title)
        if save:
            self.save_fig(title,ext,fig,0)
            

    def plot_class_performance(self,
                               scores,
                               score_name,
                               epoch = 0,
                               title="",
                               ext="png",save=True):

      fig = plt.Figure(figsize=(15,10), facecolor="w", edgecolor="k")

      ax = fig.add_subplot(1, 1, 1)

      tick_marks = np.arange(len(self.categorical_labels))
      width = 0.75
      ax.bar(
          tick_marks,
          scores,
          width,
          color="orange",
          tick_label=self.categorical_labels,
          edgecolor="w",
          linewidth=1,
      )

      ax.set_xlabel("Targets")
      ax.set_xticks(tick_marks)
      ax.set_xticklabels(self.categorical_labels, rotation=-45, ha="center")
      ax.xaxis.set_label_position("bottom")
      ax.xaxis.tick_bottom()

      y_tick = np.linspace(0, 1, num=10)
      ax.set_ylabel(score_name)
      ax.set_yticks(y_tick)
      y_labels = [format(i, ".1f") for i in y_tick]
      ax.set_yticklabels(y_labels, ha="center")
      ax.set_title(title)

      fig.set_tight_layout(True)
      if save:
            self.save_fig(title,ext,fig,epoch)

    def visualize_embeddings_umap(self,
                                  epoch =0,
                                  title='',
                                  ext = "png",
                                  save =True,
                                  **umap_kwargs):

        #Init umapper
        umapper = umap.UMAP(**umap_kwargs) 
        # Compute umap embeddings
        umap_embeddings = umapper.fit_transform(self.embeddings)
        # Plot embeddings
        with sns.plotting_context(context="poster"):
            fig, ax = plt.subplots(1, figsize=(14, 10))
            plt.scatter(*umap_embeddings.T, s=0.8, c= self.true_labels, cmap= "tab20b", alpha=1)
            # plt.scatter(*umap_embeddings.T, s=1.5, c= self.true_labels, cmap='tab10', alpha=0.8)


            plt.setp(ax, xticks=[], yticks=[])
            cbar = plt.colorbar(boundaries=np.arange(self.n_classes+1)-0.5)
            cbar.set_ticks(np.arange(self.n_classes))
            cbar.set_ticklabels(self.categorical_labels)
            plt.title(title)

        if save:
            self.save_fig(title,ext,fig,epoch)
