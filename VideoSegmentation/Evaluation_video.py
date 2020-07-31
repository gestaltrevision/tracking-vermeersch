import os
import warnings
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

class Evaluator_video():

    def __init__(self,     
                results_folder,
                encoder ,
                true_labels,
                predicted_probabilities,
                metrics_dict= None):

        self.true_labels  = true_labels
        self.predicted_probabilities = predicted_probabilities
        self.predicted_labels = None
        self.predicted_best_probabilities = None
        self.metrics_dict = metrics_dict
        self.results_folder = results_folder
        self.categorical_labels = encoder.classes_
        self.n_classes = len(self.categorical_labels)
        self.initialize_metric_dict()

    def is_defined(self,argument):
        if(getattr(self,argument)==None):
            raise ValueError
        
    def initialize_metric_dict(self):
        if self.metrics_dict is None:
            self.metrics_dict = {}

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


    
    def get_valid_ids(self,probabilities,conf_thresh):
        valid_idx = [idx for idx,prob in enumerate(probabilities)
                        if np.max(prob) > conf_thresh]
        return valid_idx

    def compute_coverage(self,filtered_labels):
        return len(filtered_labels)/len(self.true_labels)


    def get_best_probabilities(self):
      best_probabilities = [self.predicted_probabilities[batch_id,best_class]
                                  for batch_id,best_class in enumerate(self.true_labels)]

      self.predicted_labels = np.argmax(self.predicted_probabilities,axis = 1)
      self.predicted_best_probabilities  = best_probabilities



    def filter_predictions(self,conf_thresh):
        try:
            self.is_defined("predicted_best_probabilities")
        except:
            self.get_best_probabilities()
        valid_idx = self.get_valid_ids(self.predicted_best_probabilities,conf_thresh)
        true_labels_filtered = self.true_labels[valid_idx]
        pred_labels_filtered = self.predicted_labels[valid_idx]

        return true_labels_filtered, pred_labels_filtered

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

      # fig.set_tight_layout(True)
      if save:
            self.save_fig(title,ext,fig,epoch)

  