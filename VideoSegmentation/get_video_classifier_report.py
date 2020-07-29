import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml
from cycler import cycler
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import torch
from Evaluation_video import Evaluator_video
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--config_path", type=str,
                    help="File with config settings for testing")

def get_file_stem(path):
    base=os.path.basename(path)
    return os.path.splitext(base)[0]

def get_split_report(evaluator):
    thresholds  = np.linspace(0.1,0.9,num = 10)    
    #init metrics dict
    metrics = evaluator.init_metrics()
    for count,thr in enumerate(thresholds):
        true_labels, predicted_labels  = evaluator.filter_predictions(thr)
        evaluator.plot_confusion_matrix(true_labels,predicted_labels,title= "Confusion_Matrix",epoch=count)
        metrics = evaluator._update_test_metrics(true_labels,predicted_labels,metrics)
        #compute class Precision,Recall
        evaluator.plot_class_performance(metrics["Precision_Class"][count],"Precision",count,title= "Class_Precision")
        evaluator.plot_class_performance(metrics["Recall_Class"][count],"Recall",count,title= "Class_Recall")
        #save only avg metrics
        avg_metrics = {metric: value for metric,value in metrics.items() if not("Class" in metric)}
    return avg_metrics

def save_results(metrics,split_folder):
    table_file = os.path.join(split_folder,"metrics.csv")
    metrics_df  = pd.DataFrame.from_dict(metrics)
    metrics_df.to_csv(table_file,index=False)


if __name__ == "__main__":

    args=parser.parse_args()
    # load model config
    config_path = args.config_path
    with open(config_path,"rb") as f:
        config = yaml.safe_load(f)

    model_folder = config["model_folder"]
    dataset_folder = config["dataset_folder"]
    results_folder  = config["results_folder"]
    batch_size = config["batch_size"]
    # trunk_arch  = config["trunk_arch"]
    embedding_dim = config["embedding_dim"]
    #create results folder if its not yet been created
    if not(os.path.isdir(results_folder)):
        os.makedirs(results_folder)

   
    #wrap datasets & dataloaders
    splits ={
        "train" : {"dataset": train_dataset,"loader":train_loader},
        "test" : {"dataset": test_dataset,"loader":test_loader},
        "val" : {"dataset": validation_dataset,"loader":val_loader},
    }

   
    #Compute results
    classes = np.arange(len(train_dataset.classes))
    metrics_dict={ "Accuracy":balanced_accuracy_score,
                "Mcc":matthews_corrcoef,
                "Precision_Avg": [precision_score,{"average":"micro"}],
                "Recall_Avg" : [recall_score,{"average":"micro"}],
                "Precision_Class": [precision_score,{"labels":classes,"average":None}],
                "Recall_Class" : [recall_score,{"labels":classes,"average":None}],
                }

    #set-up evaluator
    split_folder = os.path.join(results_folder,split) 
    evaluator = Evaluator(models,
                        prepare_batch_cnn,
                        splits[split]["dataset"],
                        splits[split]["loader"],
                        split_folder,
                        device,
                        metrics_dict = metrics_dict)
    #compute report
    report = get_split_report(evaluator,split)
    #save report
    save_results(report,split_folder)
    print(f"Correctly process split {split}")

