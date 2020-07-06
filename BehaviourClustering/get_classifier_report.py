import os

import matplotlib.pyplot as plt
import numpy as np
import umap
from cycler import cycler
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.notebook import tqdm

import torch
import argparse
from batch_preprocessing import prepare_batch_cnn, prepare_batch_embeddings
from BehaviourDatasets import TSDataset
from Evaluation import Evaluator
from MLP import MLP
from pytorchtools import load_pretrained_model
from torch.utils.data import DataLoader, Dataset
from TSResNet import tsresnet18, tsresnet_shallow
import yaml, pickle

parser = argparse.ArgumentParser()

parser.add_argument("--config_path", type=str,
                    help="File with config settings for testing")

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

    #set-up
    scaler=StandardScaler()
    encoder=LabelEncoder
    data_types=[True,True,True] 
    level = None
    n_components=9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_batch_fcn=prepare_batch_cnn

    #creating train and valid datasets
    train_dataset= TSDataset(dataset_folder,scaler,"Train",level,data_types)
    validation_dataset= TSDataset(dataset_folder,scaler,"Val",level,data_types)
    test_dataset= TSDataset(dataset_folder,scaler,"Test",level,data_types)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader= DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
    test_loader= DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    #wrap datasets & dataloaders
    splits ={
        "train" : {"dataset": train_dataset,"loader":train_loader},
        "test" : {"dataset": test_dataset,"loader":test_loader},
        "val" : {"dataset": validation_dataset,"loader":val_loader},
    }

    # Load Pretrained model
    models={}
    # Set trunk model 
    # trunk_arch=tsresnet18
    num_classes=train_dataset.num_classes
    trunk_arch = tsresnet18
    trunk_params={"num_classes":num_classes,"n_components":n_components}
    trunk_model, trunk_output_size  = trunk_arch(**trunk_params)
    trunk_model  = torch.nn.DataParallel(trunk_model)
    models["trunk"] = load_pretrained_model("trunk", model_folder, trunk_model, device)

    # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    embedder_model = torch.nn.DataParallel(MLP([trunk_output_size, embedding_dim]))
    models["embedder"] = load_pretrained_model("embedder",model_folder, embedder_model, device) 
    #Classifier
    classifier_model =  torch.nn.DataParallel(MLP([embedding_dim, num_classes]))
    models["classifier"] = load_pretrained_model("classifier",model_folder, classifier_model, device) 

    #Compute results
    classes = np.arange(len(train_dataset.classes))
    metrics_dict={ "Accuracy":balanced_accuracy_score,
                "Mcc":matthews_corrcoef,
                "Precision_Avg": [precision_score,{"average":"micro"}],
                "Recall_Avg" : [recall_score,{"average":"micro"}],
                "Precision_Class": [precision_score,{"labels":classes,"average":None}],
                "Recall_Class" : [recall_score,{"labels":classes,"average":None}],
                }

    for split in tqdm(splits.keys()):
        #compute confusion matrices
        split_folder = os.path.join(results_folder,split) 
        evaluator = Evaluator(models,
                            prepare_batch_cnn,
                            splits[split]["dataset"],
                            splits[split]["loader"],
                            split_folder,
                            device,
                            metrics_dict = metrics_dict)


        thresholds  = np.linspace(0.01,1,num = 15)
        #init metrics dict
        metrics = evaluator.init_metrics()
        for count,thr in enumerate(thresholds):
            true_labels, predicted_labels  = evaluator.filter_predictions(thr)
            evaluator.plot_confusion_matrix(true_labels,predicted_labels,title= "Confusion_Matrix",epoch=count)
            metrics = evaluator._update_test_metrics(true_labels,predicted_labels,metrics)
            #compute class Precision,Recall
            evaluator.plot_class_performance(metrics["Precision_Class"][count],"Precision",count,title= "Class_Precision")
            evaluator.plot_class_performance(metrics["Recall_Class"][count],"Recall",count,title= "Class_Recall")

        #save results
        table_file = os.path.join(split_folder,"metrics.pkl")
        with open(table_file,"wb") as f:
            pickle.dump(metrics,f)

