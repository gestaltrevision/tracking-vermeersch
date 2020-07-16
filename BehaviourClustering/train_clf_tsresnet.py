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
import os

import matplotlib.pyplot as plt
import numpy as np
import umap
from cycler import cycler
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from Training_modular import Trainer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ax import load

def fetch_hyperparameters(file):
  experiment = load(file)
  data = experiment.fetch_data()
  df = data.df
  best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
  best_arm = experiment.arms_by_name[best_arm_name]
  return best_arm.parameters

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
    output_folder  = config["output_folder"]
    batch_size = config["batch_size"]
    embedding_dim = config["embedding_dim"]
    #create results folder if its not yet been created
    if not(os.path.isdir(output_folder)):
        os.makedirs(output_folder)
    #LOAD TRAINING CONFIG
    params_opt  = fetch_hyperparameters(config["optim_file"])
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader= DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
    data_loaders = [train_loader,val_loader]
   
    # Load Pretrained model
    models={}
    # Set trunk model  
    num_classes=train_dataset.num_classes
    trunk_arch = tsresnet18
    trunk_params={"num_classes":num_classes,"n_components":n_components}
    trunk_model, trunk_output_size  = trunk_arch(**trunk_params)
    # trunk  = torch.nn.DataParallel(trunk_model)
    # Set embedder model
    embedder_model = MLP([trunk_output_size, embedding_dim])
    # embedder = torch.nn.DataParallel(MLP([trunk_output_size, embedding_dim]))
    #Classifier
    classifier =  torch.nn.DataParallel(MLP([embedding_dim, num_classes]))

    if(config["pretrain"]==True):
       trunk = load_pretrained_model("trunk", model_folder, trunk_model, device)
       embedder = load_pretrained_model("embedder", model_folder, embedder_model, device)
    else:
        trunk = torch.nn.DataParallel(trunk_model)
        embedder = torch.nn.DataParallel(embedder_model)

    trunk_optimizer = torch.optim.SGD(trunk.parameters(), lr= params_opt["lr"] , momentum= 0.9,weight_decay= 5e-5)
    embedder_optimizer = torch.optim.SGD(embedder.parameters(), lr= params_opt["lr"] , momentum= 0.9, weight_decay= 5e-5)
    classifier_optimizer =  torch.optim.SGD(classifier.parameters(), lr= params_opt["lr"] , momentum= 0.9, weight_decay= 5e-5)
    #set schedulers
    scheduler_trunk = CosineAnnealingWarmRestarts(trunk_optimizer,params_opt["T_0"],params_opt["T_mult"])
    scheduler_embedder = CosineAnnealingWarmRestarts(embedder_optimizer,params_opt["T_0"],params_opt["T_mult"])
    scheduler_classifier = CosineAnnealingWarmRestarts(classifier_optimizer,params_opt["T_0"],params_opt["T_mult"])

    #wrap
    freeze_these  = config["freeze"]
    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}
    optimizers_list = [trunk_optimizer,embedder_optimizer,classifier_optimizer]
    scheduler_list  = [scheduler_trunk,scheduler_embedder,scheduler_classifier]
    optimizers = {f"{model}_optimizer": optimizer for model,optimizer 
                                                in zip(models.keys(),optimizers_list)
                                                if not(model in freeze_these)}
    
    lr_schedulers = {f"{model}_scheduler_by_iteration": scheduler for model,scheduler,
                                                    in zip(models.keys(),scheduler_list)
                                                    if not(model in freeze_these)}

    # Set the classification loss:
    class_ratios = train_dataset.get_class_ratios()
    class_weights = 1./class_ratios
    class_weights = torch.FloatTensor(class_weights).to(device)
    loss_weights = class_weights if config["balanced"] else None
    classification_loss = torch.nn.CrossEntropyLoss(weight=loss_weights)
    loss_funcs = {"classifier_loss": classification_loss}
    #train
    n_epochs=config["n_epochs"]
    patience = config["patience"]

    metrics_dict={ "Accuracy":balanced_accuracy_score,
                "Mcc":matthews_corrcoef
                }
    trainer=Trainer(models,
                    loss_funcs,
                    optimizers,
                    data_loaders,
                    prepare_batch_cnn,
                    device,
                    metrics_dict,
                    lr_schedulers= lr_schedulers,
                    freeze_these = freeze_these)
    #train
    trainer.train(n_epochs,output_folder,patience)
                

