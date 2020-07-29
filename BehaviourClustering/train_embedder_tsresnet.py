import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
import torch
import yaml
from ax import load
from cycler import cycler
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import common_functions
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import umap
from batch_preprocessing import prepare_batch_cnn, prepare_batch_embeddings
from BehaviourDatasets import TSDataset
from Evaluation import Evaluator
from MLP import MLP
from pytorchtools import load_pretrained_model
from Training_modular import Trainer
from TSResNet import tsresnet18, tsresnet_shallow

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
    logging.getLogger().setLevel(logging.INFO)
    logging.info("VERSION %s"%pytorch_metric_learning.__version__)
    args=parser.parse_args()
    # load model config
    config_path = args.config_path
    with open(config_path,"rb") as f:
        config = yaml.safe_load(f)

    model_folder = config["model_folder"]
    train_base_dir = config["train_base_dir"]
    train_base_root = config["train_base_root"]
    dataset_folder_val = config["val_folder"]
    data_subsets = config["data_subsets"]
    output_folder  = config["output_folder"]
    batch_size = config["batch_size"]
    embedding_dim = config["embedding_dim"]
    #create dataset folders list
    if(config["ActiveLearning"]):
        data_folders  = [os.path.join(train_base_dir,f"{train_base_root}_{subset}")
                            for subset in data_subsets]
        data_folders.append(config["base_folder"])
    else:
        data_folders  = config["base_folder"]
        
    #create results folder if its not yet been created
    if not(os.path.isdir(output_folder)):
        os.makedirs(output_folder)
    #LOAD TRAINING CONFIG
    params_opt  = fetch_hyperparameters(config["optim_file"])
    params_loss = fetch_hyperparameters(config["loss_optim_file"])
    #set-up
    scaler=StandardScaler()
    encoder=LabelEncoder
    data_types=[True,True,True] 
    level = None
    n_components=9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_batch_fcn=prepare_batch_cnn

    #creating train and valid datasets
    train_dataset= TSDataset(data_folders,scaler,"Train",level,data_types,
                            config["ActiveLearning"],config["base_folder"])

    validation_dataset= TSDataset(dataset_folder_val,scaler,"Val",level,data_types,
                                    config["ActiveLearning"],config["base_folder"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader= DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
    data_loaders = [train_loader,val_loader]
   
    # Load Pretrained model
    models={}
    # Set trunk model  
    num_classes = train_dataset.num_classes
    trunk_arch = tsresnet18
    trunk_params={"num_classes":num_classes,"n_components":n_components}
    trunk_model, trunk_output_size  = trunk_arch(**trunk_params)
    # Set embedder model
    embedder_model = MLP([trunk_output_size, embedding_dim])

    if(config["pretrain"]==True):
       trunk = load_pretrained_model("trunk", model_folder, trunk_model, device)
       embedder = load_pretrained_model("embedder", model_folder, embedder_model, device)
    else:
        trunk = torch.nn.DataParallel(trunk_model)
        embedder = torch.nn.DataParallel(embedder_model)
    #override opt parameter for Active Learning case
    if(config["ActiveLearning"]):
        params_opt = {"lr":3e-3,"T_0":1,"T_mult":2}
    
    #set optimizers
    trunk_optimizer = torch.optim.SGD(trunk.parameters(), lr= params_opt["lr"] , momentum= 0.9,weight_decay= 5e-5)
    embedder_optimizer = torch.optim.SGD(embedder.parameters(), lr= params_opt["lr"] , momentum= 0.9, weight_decay= 5e-5)
    #set schedulers
    scheduler_trunk = CosineAnnealingWarmRestarts(trunk_optimizer,params_opt["T_0"],params_opt["T_mult"])
    scheduler_embedder = CosineAnnealingWarmRestarts(embedder_optimizer,params_opt["T_0"],params_opt["T_mult"])
    # Set the loss function
    loss = losses.ContrastiveLoss(pos_margin= params_loss["pos_margin"], neg_margin = params_loss["neg_margin"])
    # Set the mining function
    miner = miners.MultiSimilarityMiner(epsilon= params_loss["miner_epsilon"])
    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=len(train_dataset))

    # Wrap
    models = {"trunk": trunk, "embedder": embedder}
    optimizers_list = [trunk_optimizer,embedder_optimizer]
    scheduler_list  = [scheduler_trunk,scheduler_embedder]
    optimizers = {f"{model}_optimizer": optimizer for model,optimizer 
                                                in zip(models.keys(),optimizers_list) }
                                              
    
    lr_schedulers = {f"{model}_scheduler_by_iteration": scheduler for model,scheduler,
                                                    in zip(models.keys(),scheduler_list) }
                                                 
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}

    # train 
    record_keeper, _, _ = logging_presets.get_record_keeper(
                                                        os.path.join(output_folder,"logs"),
                                                        os.path.join(output_folder,"Tensordboard"))
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"val": validation_dataset}

    tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook = hooks.end_of_testing_hook, 
                                                dataloader_num_workers = 32,
                                                data_and_label_getter = prepare_batch_embeddings)

    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, 
                                                dataset_dict, 
                                                output_folder, 
                                                test_interval = 10,
                                                patience = config["patience"])
    trainer = trainers.MetricLossOnly(models,
                                  optimizers,
                                  batch_size,
                                  loss_funcs,
                                  mining_funcs,
                                  train_dataset,
                                  sampler=sampler,
                                  lr_schedulers= lr_schedulers,
                                  dataloader_num_workers = 32,
                                  end_of_iteration_hook = hooks.end_of_iteration_hook,
                                  end_of_epoch_hook = end_of_epoch_hook,
                                  data_and_label_getter = prepare_batch_embeddings)

    trainer.train(num_epochs = config["n_epochs"])
