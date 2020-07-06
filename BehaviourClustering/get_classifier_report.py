from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
import umap
from cycler import cycler
import record_keeper
import pytorch_metric_learning
logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)
import os 
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader
import joblib
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler,StandardScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,matthews_corrcoef,confusion_matrix
# Load the TensorBoard notebook extension
%load_ext tensorboard
from pytorchtools import load_pretrained_model
from TSResNet import tsresnet_shallow,tsresnet18
from MLP import MLP
from batch_preprocessing import prepare_batch_cnn, prepare_batch_embeddings
from torch.optim.lr_scheduler import ReduceLROnPlateau


from BehaviourDatasets import TSDataset

model_folder = "/gdrive/My Drive/Models/TsResnet18/Embedder_32/Embedder_500"
params = { "lr": 8.976686287282146e-05,
          "pos_margin": 0.1914028338086492,
          "neg_margin": 0.9654141233183021,
          "miner_epsilon": 0.1902556935094855,
          'dropout': 0.43426912426948544,
          'l2_embedder': 0.001295915082072328,
          'l2_trunk': 4.165466878607889e-06}

torch.manual_seed(12345)
scaler=StandardScaler()
encoder=LabelEncoder
embedding_dim  = 32
folder="/gdrive/My Drive/Datasets/HAR_Dataset_raw" #Dataset without label grouping

data_types=[True,True,True] 
level = None
n_components=9
batch_size=64
#creating train and valid datasets
train_dataset= TSDataset(folder,scaler,"Train",level,data_types)
validation_dataset= TSDataset(folder,scaler,"Val",level,data_types)
test_dataset= TSDataset(folder,scaler,"Test",level,data_types)

train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
val_loader= DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
test_loader= DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

data_loaders=[train_loader,val_loader]

# Generate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes=train_dataset.num_classes
prepare_batch_fcn=prepare_batch_cnn

exp_path = "/gdrive/My Drive/Models/TsResnet18/Embedder_32/Classifiers/Classifier_frozen"

models={}
# Set trunk model 
trunk_model, trunk_output_size  = trunk_arch(**trunk_params)
trunk_model  = torch.nn.DataParallel(trunk_model)
models["trunk"] = load_pretrained_model("trunk", exp_path, trunk_model, device)

# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
embedder_model = torch.nn.DataParallel(MLP([trunk_output_size, embedding_dim]))
models["embedder"] = load_pretrained_model("embedder",exp_path, embedder_model, device) 
#Classifier
classifier_model =  torch.nn.DataParallel(MLP([embedding_dim, num_classes]))
models["classifier"] = load_pretrained_model("classifier",exp_path, classifier_model, device) 
