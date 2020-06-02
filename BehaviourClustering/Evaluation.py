import os

import numpy as np
import torch
import torch.nn.functional as F

from Training_modular import Trainer
from pytorchtools import to_numpy
from sklearn.metrics import classification_report

class Evaluator(Trainer):

    def __init__(self,
                models,
                prepare_batch,
                dataset,
                data_loader,
                exp_path,
                device=None,
                metrics_dict= None):
                
        self.models = models
        self.prepare_batch=prepare_batch
        self.device = device
        self.dataset=dataset
        self.data_loader=data_loader
        self.exp_path=exp_path
        self.n_classes=dataset.num_classes
        self.metrics_dict = metrics_dict
        self.true_labels = None
        self.predicted_best_probabilities = None
        self.predicted_labels = None
        self.initialize_device()
        self.initialize_metric_dict()

    def is_defined(self,argument):
        if(getattr(self,argument)==None):
            raise ValueError
    def init_metrics(self):
        #init metrics
        metrics={}
        for key in self.metrics_dict.keys():
            metrics[key] = 0
        return metrics 
    def _normalize_probabilities(self,preds):
        """Network outputs (Unnormalized class probabilities)
        to normalized probabilities"""
        # preds=preds/self.ratios #thresholding
        return F.softmax(preds,dim=1)
    
    def get_valid_ids(self,probabilities,conf_thresh):
        valid_idx = [idx for idx,prob in enumerate(probabilities)
                        if np.max(prob) > conf_thresh]
        return valid_idx

    def compute_coverage(self,probabilities,conf_thresh):
        valid_idx = self.get_valid_ids(probabilities,conf_thresh)
        coverage = len(valid_idx)/ len(probabilities)
        return coverage

    def _logits_to_predictions(self,logits,labels):
        """"From model outputs (Unnormalized probabilities,Tensors) 
            to predictions(Classes,np.array)"""
        probabilities = self._logit_to_probabilies(logits)
        #Filtering predicted labels with model confidence below threshold
        labels_pred = torch.argmax(probabilities,dim=1)
        best_probabilities = [to_numpy(probabilities[batch_id,best_class])
                                     for batch_id,best_class in enumerate(labels_pred)]

        labels_pred = to_numpy(labels_pred)
        labels = to_numpy(labels)
        return best_probabilities, labels_pred, labels

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
        predicted_best_probabilities = []

        for batch in self.data_loader:
            logits,labels = self.inference(batch)
            best_probabilities, labels_pred, labels = self._logits_to_predictions(logits,labels)
            true_labels.append(labels)
            predicted_labels.append(labels_pred)
            predicted_best_probabilities.append(best_probabilities)

        self.true_labels = np.concatenate(true_labels)
        self.predicted_labels = np.concatenate(predicted_labels)
        self.predicted_best_probabilities = np.concatenate(predicted_best_probabilities)


    def compute_coverage_array(self):
        try:
            self.is_defined("predicted_labels")
        except:
            self._get_set_predictions()

        conf_thresh_arr = np.linspace(0.5,0.95,10)
        coverage_arr = np.empty(len(conf_thresh_arr))
        for idx,conf_thresh in enumerate(conf_thresh_arr): 
            coverage_arr[idx] = self.compute_coverage(self.predicted_best_probabilities,conf_thresh)
        return coverage_arr, conf_thresh_arr


    def filter_predictions(self,conf_thresh):
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

# if __name__ == "__main__":
#     import os 
#     import numpy as np
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F

#     from torch.utils.data import Dataset,DataLoader
#     from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
#     from BehaviourDatasets import TSDataset,TSDatasetSiamese
#     import joblib
#     # from tqdm.notebook import tqdm
#     from sklearn.preprocessing import OneHotEncoder,LabelEncoder

#     from sklearn.preprocessing import RobustScaler,StandardScaler
#     from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,matthews_corrcoef,confusion_matrix
    # Load the TensorBoard notebook extension
    # scaler=StandardScaler()
    # encoder=LabelEncoder
    # folder="/gdrive/My Drive/Datasets/HAR_Dataset_raw" #Dataset without label grouping

    # data_types=[True,True,True] #dont select gaze
    # level="AG"
    # n_components=9
    # batch_size=128
    # #creating train and valid datasets
    # train_dataset= TSDataset(folder,scaler,"Train",level,data_types)
    # validation_dataset= TSDataset(folder,scaler,"Val",level,data_types)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    # val_loader= DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
    # data_loaders=[train_loader,val_loader]
    # def load_pretrained_model(model_name, model_dir, model, device):
    #     checkpoint_path=os.path.join(model_dir,f"checkpoint_{model_name}.pth")
    #     model.load_state_dict(torch.load(checkpoint_path))
    #     return model.to(device)
    # from TSResNet import tsresnet_shallow,tsresnet18,tsresnet34
    # from batch_preprocessing import prepare_batch_siamese_cnn,prepare_batch_cnn,prepare_batch_rnn
    # from torch.optim.lr_scheduler import ReduceLROnPlateau

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_classes=train_dataset.num_classes
    # prepare_batch_fcn=prepare_batch_cnn
    # embedding_dim  = 64
    # exp_path="/gdrive/My Drive/BaselineResults/Basic/AG"

    # models={}
    # # Set trunk model 
    # trunk_arch=tsresnet_shallow
    # trunk_params={"num_classes":num_classes,"n_components":n_components}
    # trunk_model, trunk_output_size  = trunk_arch(**trunk_params)
    # models["trunk"] = load_pretrained_model("trunk", exp_path, trunk_model, device)

    # # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    # embedder_model = MLP([trunk_output_size, embedding_dim])
    # models["embedder"] = load_pretrained_model("embedder",exp_path, embedder_model, device) 

    # #Set classifier model
    # classifier_model = MLP([embedding_dim, num_classes])
    # models["classifier"] = load_pretrained_model("classifier", exp_path, classifier_model,
    #                                             device)

