import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from numpy import interp
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from Training_modular import Trainer
from pytorchtools import to_numpy

def plot_confusion_matrix(cm, classes,
                          exp_path,
                          save,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          ext="png"):
                         
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    eg. usage
    cnf_matrix = confusion_matrix(labels, labels_pred,labels=range(len(train_dataset.classes)))
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=train_dataset.classes,
                      title='Normalized Confusion Matrix',normalize=True) 
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm) #Raw Matrix 

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=75)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    fig_file= ".".join([title,ext])
    if save:
        plt.savefig(os.path.join(exp_path,fig_file))

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
        valid_idx = self.get_valid_ids(self,probabilities,conf_thresh)
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

    def plot_coverage(self,title = " "):
        coverage_arr, conf_thresh_arr = self.compute_coverage_array()
        plt.plot(conf_thresh_arr, coverage_arr)
        plt.xlabel("confidence threshold")
        plt.ylabel("coverage")
        plt.title(title)
        plt.show()

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
        metrics = self.init_metrics()
        metrics = self._update_metrics(true_labels_filtered,pred_labels_filtered,metrics)
        return metrics

    def _get_set_scores(self):
        true_labels=[]
        predicted_scores=[]
        for batch in self.data_loader:
            logits,labels=self.inference(batch)
            # scores=self._normalize_probabilities(preds).to("cpu").numpy().astype(np.float32)
            scores = to_numpy(self._normalize_probabilities(logits))
            #from torch tensors to numpy arrays
            labels = to_numpy(labels)
            #from integer encodings to binary encodings
            labels = label_binarize(labels, classes=range(self.n_classes))
            true_labels.append(labels)
            predicted_scores.append(scores)

        true_labels=np.concatenate(true_labels)
        predicted_scores=np.concatenate(predicted_scores)

        return true_labels,predicted_scores

    def plt_conf_matrix(self,title="",normalize=True,save=True):
        true_labels,predicted_labels = self._get_set_predictions()
        labels_classes=self.dataset.classes
        #Compute Confusion Matrix
        cnf_matrix = confusion_matrix(true_labels, 
                                    predicted_labels,labels=range(len(labels_classes)))
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plot_confusion_matrix(cnf_matrix,labels_classes,self.exp_path,save,
                        title=title,normalize=normalize) 

    def get_roc_scores(self):
      #get labels,scores
      labels,scores=self._get_set_scores()
      # Compute ROC curve and ROC area for each class
      fpr = {}
      tpr = {}
      roc_auc = {}
      #compute roc for each class
      for i in range(self.n_classes):
          fpr[i], tpr[i], _ = roc_curve(labels[:, i], scores[:, i])
          roc_auc[i] = auc(fpr[i], tpr[i])

      # Compute micro-average ROC curve and ROC area
      fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), scores.ravel())
      roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

      # First aggregate all false positive rates
      all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))

      # Then interpolate all ROC curves at this points
      mean_tpr = np.zeros_like(all_fpr)
      for i in range(self.n_classes):
          mean_tpr += interp(all_fpr, fpr[i], tpr[i])

      # Finally average it and compute AUC
      mean_tpr /= self.n_classes
      fpr["macro"] = all_fpr
      tpr["macro"] = mean_tpr
      roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

      return fpr,tpr,roc_auc

    def plot_roc_curves(self):
      #get roc scores
      fpr,tpr,roc_auc=self.get_roc_scores()
      # Plot all ROC curves
      plt.figure()
      lw = 2
      plt.plot(fpr["micro"], tpr["micro"],
              label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
              color='deeppink', linestyle=':', linewidth=4)

      plt.plot(fpr["macro"], tpr["macro"],
              label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
              color='navy', linestyle=':', linewidth=4)

      colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

      for i, color in zip(range(self.n_classes), colors):
          plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                  label='ROC curve of class {0} (area = {1:0.2f})'
                  ''.format(self.dataset.classes[i], roc_auc[i]))

      plt.plot([0, 1], [0, 1], 'k--', lw=lw)
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Some extension of Receiver operating characteristic to multi-class')
      plt.legend(loc="lower right")
      plt.show()



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


    evaluator_val = Evaluator(None,
                            None,
                            None,
                            None,
                            None,
                            None)
    logits = torch.rand([10,3]).to(device)
    labels = torch.rand([10,1]).to(device)
    evaluator_val._logits_to_predictions(logits,labels)
    pass
