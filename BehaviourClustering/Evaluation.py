import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from Training import Trainer
import os 

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

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    fig_file= ".".join([title,ext])
    if save:
        plt.savefig(os.path.join(exp_path,fig_file))

import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from numpy import interp
from sklearn.metrics import roc_auc_score

class Evaluator(Trainer):

    def __init__(self,model,criterion,dataset,data_loader,prepare_batch,exp_path,ratios):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=model.to(self.device)
        self.criterion=criterion
        self.dataset=dataset
        self.data_loader=data_loader
        self.prepare_batch=prepare_batch
        self.exp_path=exp_path
        self.n_classes=dataset.num_classes
        self.ratios=torch.from_numpy(ratios).to(self.device)


    def _normalize_probabilities(self,preds):
        """Network outputs (Unnormalized class probabilities)
        to normalized probabilities"""
        # preds=preds/self.ratios #thresholding
        return F.softmax(preds,dim=1)

    def _prob_to_predictions(self,preds,labels):
        """"From model outputs (Unnormalized probabilities,Tensors) 
            to predictions(Classes,np.array)"""
        probabilities=self._normalize_probabilities(preds)

        labels_pred=torch.argmax(probabilities,dim=1)
        labels_pred=labels_pred.to("cpu").numpy().astype(np.int32)
        labels=labels.to("cpu").numpy().astype(np.int32)
        return labels_pred,labels

    def _get_set_predictions(self):
        true_labels=[]
        predicted_labels=[]
        for batch in self.data_loader:
            preds,labels=self.inference(batch)
            labels_pred,labels=self._prob_to_predictions(preds,labels)
            true_labels.append(labels)
            predicted_labels.append(labels_pred)

        true_labels=np.concatenate(true_labels)
        predicted_labels=np.concatenate(predicted_labels)

        return true_labels,predicted_labels
      
    def _get_set_scores(self):
        true_labels=[]
        predicted_scores=[]
        for batch in self.data_loader:
            preds,labels=self.inference(batch)

            scores=self._normalize_probabilities(preds).to("cpu").numpy().astype(np.float32)
            #from torch tensors to numpy arrays
            labels=labels.to("cpu").numpy().astype(np.int32)
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

if __name__ == "__main__":
    from TSResNet import tsresnet_shallow,tsresnet18,tsresnet34
    from batch_preprocessing import prepare_batch_siamese_cnn,prepare_batch_cnn,prepare_batch_rnn
    import os 
    # import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from torch.utils.data import Dataset,DataLoader
    from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
    from BehaviourDatasets import TSDataset,TSDatasetSiamese
    import joblib
    from sklearn.preprocessing import OneHotEncoder,LabelEncoder

    from sklearn.preprocessing import RobustScaler,StandardScaler
    # from torch.utils.tensorboard import SummaryWriter
    from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,matthews_corrcoef,confusion_matrix
    
    scaler=StandardScaler()
    encoder=LabelEncoder
    folder=r"C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\AnnotatedData\Accelerometer_Data\Datasets\HAR_Dataset_RO" #Dataset Random Oversampling

    data_types=[True,True,True] #select all components
    level="AG"
    n_components=9
    batch_size=128
    #creating train and valid datasets
    train_dataset= TSDataset(folder,scaler,"Train",level,data_types)
    validation_dataset= TSDataset(folder,scaler,"Val",level,data_types)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader= DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
    data_loaders=[train_loader,val_loader]


    exp_path=r"C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\tracking-vermeersch\BehaviourClustering\run\Baseline_Resnet_RO_AG"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path=os.path.join(exp_path,"checkpoint.pth")
    num_classes=train_dataset.num_classes
    model_arch=tsresnet_shallow
    model_params={"num_classes":num_classes,"n_components":n_components}
    model =model_arch(**model_params)
    model.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cpu')))
    model=model.to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    prepare_batch_fcn=prepare_batch_cnn
    ratios=np.array(train_dataset.get_class_ratios())
    evaluator_val=Evaluator(model,criterion,validation_dataset,val_loader,prepare_batch_fcn ,exp_path,ratios) #val set
    evaluator_train=Evaluator(model,criterion,train_dataset,train_loader,prepare_batch_fcn,exp_path,ratios) #val set

    evaluator_train.plot_roc_curves()
    pass