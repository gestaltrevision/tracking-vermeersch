import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from Training import Trainer
import os 

def plot_confusion_matrix(cm, classes,
                          exp_path,
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
    plt.savefig(os.path.join(exp_path,'books_read.png'))


class Evaluator(Trainer):

    def __init__(self,model,criterion,dataset,data_loader,prepare_batch,exp_path):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=model.to(self.device)
        self.criterion=criterion
        self.dataset=dataset
        self.data_loader=data_loader
        self.prepare_batch=prepare_batch
        self.exp_path=exp_path

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

    def plt_conf_matrix(self,title="",normalize=True):
        true_labels,predicted_labels = self._get_set_predictions()
        labels_classes=self.dataset.classes
        #Compute Confusion Matrix
        cnf_matrix = confusion_matrix(true_labels, 
                                    predicted_labels,labels=range(len(labels_classes)))
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plot_confusion_matrix(cnf_matrix,labels_classes,self.exp_path,
                        title=title,normalize=normalize) 
