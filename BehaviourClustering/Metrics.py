from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import interp
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from Evaluation import Evaluator

def plot_confusion_matrix(cm, classes,
                          results_folder,
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
        plt.savefig(os.path.join(results_folder,fig_file))

class Results_plotter(Evaluator):
    def __init__(self,
                true_labels,
                predicted_labels,
                probabilities_pred,
                predicted_best_probabilities,
                results_folder,
                dataset,
                coverages = None ,
                conf_thresh_arr = None,
                embeddings = None):

        self.true_labels = true_labels
        self.predicted_probabilities = probabilities_pred
        self.predicted_labels = predicted_labels
        self.scores = predicted_best_probabilities
        self.coverages = coverages
        self.conf_thresh_arr = conf_thresh_arr
        self.results_folder = results_folder
        self.dataset = dataset
        self.embeddings = embeddings
        self.categorical_labels = self.dataset.classes
        self.n_classes = len(self.categorical_labels)


    def plot_coverage(self, title = " ",ext="png",save = True):
        plt.plot(self.conf_thresh_arr, self.coverages)
        plt.xlabel("confidence threshold")
        plt.ylabel("coverage")
        plt.title(title)
        fig_file= ".".join([title,ext])
        if save:
            plt.savefig(os.path.join(self.results_folder,fig_file))
        plt.show()

    def plot_preds_distributions(self,label,title,save = True, ext= "png"):
        idx = np.squeeze(np.argwhere(self.predicted_labels == label))
        filtered_probs = self.predicted_probabilities[idx]

        for index, label in enumerate(self.categorical_labels):
            plt.hist(filtered_probs[:,index],label = label,alpha = 0.8)

        plt.legend()
        plt.title(title)

        fig_file= ".".join([title,ext])
        if save:
            plt.savefig(os.path.join(self.results_folder,fig_file))
        plt.show()

    def plt_conf_matrix(self, title = "" , normalize =True, save = True):
        #Compute confusion matrix
        cnf_matrix = confusion_matrix(self.true_labels, 
                                        self.predicted_labels,labels=range(self.n_classes))
        np.set_printoptions(precision=2)

        # Plot confusion matrix
        plot_confusion_matrix(cnf_matrix,
                                self.categorical_labels,
                                self.results_folder,save,
                                title=title,
                                normalize=normalize) 

    def get_roc_scores(self):
        labels = label_binarize(self.true_labels, classes=range(self.n_classes))
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        labels = self.true_labels.copy()
        scores = self.scores.copy()

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

    def plot_roc_curves(self, title = "", ext= "png", save = True):
        #get roc scores
        fpr,tpr,roc_auc = self.get_roc_scores()

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
                    ''.format(self.categorical_labels[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        fig_file= ".".join([title,ext])
        if save:
            plt.savefig(os.path.join(self.results_folder,fig_file))
        plt.show()



    def visualize_embeddings_umap(self,
                                  title='',
                                  ext = "png",
                                  save =True,
                                  **umap_kwargs):
        #Init umapper
        umapper = umap.UMAP(**umap_kwargs) 
        # Compute umap embeddings
        umap_embeddings = umapper.fit_transform(self.embeddings)
        # Plot embeddings
        sns.set(style='white', context='poster')
        _, ax = plt.subplots(1, figsize=(14, 10))
        # plt.scatter(*umap_embeddings.T, s=0.8, c= self.true_labels, cmap='Spectral', alpha=1)
        plt.scatter(*umap_embeddings.T, s=1.5, c= self.true_labels, cmap='tab10', alpha=0.8)


        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(self.n_classes+1)-0.5)
        cbar.set_ticks(np.arange(10))
        cbar.set_ticklabels(self.categorical_labels)
        plt.title(title)
        # Save
        fig_file= ".".join([title,ext])
        if save:
            plt.savefig(os.path.join(self.results_folder,fig_file))
        plt.show()

