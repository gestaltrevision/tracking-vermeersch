from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import interp
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Evaluation import Evaluator

class Results_plotter(Evaluator):
    def __init__(self,
                true_labels,
                predicted_labels,
                results_folder,
                dataset,
                probabilities_pred = None,
                predicted_best_probabilities = None,
                coverages = None ,
                conf_thresh_arr = None,
                embeddings = None):

        self.true_labels = true_labels
        self.predicted_probabilities = probabilities_pred
        self.predicted_labels = predicted_labels
        self.scores = probabilities_pred
        self.coverages = coverages
        self.conf_thresh_arr = conf_thresh_arr
        self.results_folder = results_folder
        self.dataset = dataset
        self.embeddings = embeddings
        self.categorical_labels = self.dataset.classes
        self.n_classes = len(self.categorical_labels)

    def initialize_writer(self,writer_folder):
      if not os.path.exists(writer_folder):
          os.makedirs(writer_folder)
      return SummaryWriter(writer_folder)

    def to_tensorboard(self,base_dir,title,fig,epoch):
      writer = self.initialize_writer(base_dir)
      writer.add_figure(tag= title,figure = fig, global_step = epoch)

    def save_fig(self,title,ext,fig,epoch):
        # Save
        base_dir  = os.path.join(self.results_folder,title)
        fig_file= ".".join([title,ext])
        fig_dir = os.path.join(base_dir,fig_file)
        plt.savefig(fig_dir)
        self.to_tensorboard(base_dir,title,fig,epoch)
        plt.show()

    def plot_coverage(self, title = " ",ext="png",save = True):
        plt.plot(self.conf_thresh_arr, self.coverages)
        plt.xlabel("confidence threshold")
        plt.ylabel("coverage")
        plt.title(title)

        if save:
            self.save_fig(title,ext,fig)

    def plot_preds_distributions(self,label,title,save = True, ext= "png"):
        idx = np.squeeze(np.argwhere(self.predicted_labels == label))
        filtered_probs = self.predicted_probabilities[idx]

        for index, label in enumerate(self.categorical_labels):
            plt.hist(filtered_probs[:,index],label = label,alpha = 0.8)

        plt.legend()
        plt.title(title)

        if save:
            self.save_fig(title,ext,fig)

    def plot_confusion_matrix(self, epoch =0, title="", ext = "png", save = True, normalize = True):
        try:
            cnf_matrix = confusion_matrix(self.true_labels, 
                                            self.predicted_labels,labels=range(self.n_classes))
            if (normalize):
              cnf_matrix= cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
              cnf_matrix  = np.around(cnf_matrix,2)
              
        except:
            cnf_matrix  = np.zeros((self.n_classes,self.n_classes))
            print("Not enough samples to compute confusion matrix")
       
        df = pd.DataFrame(cnf_matrix,self.categorical_labels,self.categorical_labels)
        fig = plt.figure(figsize=(15,10),dpi = 125)

        sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
        sns.heatmap(df,cmap ="Oranges", annot=True) 
        plt.title(title)
        if save:
            self.save_fig(title,ext,fig,epoch)

        
    def plot_pr_curves(self,base_dir):
        labels = label_binarize(self.true_labels, classes=range(self.n_classes))
        scores = self.scores.copy()
        #save pr_curve for each class
        writer = self.initialize_writer(base_dir)
        for i in range(self.n_classes):
            curve_title = "pr_curve_{}".format(self.categorical_labels[i])
            writer.add_pr_curve(curve_title, labels[:,i], scores[:,i])

    def get_roc_scores(self):
        labels = label_binarize(self.true_labels, classes=range(self.n_classes))
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        # labels = self.true_labels.copy()
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
        if save:
            self.save_fig(title,ext,fig)



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
        # with sns.set(style='white', context='poster'):
        with sns.plotting_context(context="poster"):
            _, ax = plt.subplots(1, figsize=(14, 10))
            plt.scatter(*umap_embeddings.T, s=0.8, c= self.true_labels, cmap= "tab20b", alpha=1)
            # plt.scatter(*umap_embeddings.T, s=1.5, c= self.true_labels, cmap='tab10', alpha=0.8)


            plt.setp(ax, xticks=[], yticks=[])
            cbar = plt.colorbar(boundaries=np.arange(self.n_classes+1)-0.5)
            cbar.set_ticks(np.arange(self.n_classes))
            cbar.set_ticklabels(self.categorical_labels)
            plt.title(title)

        if save:
            self.save_fig(title,ext,fig)