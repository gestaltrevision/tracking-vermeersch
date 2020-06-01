from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import interp
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

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

def plot_coverage(coverage_arr,
                    conf_thresh_arr,
                    exp_path,
                    title = " ",
                    ext="png",
                    save = True):
    plt.plot(conf_thresh_arr, coverage_arr)
    plt.xlabel("confidence threshold")
    plt.ylabel("coverage")
    plt.title(title)
    fig_file= ".".join([title,ext])
    if save:
        plt.savefig(os.path.join(exp_path,fig_file))
    plt.show()

def plt_conf_matrix(labels,
                    predicted_labels,
                    dataset,
                    exp_path,
                    title="",
                    normalize=True,
                    save=True):
    labels_classes=dataset.classes
    #Compute Confusion Matrix
    cnf_matrix = confusion_matrix(labels, 
                                    predicted_labels,labels=range(len(labels_classes)))
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix,labels_classes,exp_path,save,
                    title=title,normalize=normalize) 

def get_roc_scores(labels,scores,n_classes):
    labels = label_binarize(labels, classes=range(n_classes))
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    #compute roc for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr,tpr,roc_auc

def plot_roc_curves(labels,
                    scores,
                    n_classes,
                    dataset,
                    exp_path,
                    title ="",
                    ext="png",
                    save =True):
    #get roc scores
    fpr,tpr,roc_auc = get_roc_scores(labels,scores,n_classes)

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

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(dataset.classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    fig_file= ".".join([title,ext])
    if save:
        plt.savefig(os.path.join(exp_path,fig_file))
    plt.show()
