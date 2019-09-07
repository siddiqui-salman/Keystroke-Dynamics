from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np


def roc_curve_population(genuine,impostor,sample):
    labels = [0] * len(genuine) + [1] * len(impostor)
    score = list(np.array(genuine['Score']))+list(np.array(impostor['Score']))
    fpr, tpr, threshold = roc_curve(labels, score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area =' + '{:.2f}'.format(roc_auc) + ') (Class)')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example for N= '+ '{}'.format(sample))
    plt.legend(loc="lower right")
    plt.show()


def det_curve_population(t_s,fpr_s, ipr_s,genuine,impostor,sample,n):
    if n == 1:
        labels = [0] * len(genuine) + [1] * len(impostor)
        score = list(np.array(genuine['Score'])) + list(np.array(impostor['Score']))
        fpr, tpr, threshold = roc_curve(labels, score)
        ipr = 1 - tpr
    else:
        fpr=fpr_s
        ipr=ipr_s
        threshold=t_s

    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(threshold, fpr, color='red', label='False Positive Rate')
    plt.plot(threshold, ipr, color='green', label='Impostor Pass Rate')
    plt.xlim([0.0, max(threshold)])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Error Rate')
    plt.xlabel('Threshold')
    plt.title('Detection Error Tradeoff for N= '+'{}'.format(sample))
    plt.legend(loc="lower right")
    plt.show()

