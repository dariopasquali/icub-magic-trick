import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats
pd.options.mode.chained_assignment = None  # default='warn'

# ========= MACHINE LEARNING ========================
from imblearn.over_sampling import SMOTE

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.utils.multiclass import unique_labels

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

def printTree(feature_cols,tree):
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('magic_tree.png')
    Image(graph.create_png())

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.get_cmap("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def calcMetrics(Yt, Yp, classes, model, norm=True, cols=None, print_metrics=False):
    
    if(len(classes) == 2):
        average = "binary"
    else:
        average = None
    
    # Metrics
    acc = metrics.accuracy_score(Yt, Yp)
    acc_bal = metrics.balanced_accuracy_score(Yt, Yp)
    prec = metrics.precision_score(Yt, Yp, average=average)
    recall = metrics.recall_score(Yt, Yp, average=average)
    f1 = metrics.f1_score(Yt, Yp, average=average)
    #roc_auc = metrics.roc_auc_score(Yt, Yp, average=average)
    roc_auc = np.NaN
    
    #feature_imp = pd.Series(model.feature_importances_,index=cols).sort_values(ascending=True)
    feature_imp = None
    
    if(print_metrics):
        print("")
        plot_confusion_matrix(Yt, Yp, classes, normalize=norm)
        print("")
        print("Accuracy:", acc)
        print("Balanced Accuracy: ", acc_bal)
        print("Precision: ", prec)
        print("Recall: ", recall)
        print("F1 Score: ", f1)
        print("ROC AuC Score: ", roc_auc)
       
        #fig, ax = plt.subplots(figsize=(10, 10))
        #ax.barh(feature_imp.index,feature_imp.values, align='center')
        #ax.set_xlabel('Performance')
        #ax.set_title('How fast do you want to go today?')
        #plt.grid(True)
    
    return acc, acc_bal, prec, recall, f1, roc_auc, feature_imp

class ModelMetricAggregator:
    
    def __init__(self, model, mode=''):
        self.model = model
        self.mode = mode
        self.accuracy = []
        self.balanced_accuracy = []
        self.precision = []
        self.recall = []
        self.f1_score = []
        self.roc_auc = []
        
    def addMetrics(self,acc, bal_acc, prec, rec, f1, roc_auc):
        self.accuracy.append(acc)
        self.balanced_accuracy.append(bal_acc)
        self.precision.append(prec)
        self.recall.append(rec)
        self.f1_score.append(f1)
        self.roc_auc.append(roc_auc)
        
    def getMetrics(self):
        return [
            self.model,
            self.mode,
            np.mean(self.accuracy),
            np.mean(self.balanced_accuracy),
            np.mean(self.precision),
            np.mean(self.recall),
            np.mean(self.f1_score),
            np.mean(self.roc_auc)
        ]