import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats
pd.options.mode.chained_assignment = None  # default='warn'

from data_cleaning import *
from eye_feature_tools import *

# ==========================================================

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.model_selection import GridSearchCV

import sys
sys.stdout = open("ML_report/report_2.txt", "w")
print ("test sys.stdout")

# ==========================================================

def grid_search_hyp_tuning(model, param_map, scores, X_train, X_test, y_train, y_test, oversamp=True):

    #print("Resplit the Test set in Test and Validtion")
    #X_train, X_val, y_train, y_val = train_test_split(X_train_0, y_train_0, test_size=0.25,random_state=42, shuffle=False)

    if(oversamp):
        """
        for m in param_map:
            for key in m.keys():
                new_key = "clf__" + key
                m[new_key] = m.pop(key)
        """
        model = Pipeline([
                ('sampling', SMOTE()),
                ('clf', model)
            ])

    for score in scores:
        print("# =========================== Tuning hyper-parameters for %s" % score)
        print()

        sc = score
        #sc = '%s_macro' % score
        #if(oversamp):
        #    sc = 'clf__%s_macro' % score

        clf = GridSearchCV(model, param_map, cv=4, scoring=sc, verbose=1)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        #means = clf.cv_results_['mean_test_score']
        #stds = clf.cv_results_['std_test_score']
        #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print(">>>>>>> Test Set Report <<<<<<<<")
        print()
        print("The model is trained on the full Training set.")
        print("The scores are computed on the never revealed Test set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        print("Accuracy:",metrics.accuracy_score(y_true, y_pred))
        print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_true, y_pred))
        print("Precision:",metrics.precision_score(y_true, y_pred))
        print("Recall:",metrics.recall_score(y_true, y_pred))
        print("F1:",metrics.f1_score(y_true, y_pred))
        print("AUROC:",metrics.roc_auc_score(y_true, y_pred))
        print()

def grid_search_Decision_Tree(X_train, X_test, y_train, y_test, oversamp=True):

    scores = [
        'precision',
        'recall', 
        'f1',
        'balanced_accuracy',
        'roc_auc'
        ]

    param_map = {
        'criterion' : ['entropy', 'gini'],
        'splitter' : ['best', 'random'],
        'max_depth' : [2, 4, 6, 8, 10, 20],
        'min_samples_split' : [2, 3, 4, 5, 6],
        'min_samples_leaf' : [1, 2, 3, 4, 5],
        'max_features' : ['auto', 'sqrt']
    }

    if(oversamp):
        param_map = {
        'clf__criterion' : ['entropy', 'gini'],
        'clf__splitter' : ['best', 'random'],
        'clf__max_depth' : [2, 4, 6, 8, 10, 20],
        'clf__min_samples_split' : [2, 3, 4, 5, 6],
        'clf__min_samples_leaf' : [1, 2, 3, 4, 5],
        'clf__max_features' : ['auto', 'sqrt']
        }


    grid_search_hyp_tuning(DecisionTreeClassifier(), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp)

def grid_search_Random_Forest(X_train, X_test, y_train, y_test, oversamp=True):

    scores = [
        'precision',
        'recall', 
        'f1',
        'balanced_accuracy',
        'roc_auc'
        ]

    param_map = {
        'n_estimators' : [10, 80, 100, 800],
        'bootstrap' : [True, False],
        'max_depth' : [2, 8, 10, 20, 50],
        'min_samples_split' : [2, 3, 4, 5, 6],
        'min_samples_leaf' : [1, 2, 3, 4, 5],
        'max_features' : ['auto', 'sqrt']
    }

    if(oversamp):
        param_map = {
        'clf__n_estimators' : [10, 80, 100],
        'clf__bootstrap' : [True, False],
        'clf__max_depth' : [2, 8, 10, 20, 50],
        'clf__min_samples_split' : [2, 4,  6],
        'clf__min_samples_leaf' : [1,  3,  5],
        'clf__max_features' : ['auto', 'sqrt']
        }


    grid_search_hyp_tuning(RandomForestClassifier(), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp)

def grid_search_SVM(X_train, X_test, y_train, y_test, oversamp=True):

    scores = [
        'precision',
        'recall', 
        'f1',
        'balanced_accuracy',
        'roc_auc'
        ]

    param_map = [
            {'class_weight' : ['balanced'], 'kernel' : ['rbf'], 'gamma': [1e-3, 1e-4, 0.01, 0.1, 1, 1e-5], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            {'class_weight' : ['balanced'], 'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
            {'class_weight' : ['balanced'], 'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        ]

    if(oversamp):
        param_map = [
            {'clf__kernel': ['rbf'], 'clf__gamma': [1e-3, 1e-4, 0.01, 0.1, 1, 1e-5], 'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            {'clf__kernel': ['sigmoid'], 'clf__gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'clf__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
            {'clf__kernel': ['linear'], 'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        ]


    grid_search_hyp_tuning(svm.SVC(), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp)

def grid_search_ADA(X_train, X_test, y_train, y_test, oversamp=True):

    scores = [
        'precision',
        'recall', 
        'f1',
        'balanced_accuracy',
        'roc_auc'
        ]

    param_map = {
        'n_estimators': [10, 50, 100, 200],
        'learning_rate' : [0.01,0.05,0.1,0.3,1],
        'algorithm' : {'SAMME', 'SAMME.R'}
        }

    if(oversamp):
        param_map = {
        'clf__n_estimators': [10, 50, 100, 200],
        'clf__learning_rate' : [0.01, 0.05, 0.1, 0.3, 1],
        'clf__algorithm' : ['SAMME', 'SAMME.R']
        }

    
    grid_search_hyp_tuning(AdaBoostClassifier(), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp)


def multiple_grid_search(data, col_sets=[], norm_by_subject=True, dropna=True, oversamp=True, oversamp_mode="minmax"):


    for cols in col_sets:
        if(dropna):
            data = data.dropna()
        else:
            data = data.fillna(0)

        if(norm_by_subject):
            data = normalizeWithinSubject(data, cols, mode=oversamp_mode)

        print(" Multiple Grid Search ")
        print("==============================")
        print("columns {}".format(cols))
        print("number of datapoint: {}".format(len(data.index.values)))
        print("Normalize: {}".format(norm_by_subject))
        print("Oversample: {}".format(oversamp))
        print("Oversample Mode: {}".format(oversamp_mode))
        print("Drop NaN: {}".format(dropna))
        print("==============================")
        
        X = data[cols]
        y = data['label']

        print("Split in Train and Test set, equal for all the models")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42, shuffle=False)

        print("==============================")
        print("Train Set Datapoints {}".format(len(X_train)))
        print("Test Set Datapoints {}".format(len(X_test)))
        print("==============================")

        sys.stdout.flush()
        """print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        print("==============================")
        print(" Support Vector Machine ")
        print("==============================")
        grid_search_SVM(X_train, X_test, y_train, y_test, oversamp=oversamp)

        sys.stdout.flush()
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        print("==============================")
        print(" Decision Tree ")
        print("==============================")
        grid_search_Decision_Tree(X_train, X_test, y_train, y_test, oversamp=oversamp)

        sys.stdout.flush()
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        print("==============================")
        print(" Adaptive Boosting ")
        print("==============================")
        grid_search_ADA(X_train, X_test, y_train, y_test, oversamp=oversamp)

        sys.stdout.flush()
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")"""

        print("==============================")
        print(" Random Forest ")
        print("==============================")
        grid_search_Random_Forest(X_train, X_test, y_train, y_test, oversamp=oversamp)

        sys.stdout.flush()

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")




def train_SVM(data, cols, norm_by_subject=True, dropna=True, oversamp=True):

    if(dropna):
        data = data.dropna()
    else:
        data = data.fillna(0)

    if(norm_by_subject):
        data = normalizeWithinSubject(data, cols, mode="minmax")

    print("Train a Support Vector Machine")
    print("==============================")
    print("columns {}".format(cols))
    print("number of datapoint: {}".format(len(data.index.values)))
    print("Normalize: {}".format(norm_by_subject))
    print("Oversample: {}".format(oversamp))
    print("Drop NaN: {}".format(dropna))
    print("==============================")

    X = data[cols]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109) # 70% training and 30% test
    
    if(oversamp):
        X_train, y_train = ADASYN().fit_resample(X_train, y_train)

    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    print("F1:",metrics.f1_score(y_test, y_pred))
    

