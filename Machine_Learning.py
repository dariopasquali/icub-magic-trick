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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.model_selection import GridSearchCV

import sys


# ==========================================================

class GridSearchEngine:

    def __init__(self):
        self.ml_methods = {}

    def add_decision_tree(self):
        self.ml_methods['decision_tree'] = self.grid_search_Decision_Tree

    def add_random_forest(self):
        self.ml_methods['random_forest'] = self.grid_search_Random_Forest

    def add_svm(self):
        self.ml_methods['svm'] = self.grid_search_SVM

    def add_ada(self):
        self.ml_methods['ada'] = self.grid_search_ADA

    def add_naive_bayes(self):
        self.ml_methods['naive_bayes'] = self.grid_search_naive_bayes

    def add_knn(self):
        self.ml_methods['knn'] = self.grid_search_knn

    def add_mlp(self):
        self.ml_methods['mlp'] = self.grid_search_mlp


    def grid_search_Decision_Tree(self, X_train, X_test, y_train, y_test, oversamp=True):

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


        report = self.grid_search_hyp_tuning(DecisionTreeClassifier(), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp)
        report['model'] = 'decision_tree'
        return report

    def grid_search_Random_Forest(self, X_train, X_test, y_train, y_test, oversamp=True):

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


        report = self.grid_search_hyp_tuning(RandomForestClassifier(), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp)
        report['model'] = 'random_forest'
        return report

    def grid_search_SVM(self, X_train, X_test, y_train, y_test, oversamp=True):

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


        report = self.grid_search_hyp_tuning(svm.SVC(), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp)
        report['model'] = 'svm'
        return report

    def grid_search_ADA(self, X_train, X_test, y_train, y_test, oversamp=True):

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

        
        report = self.grid_search_hyp_tuning(AdaBoostClassifier(), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp)
        report['model'] = 'ada'
        return report

    def grid_search_naive_bayes(self, X_train, X_test, y_train, y_test, oversamp=True):

        scores = [
            'precision',
            'recall', 
            'f1',
            'balanced_accuracy',
            'roc_auc'
            ]

        param_map = {
            'var_smoothing': [1e-9, 2e-9, 10e-9, 20e-9, 50e-9]
            }

        if(oversamp):
            param_map = {
            'clf__var_smoothing': [1e-9, 2e-9, 10e-9, 20e-9, 50e-9]
            }

        
        report = self.grid_search_hyp_tuning(GaussianNB(), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp)
        report['model'] = 'gaussian_nb'
        return report

    def grid_search_knn(self, X_train, X_test, y_train, y_test, oversamp=True):

        scores = [
            'precision',
            'recall', 
            'f1',
            'balanced_accuracy',
            'roc_auc'
            ]

        param_map = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'p' : [1, 2, 3, 4]
            }

        if(oversamp):
            param_map = {
            'clf__n_neighbors': [3, 5, 7, 9],
            'clf__weights': ['uniform', 'distance'],
            'clf__algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'clf__p' : [1, 2, 3, 4]
            }

        
        report = self.grid_search_hyp_tuning(KNeighborsClassifier(), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp, standardize=True)
        report['model'] = 'knn'
        return report

    def grid_search_mlp(self, X_train, X_test, y_train, y_test, oversamp=True):

        scores = [
            'precision',
            'recall', 
            'f1',
            'balanced_accuracy',
            'roc_auc'
            ]

        param_map = {
            'hidden_layer_sizes': [(100,), (300, ), (32, 8, 32,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha' : [0.0001, 0.05, 0.005],
            }

        if(oversamp):
            param_map = {
            'clf__hidden_layer_sizes':  [(100,), (300, ), (32, 8, 32,)],
            'clf__activation': ['tanh', 'relu'],
            'clf__solver': ['sgd', 'adam'],
            'clf__alpha' : [0.0001, 0.05, 0.005],
            }


        
        report = self.grid_search_hyp_tuning(MLPClassifier(batch_size = 50), param_map, scores, X_train, X_test, y_train, y_test, oversamp=oversamp, standardize=True)
        report['model'] = 'mlp'
        return report

    def concat_dict(self, source, dict_to_add):
        df = source.copy()
        
        for col in dict_to_add:
            df[col] = dict_to_add[col]

        return df

    def grid_search_hyp_tuning(self, model, param_map, scores, X_train, X_test, y_train, y_test, oversamp=True, standardize=False):

        #print("Resplit the Test set in Test and Validtion")
        #X_train, X_val, y_train, y_val = train_test_split(X_train_0, y_train_0, test_size=0.25,random_state=42, shuffle=False)

        report = pd.DataFrame(
                columns=['tune_for',
                'mean',
                'std',
                'accuracy',
                'bal_accuracy',
                'precision',
                'recall',
                'f1',
                'AUROC',
                'params'
                ])

        pipeline = None

        if(oversamp):
            pipeline = Pipeline([
                    ('sampling', SMOTE()),
                    ('clf', model)
                ])

            if(standardize):
                pipeline = Pipeline([
                    ('scale', StandardScaler()),
                    ('sampling', SMOTE()),
                    ('clf', model)
                ])
        else:
            if(standardize):
                pipeline = Pipeline([
                    ('scale', StandardScaler()),
                    ('clf', model)
                ])        

        for score in scores:
            print("# =========================== Tuning hyper-parameters for %s" % score)
            print()

            sc = score
            #sc = '%s_macro' % score
            #if(oversamp):
            #    sc = 'clf__%s_macro' % score

            clf = GridSearchCV(pipeline, param_map, cv=4, scoring=sc, verbose=1)
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
            #print(classification_report(y_true, y_pred))
            print()
            print("Accuracy:",metrics.accuracy_score(y_true, y_pred))
            print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_true, y_pred))
            print("Precision:",metrics.precision_score(y_true, y_pred))
            print("Recall:",metrics.recall_score(y_true, y_pred))
            print("F1:",metrics.f1_score(y_true, y_pred))
            print("AUROC:",metrics.roc_auc_score(y_true, y_pred))
            print("Confusion Matrix:", confusion_matrix(y_true, y_pred))
            print()

            tune_rep = pd.DataFrame(columns=['tune_for',
                'mean',
                'std',
                'accuracy',
                'bal_accuracy',
                'precision',
                'recall',
                'f1',
                'AUROC',
                'params'
                ],
                data=[[score,
                str(clf.cv_results_['mean_test_score'][clf.best_index_]),
                str(clf.cv_results_['std_test_score'][clf.best_index_]),
                metrics.accuracy_score(y_true, y_pred),
                metrics.balanced_accuracy_score(y_true, y_pred),
                metrics.precision_score(y_true, y_pred),
                metrics.recall_score(y_true, y_pred),
                metrics.f1_score(y_true, y_pred),
                metrics.roc_auc_score(y_true, y_pred),
                str(clf.best_params_)]])

            report = report.append(tune_rep, ignore_index=True)
        
        return report



    def multiple_grid_search(self, data, col_sets={}, norm_by_subject=True, dropna=True, oversamp=True, oversamp_mode="minmax"):

        if(dropna):
            print("drop NaN")
            data = data.dropna()
        else:
            print("fill NaN with 0")
            data = data.fillna(0)

        temp_c = list(data.columns)
        temp_c.remove('label')

        csv_cols=[
            'cols_set',
            'datapoints',
            'normalize',
            'oversample',
            'drop_nan',
            'train_set',
            'test_set',
            'model',
            'tune_for',
            'mean',
            'std',
            'accuracy',
            'bal_accuracy',
            'precision',
            'recall',
            'f1',
            'AUROC',
            'params'
            ]

        if(norm_by_subject):
            print("normalize whithin subject")
            cols = temp_c.copy()
            cols.remove('subject')
            cols.remove('card_class')
            cols.remove('source')
            cols.remove('show_order')
            data = normalizeWithinSubject(data, cols, mode=oversamp_mode)

        y = data['label']
        X = data[temp_c]

        print("Split in Train and Test set, equal for all the models and column set")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42, shuffle=False)

        report = pd.DataFrame(columns=[
            'tune_for',
            'mean',
            'std',
            'accuracy',
            'bal_accuracy',
            'precision',
            'recall',
            'f1',
            'AUROC',
            'params',
            'model',
            'col_set',
            'datapoints',
            'normalize',
            'oversample',
            'drop_nan',
            'train_set',
            'test_set'
            ])

        for col_key in col_sets:

            cols = col_sets[col_key]

            root_dict = {
                'cols_set' : str(col_key),
                'datapoints' :len(data.index.values),
                'normalize' : str(norm_by_subject),
                'oversample' : oversamp_mode,
                'drop_nan' : str(dropna),
                'train_set' : len(X_train),
                'test_set' : len(X_test)
            }

            # Select only the defined column set
            X_train_1 = X_train[cols]
            X_test_1 = X_test[cols]

            print(" Multiple Grid Search ")
            print("==============================")
            print("columns {}".format(cols))
            print("number of datapoints: {}".format(len(data.index.values)))
            print("Normalize: {}".format(norm_by_subject))
            print("Oversample: {}".format(oversamp))
            print("Oversample Mode: {}".format(oversamp_mode))
            print("Drop NaN: {}".format(dropna))
            print("==============================")
            
                
            print("==============================")
            print("Train Set Datapoints {}".format(len(X_train_1)))
            print("Test Set Datapoints {}".format(len(X_test_1)))
            print("==============================")

            sys.stdout.flush()
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            for model in self.ml_methods:

                ml_method = self.ml_methods[model]

                print("==============================")
                print(" {} ".format(model))
                print("==============================")
                sys.stdout.flush()
                rep = ml_method(X_train_1, X_test_1, y_train, y_test, oversamp=oversamp)
                df = self.concat_dict(rep, root_dict)
                report = report.append(df, ignore_index=True)

                sys.stdout.flush()
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        report = report[csv_cols]
        return report

