import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats
pd.options.mode.chained_assignment = None  # default='warn'

# ========= MACHINE LEARNING ========================
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import classification_report

# Importing libraries for building the neural network
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from pprint import pprint

# ========= PROJECT ========================

from eye_feature_tools import *
from evaluation_metrics import *

# Evaluate the paired t-test for each feature
def paired_t_test(features, origin_cols, features_to_test, print_result=False, only_rel=False):

    results = pd.DataFrame(columns=["feature", 't-score', 'p', 'is_sign'])

    nonTargets, targets, TnT = aggregate_target_nontarget(features, origin_cols)

    nonTargets = nonTargets.set_index(nonTargets['subject'])
    targets = targets.set_index(targets['subject'])

    nonTargets = nonTargets.sort_index()
    targets = targets.sort_index()


    for col in features_to_test:
        t, p = stats.ttest_rel(targets[col],nonTargets[col])

        is_rel = ""
        if(p <= 0.05):
            is_rel = "*"
        
        if(p <= 0.001):
            is_rel = "**"

        res = pd.DataFrame(data=[[col, t, p, is_rel]], columns=["feature", 't-score', 'p', 'is_sign'])
        results = results.append(res, ignore_index=True)

    if(print_result):
        print("===== PAIRED T-test RESULTS =====")
        for index, row in results.iterrows():

            if(only_rel and "*" not in row['is_sign']):
                continue

            print("{} T:{} p:{}   {}".format(
                row['feature'], row['t-score'], row['p'], row['is_sign']
                ))

    return results


def max_mean_heuristic(features, features_to_test, print_result=False, only_rel=False):

    results = pd.DataFrame(columns=["feature", 'accuracy', 'is_sign'])
    subjects = features.groupby('subject').count().index.values
    cards = features.groupby('card_class').count().index.values

    for col in features_to_test:
        feat_restricted = features[['subject', col, 'card_class', 'label']]

        count = 0

        for sub in subjects:
            sub_feats = feat_restricted.loc[feat_restricted['subject'] == sub]
            sub_card = sub_feats.loc[sub_feats['label'] == 1]['card_class'].values[0]
            
            diffs = []

            for card in cards:

                


            max_f = sub_feats[col].max()
            max_card = sub_feats.loc[sub_feats[col] == max_f]['card_class'].values[0]

            if(sub_card == max_card):
                count += 1

        accuracy = count/len(subjects)
        is_sign = (accuracy >= 0.5)

        res = pd.DataFrame(data=[[col, accuracy, is_sign]], columns=["feature", 'accuracy', 'is_sign'])
        results = results.append(res, ignore_index=True)


    results = results.sort_values(by=['accuracy'], ascending=True)

    if(print_result):
        print("===== TAKE MAX RESULTS =====")
        for index, row in results.iterrows():

            if(only_rel and not row['is_sign']):
                continue

            print("{} acc:{}   {}".format(
                row['feature'], row['accuracy'], row['is_sign']
                ))




def take_max_heuristic(features, features_to_test, print_result=False, only_rel=False):

    results = pd.DataFrame(columns=["feature", 'accuracy', 'is_sign'])
    subjects = features.groupby('subject').count().index.values

    for col in features_to_test:
        feat_restricted = features[['subject', col, 'card_class', 'label']]

        count = 0

        for sub in subjects:
            sub_feats = feat_restricted.loc[feat_restricted['subject'] == sub]
            sub_card = sub_feats.loc[sub_feats['label'] == 1]['card_class'].values[0]
            
            max_f = sub_feats[col].max()
            max_card = sub_feats.loc[sub_feats[col] == max_f]['card_class'].values[0]

            if(sub_card == max_card):
                count += 1

        accuracy = count/len(subjects)
        is_sign = (accuracy >= 0.5)

        res = pd.DataFrame(data=[[col, accuracy, is_sign]], columns=["feature", 'accuracy', 'is_sign'])
        results = results.append(res, ignore_index=True)


    results = results.sort_values(by=['accuracy'], ascending=True)

    if(print_result):
        print("===== TAKE MAX RESULTS =====")
        for index, row in results.iterrows():

            if(only_rel and not row['is_sign']):
                continue

            print("{} acc:{}   {}".format(
                row['feature'], row['accuracy'], row['is_sign']
                ))


# =================== MACHINE LEARNING MODELS ======================
# Wrappers to put the defined Hyper-params
# Inrternal method to train with the defined Hyper-params

def decisionTreeFactory(depth=4, criterion='gini', weights=None, classes=[0,1]):
    
    def trainDecisionTree(Xtr, Xte, Ytr, Yte, feat_cols, print_metrics=False):
        model = DecisionTreeClassifier(max_depth=depth, criterion=criterion, class_weight=weights)
        model = model.fit(Xtr,Ytr)
        y_pred = model.predict(Xte)
        return calcMetrics(Yte, y_pred, classes, model, norm=True, cols=feat_cols, print_metrics=print_metrics), model
    
    return trainDecisionTree


def randomForestFactory(n_estimators=100, depth=110, weights=None, classes=[0,1], \
    bootstrap=False, max_features='sqrt', min_samples_leaf=1, min_samples_split=5):
    
    def trainRandomForest(Xtr, Xte, Ytr, Yte, feat_cols, print_metrics=False):
        model=RandomForestClassifier(n_estimators=n_estimators, class_weight=weights, max_depth=depth)
        model.fit(Xtr,Ytr)    
        y_pred=model.predict(Xte)

        return calcMetrics(Yte, y_pred, classes, model, norm=True, cols=feat_cols, print_metrics=print_metrics), model
    
    return trainRandomForest


def ADAfactory(estimator=None, n_estim=1000, classes=[0,1]):

    def trainADA(Xtr, Xte, Ytr, Yte, feat_cols, print_metrics=False):
        model = AdaBoostClassifier(base_estimator=estimator, n_estimators=n_estim)
        model.fit(Xtr , Ytr)
        y_pred = model.predict(Xte)

        return calcMetrics(Yte, y_pred, classes, model, norm=True, cols=feat_cols, print_metrics=print_metrics), model
    
    return trainADA


def gradientBoostingFactory(n_estim=1000, classes=[0,1]):
    
    def trainGradientBoosting(Xtr, Xte, Ytr, Yte, feat_cols, print_metrics=False):
        model = GradientBoostingClassifier(n_estimators=n_estim)
        model.fit(Xtr , Ytr)
        y_pred = model.predict(Xte)

        return calcMetrics(Yte, y_pred, classes, model, norm=True, cols=feat_cols, print_metrics=print_metrics), model
    
    return trainGradientBoosting

def linearRegressionFactory(n_estim=1000, classes=[0,1]):
    
    def trainLinearRegression(Xtr, Xte, Ytr, Yte, feat_cols, print_metrics=False):

        model = LinearRegression()  
        model.fit(Xtr, Ytr)
        y_pred = model.predict(Xte)
        return calcMetrics(Yte, y_pred, classes, model, norm=True, cols=feat_cols, print_metrics=print_metrics), model
    
    return trainLinearRegression


def crossValSMOTE(X_sub, y_sub, train_features, cols, model_names, train_eval_methods, N=5, split_mode='stratified', oversample="ADASYN"):

    """
    Generate the split in the train_subjects vector
    Filter the train_features by the subject
    Generate X and y
    """

    metrics_cols = ['baseline','model', 'accuracy', 'bal_accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    mDF = pd.DataFrame(columns=metrics_cols)
    
    if(len(model_names) != len(train_eval_methods)):
        return

    # split the subjects in N folds
    splitter = StratifiedKFold(n_splits=N)
    split_gen = splitter.split(X_sub, y_sub)

    if(split_mode == 'kfold'):
        splitter = KFold(n_splits=N, random_state=11)
        split_gen = splitter.split(X_sub)
    
    aggs = []
    
    for i in range(0, len(model_names)):
        aggregator = ModelMetricAggregator("sub", model_names[i])
        aggs.append(aggregator)
        
    # Generate Train and Val subject index
    for train_index, test_index in split_gen:
        # Generate Train and Test set
        sub_train = X_sub.iloc[train_index].values 
        sub_test = X_sub.iloc[test_index].values

        print("Train Set: {}".format(sub_train))
        print("Test Set: {}".format(sub_test))
        print("==========================================")

        train = train_features.loc[train_features['subject'].isin(sub_train)]
        test = train_features.loc[train_features['subject'].isin(sub_test)]

        X_train = train[cols]
        y_train = train['label']
        X_test = test[cols]
        y_test = test['label']

        #print(X_train.shape)
        #print(y_train.shape)

        if(oversample == "SMOTE"):
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)

        if(oversample == "ADASYN"):
            X_train, y_train = ADASYN().fit_resample(X_train, y_train)

        #print(X_train.shape)
        #print(y_train.shape)
        
        for i in range(0, len(model_names)):

            print("TRAIN --->{}".format(model_names[i]))

            (acc, bal_acc, prec, rec, f1, auc, best), model = \
                train_eval_methods[i](X_train, X_test, y_train, y_test, cols)
            
            aggs[i].addMetrics(acc, bal_acc, prec, rec, f1, auc)            
     
    
    for agg in aggs:
        row = agg.getMetrics()
        series = pd.Series(row, index=metrics_cols)
        mDF = mDF.append(series, ignore_index=True)
    
    return mDF


def evaluate_machine_learning_models(features, feature_cols, N=4, oversample="ADASYN"):

    # Get all the subjects
    
    sub_label = features[['subject', 'label']]
    sub_label = sub_label.loc[sub_label['label'] == 1]
    subjects = sub_label.groupby('subject').count().index.values

    print(subjects)

    sub_label = sub_label.reset_index()

    X = sub_label['subject']
    y = sub_label['label']

    X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if(N == 0):
        N = len(X_sub_train)

    
    res = crossValSMOTE(X_sub_train, y_sub_train, features,
                      feature_cols,
                      ['tree', 'forest'],
                      [decisionTreeFactory(), randomForestFactory()],
                      N=N, oversample=oversample)

    return None
    #return res



def evaluate_random_forest(features, feature_cols, N=4, oversample="SMOTE"):

    X = features[feature_cols]
    y = features['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=False)
    
    if(N == 0):
        N = len(X_train)
    
    if(oversample == "SMOTE"):
        print(oversample)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    if(oversample == "ADASYN"):
        print(oversample)
        X_train, y_train = ADASYN().fit_resample(X_train, y_train)

    factory = randomForestFactory(n_estimators=500, depth=4)
    (acc, bal_acc, prec, rec, f1, auc, best), model = \
                factory(X_train, X_val, y_train, y_val, feature_cols, print_metrics=True)

    print('\nParameters currently in use:\n')
    pprint(model.get_params())
    plt.show()

def trainMLP(features, feature_cols, oversample=None):


    # GET DATA
    X = features[feature_cols]
    y = features['label']

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_0, y_train_0, test_size=0.25, random_state=42, shuffle=False)
    
    if(oversample == "SMOTE"):
        print(oversample)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    if(oversample == "ADASYN"):
        print(oversample)
        X_train, y_train = ADASYN().fit_resample(X_train, y_train)
    # ============================================================

    model = Sequential()
    model.add(Dense(2,input_dim = 2, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])
    model.fit(X_train, y_train, epochs = 30, batch_size = 2, validation_data = (X_val, y_val))


    preds = model.predict_classes(X_test_0)
    print (classification_report(y_test_0, preds))


"""def hyperParamTuning(features, feature_cols, oversample="ADASYN", param_grid):
    # GET DATA
    X = features[feature_cols]
    y = features['label']

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_0, y_train_0, test_size=0.25, random_state=42, shuffle=False)
    
    if(oversample == "SMOTE"):
        print(oversample)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    if(oversample == "ADASYN"):
        print(oversample)
        X_train, y_train = ADASYN().fit_resample(X_train, y_train)
    # ============================================================"""





def decisionTreeHyperTuning(features, feature_cols, oversample="SMOTE"):

    print("______________________________________________________ RANDOM FOREST HT")

    # Number of trees in random forest
    criterion = ['entropy', 'gini']
    # Number of features to consider at every split
    splitter = ['best', 'random']
    # Maximum number of levels in tree
    max_depth = [2, 4, 6, 8, 10, 20]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 4, 5, 6]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4, 5]
    # Method of selecting samples for training each tree
    max_features = ['auto', 'sqrt']


    # Create the random grid
    """random_grid = {'criterion': criterion,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'splitter': splitter}"""


    # GET DATA
    X = features[feature_cols]
    y = features['label']

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_0, y_train_0, test_size=0.25, random_state=42, shuffle=False)
    
    if(oversample == "SMOTE"):
        print(oversample)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    if(oversample == "ADASYN"):
        print(oversample)
        X_train, y_train = ADASYN().fit_resample(X_train, y_train)
    # ============================================================

    # Define a custom scoring
    scores = {'clf__F1': 'f1'}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    dt = DecisionTreeClassifier()
    param_root=""

    if(oversample == "SMOTE"):
        dt = Pipeline([
            ('sampling', SMOTE()),
            ('clf', DecisionTreeClassifier())
        ])

    if(oversample == "ADASYN"):
        dt = Pipeline([
            ('sampling', ADASYN()),
            ('clf', DecisionTreeClassifier())
        ])

    random_grid = {
                'clf__criterion': criterion,
                'clf__max_features': max_features,
                'clf__max_depth': max_depth,
                'clf__min_samples_split': min_samples_split,
                'clf__min_samples_leaf': min_samples_leaf,
                'clf__splitter': splitter}



    print(random_grid)

    
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    dt_grid = GridSearchCV(estimator = dt, scoring=scores, refit="clf__F1", param_grid = random_grid, cv = 3, verbose=2, n_jobs = -1)

    # Fit the random search model
    dt_grid.fit(X_train, y_train)

    # ============================================================

    print("================ RF BEST GRID (VALIDATION) ====================")
    pprint(dt_grid.best_params_)
    print("------------------------------")
    grid_tree = dt_grid.best_estimator_
    y_pred_grid=grid_tree.predict(X_val)
    calcMetrics(y_val, y_pred_grid, [0,1], grid_tree, norm=True, cols=feature_cols, print_metrics=True)


    print("================= BEST PARAM TEST SET (FINAL) ======================")

    if(oversample == "SMOTE"):
        print(oversample)
        X_train_0, y_train_0 = SMOTE().fit_resample(X_train_0, y_train_0)

    if(oversample == "ADASYN"):
        print(oversample)
        X_train_0, y_train_0 = ADASYN().fit_resample(X_train_0, y_train_0)

    best_param = dt_grid.best_params_
    
    best_tree = DecisionTreeClassifier(
        criterion=best_param['clf__criterion'],
        max_depth=best_param['clf__max_depth'],
        max_features=best_param['clf__max_features'],
        min_samples_leaf=best_param['clf__min_samples_leaf'],
        min_samples_split=best_param['clf__min_samples_split'],
        splitter=best_param['clf__splitter']
     )
    best_tree.fit(X_train_0, y_train_0)
    y_pred_best=best_tree.predict(X_test_0)
    (acc, bal_acc, prec, rec, f1, auc, best) = calcMetrics(
        y_test_0, y_pred_best, [0, 1], best_tree, norm=True, cols=feature_cols, print_metrics=True)


def randomForestHyperTuning(features, feature_cols, oversample="SMOTE"):

    # Number of trees in random forest
    n_estimators = [80, 100, 800, 1000]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [4, 10, 50, 80]
    # Minimum number of samples required to split a node
    min_samples_split = [4, 5, 6]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3]
    # Method of selecting samples for training each tree
    bootstrap = [False]


    # Create the random grid
    random_grid = {'clf__n_estimators': n_estimators,
                'clf__max_features': max_features,
                'clf__max_depth': max_depth,
                'clf__min_samples_split': min_samples_split,
                'clf__min_samples_leaf': min_samples_leaf,
                'clf__bootstrap': bootstrap}


    # GET DATA
    X = features[feature_cols]
    y = features['label']

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_0, y_train_0, test_size=0.25, random_state=42, shuffle=False)
    
    if(oversample == "SMOTE"):
        print(oversample)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    if(oversample == "ADASYN"):
        print(oversample)
        X_train, y_train = ADASYN().fit_resample(X_train, y_train)
    # ============================================================

    # Define a custom scoring
    scores = {'clf__F1': 'f1'}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    #rf = RandomForestClassifier()

    rf = Pipeline([
            ('sampling', SMOTE()),
            ('clf', RandomForestClassifier())
        ])
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_grid = GridSearchCV(estimator = rf, scoring=scores, refit="clf__F1", param_grid = random_grid, cv = 3, verbose=2, n_jobs = -1)

    # Fit the random search model
    rf_grid.fit(X_train, y_train)

    # ============================================================
    print("================ BASE (VALIDATION) =======================")
    base_forest = randomForestFactory()
    (acc, bal_acc, prec, rec, f1, auc, best), model = base_forest(X_train, X_val, y_train, y_val, feature_cols, print_metrics=True)

    print("================ RF BEST GRID (VALIDATION) ====================")
    pprint(rf_grid.best_params_)
    print("------------------------------")
    grid_forest = rf_grid.best_estimator_
    y_pred_grid=grid_forest.predict(X_val)
    calcMetrics(y_val, y_pred_grid, [0,1], grid_forest, norm=True, cols=feature_cols, print_metrics=True)



    print("================= BEST PARAM TEST SET (FINAL) ======================")

    if(oversample == "SMOTE"):
        print(oversample)
        X_train_0, y_train_0 = SMOTE().fit_resample(X_train_0, y_train_0)

    if(oversample == "ADASYN"):
        print(oversample)
        X_train_0, y_train_0 = ADASYN().fit_resample(X_train_0, y_train_0)

    best_param = rf_grid.best_params_
    
    best_forest = RandomForestClassifier(
        n_estimators=best_param['clf__n_estimators'],
        max_depth=best_param['clf__max_depth'],
        max_features=best_param['clf__max_features'],
        min_samples_leaf=best_param['clf__min_samples_leaf'],
        min_samples_split=best_param['clf__min_samples_split']
     )
    best_forest.fit(X_train_0, y_train_0)
    y_pred_best=best_forest.predict(X_test_0)
    (acc, bal_acc, prec, rec, f1, auc, best) = calcMetrics(
        y_test_0, y_pred_best, [0, 1], best_forest, norm=True, cols=feature_cols, print_metrics=True)

