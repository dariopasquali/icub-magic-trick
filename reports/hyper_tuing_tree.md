# ALL THE FEATURES
```
================ RF BEST GRID (VALIDATION) ====================
{'criterion': 'entropy',
 'max_depth': 20,
 'max_features': 'auto',
 'min_samples_leaf': 4,
 'min_samples_split': 6,
 'splitter': 'best'}
------------------------------

Normalized confusion matrix
[[0.93333333 0.06666667]
 [0.5        0.5       ]]

Accuracy: 0.8421052631578947
Balanced Accuracy:  0.7166666666666667
Precision:  0.6666666666666666
Recall:  0.5
F1 Score:  0.5714285714285715
ROC AuC Score:  nan
================= BEST PARAM TEST SET (FINAL) ======================
ADASYN

Normalized confusion matrix
[[0.63636364 0.36363636]
 [0.         1.        ]]

Accuracy: 0.6923076923076923
Balanced Accuracy:  0.8181818181818181
Precision:  0.3333333333333333
Recall:  1.0
F1 Score:  0.5
ROC AuC Score:  nan
```
# ONLY DESCR MEAN
```
================ RF BEST GRID (VALIDATION) ====================
{'criterion': 'entropy',
 'max_depth': 4,
 'max_features': 'sqrt',
 'min_samples_leaf': 2,
 'min_samples_split': 6,
 'splitter': 'random'}
------------------------------

Normalized confusion matrix
[[0.26666667 0.73333333]
 [0.         1.        ]]

Accuracy: 0.42105263157894735
Balanced Accuracy:  0.6333333333333333
Precision:  0.26666666666666666
Recall:  1.0
F1 Score:  0.4210526315789474
ROC AuC Score:  nan
================= BEST PARAM TEST SET (FINAL) ======================
ADASYN

Normalized confusion matrix
[[0.36363636 0.63636364]
 [0.25       0.75      ]]

Accuracy: 0.4230769230769231
Balanced Accuracy:  0.5568181818181819
Precision:  0.17647058823529413
Recall:  0.75
F1 Score:  0.2857142857142857
ROC AuC Score:  nan
```