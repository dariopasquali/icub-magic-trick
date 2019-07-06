# ALL THE FEATURES
```
================ RF BEST GRID (VALIDATION) ====================
{'bootstrap': False,
 'max_depth': 80,
 'max_features': 'sqrt',
 'min_samples_leaf': 2,
 'min_samples_split': 5,
 'n_estimators': 80}

Normalized confusion matrix
[[0.8 0.2]
 [0.5 0.5]]

Accuracy: 0.7368421052631579
Balanced Accuracy:  0.65
Precision:  0.4
Recall:  0.5
F1 Score:  0.4444444444444445
ROC AuC Score:  nan
================= BEST PARAM TEST SET (FINAL) ======================
ADASYN

Normalized confusion matrix
[[0.72727273 0.27272727]
 [0.5        0.5       ]]

Accuracy: 0.6923076923076923
Balanced Accuracy:  0.6136363636363636
Precision:  0.25
Recall:  0.5
F1 Score:  0.3333333333333333
ROC AuC Score:  nan
```

# ONLY DIAM DESCR
```
================ RF BEST GRID (VALIDATION) ====================
{'bootstrap': False,
 'max_depth': 4,
 'max_features': 'sqrt',
 'min_samples_leaf': 1,
 'min_samples_split': 4,
 'n_estimators': 1000}
------------------------------

Normalized confusion matrix
[[0.73333333 0.26666667]
 [0.75       0.25      ]]

Accuracy: 0.631578947368421
Balanced Accuracy:  0.49166666666666664
Precision:  0.2
Recall:  0.25
F1 Score:  0.22222222222222224
ROC AuC Score:  nan
================= BEST PARAM TEST SET (FINAL) ======================
ADASYN

Normalized confusion matrix
[[0.63636364 0.36363636]
 [0.25       0.75      ]]

Accuracy: 0.6538461538461539
Balanced Accuracy:  0.6931818181818181
Precision:  0.2727272727272727
Recall:  0.75
F1 Score:  0.39999999999999997
ROC AuC Score:  nan
```