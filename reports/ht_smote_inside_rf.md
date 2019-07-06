# ALL FEATURES
```

================ RF BEST GRID (VALIDATION) ====================
{'clf__bootstrap': False,
 'clf__max_depth': 50,
 'clf__max_features': 'auto',
 'clf__min_samples_leaf': 1,
 'clf__min_samples_split': 5,
 'clf__n_estimators': 100}
------------------------------

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
SMOTE

Normalized confusion matrix
[[0.77272727 0.22727273]
 [0.5        0.5       ]]

Accuracy: 0.7307692307692307
Balanced Accuracy:  0.6363636363636364
Precision:  0.2857142857142857
Recall:  0.5
F1 Score:  0.36363636363636365
ROC AuC Score:  nan
```

# ONLY DESCR MEAN

```
================ RF BEST GRID (VALIDATION) ====================
{'clf__bootstrap': False,
 'clf__max_depth': 4,
 'clf__max_features': 'sqrt',
 'clf__min_samples_leaf': 2,
 'clf__min_samples_split': 4,
 'clf__n_estimators': 80}
------------------------------

Normalized confusion matrix
[[0.73333333 0.26666667]
 [0.25       0.75      ]]

Accuracy: 0.7368421052631579
Balanced Accuracy:  0.7416666666666667
Precision:  0.42857142857142855
Recall:  0.75
F1 Score:  0.5454545454545454
ROC AuC Score:  nan
================= BEST PARAM TEST SET (FINAL) ======================
SMOTE

Normalized confusion matrix
[[0.77272727 0.22727273]
 [0.75       0.25      ]]

Accuracy: 0.6923076923076923
Balanced Accuracy:  0.5113636363636364
Precision:  0.16666666666666666
Recall:  0.25
F1 Score:  0.2
ROC AuC Score:  nan
```
