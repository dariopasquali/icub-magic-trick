# DATA ANALYSIS AND FEATURE PREDICTION - Trial and Errors

## DATASETS

### Features

#### Early Short Response (1.5 sec after looking at the card begin)
* data/features_none_sre.csv: No baseline
* data/features_sub_sre.csv: Subtracted by baseline
* data/features_div_sre.csv: Divided by baseline

#### Late Short Response (1.5 sec before looking at the card end)
* data/features_none_srl.csv: No baseline
* data/features_sub_srl.csv: Subtracted by baseline
* data/features_div_srl.csv: Divided by baseline

For Each one, there is a problem of **imalancing** of the classes, in particular we have:
* 1/5 subject old card (class 1)
* 5/6 new card (class 0)

This imbalancing can cause a problem like: i I have to detect the 1 I'll always tell 0. So the accuracy will be very high but the classifier will be crap.

So the idea is to:
* **Rebalance the dataset**
* Use different evaluation rather than the accuracy (like the **F1_score**)

## Dataset Rebalancing

I tried two alternatives:
* Oversample the class 1 (with the SMOTE algorithm)
* Aggregate the class 0 by subject in order to balance the dataset

## MODELS
---- SMOTE APPLIED to oversample the 1 class

### Decision Tree

