# Binary Classification Metrics

**True Positive (TP)**: correctly predicted positive values
**True Negative (TN)**: correctly predicted negative values

**False Positive (FP)**: negtive label predicted as positive (I say that something is what I looking for)
**False Negative (FN)**: positive label predicted as negative (I miss something that I looking for)

**Accuracy** = {correctly predicted} / {total values}
         = (TP + TN) / (TP + TN + FP + FN)
         = if I give a value to the model, this is the probability to get the right result. It's a good evaluation metrics only if the dataset is balanced.

**Precision** = (TP) / (TP + FP)
          = Between the totally positive predicted values, how many are really positive? How many values I missclassified
          = High precision means lower errors (false positive)

**Recall** = (TP) / (TP + FN)
           = Between the totally positive values, how many I actually predicted as positive? How many values I missed

**F1 Score** = 2*(Recall * Precision) / (Recall + Precision)
             = usefull with imbalanced dataset (but not when the model always return the sae fucking value)

             