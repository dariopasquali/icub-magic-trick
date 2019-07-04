import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats
pd.options.mode.chained_assignment = None  # default='warn'

from eye_feature_tools import *

# Evaluate the paired t-test for each feature
def paired_t_test(features, columns, print_result=False):

    results = pd.DataFrame(columns=["feature", 't-score', 'p', 'i_significant'])

    nonTargets, targets, TnT = aggregate_target_nontarget(features, columns)

    for col in columns:
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
            print("{} T:{} p:{}   {}".format(
                row['feature'], row['t-score'], row['p'], row['is_sign']
                ))

    return results
    

