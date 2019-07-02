import pandas as pd
import numpy as np
from scipy import stats

from data_cleaning import *

def referEyeFeaturesToBaseline(features, baseline, right_eye_cols, left_eye_cols,
        right_base_col='diam_right', left_base_col='diam_left', mode='sub'):
    
    features = referToBaseline(features, baseline, mode=mode,
        source_cols=right_eye_cols, baseline_col=right_base_col)

    features = referToBaseline(features, baseline, mode=mode,
        source_cols=left_eye_cols, baseline_col=left_base_col)

    return features

def aggregate_target_nontarget(features, subject, cols, nonTarget, target):
    data = features[cols].loc[features['subject'] == subject]
    nt = data.loc[features['label'] == 0].drop(['card_class'], axis=1)
    t = data.loc[features['label'] == 1]

    for col in nt.columns:
        nt[col] = nt[col].mean()
    nt = nt.head(1)
    nt['card_class'] = "avg"
    
    nonTarget = nonTarget.append(nt, ignore_index=True)
    target = target.append(t, ignore_index=True)
    
    return nonTarget, target

def computePupilFeatures(data):
    # Right Eye
    p_mean_r = data["diam_right"].mean(skipna = True)
    p_std_r = data["diam_right"].std(skipna = True)
    p_max_r = data["diam_right"].max(skipna = True)
    p_min_r = data["diam_right"].min(skipna = True)
    
    # Left Eye
    p_mean_l = data["diam_left"].mean(skipna = True)
    p_std_l = data["diam_left"].std(skipna = True)
    p_max_l = data["diam_left"].max(skipna = True)
    p_min_l = data["diam_left"].min(skipna = True)
    
    return p_mean_r, p_std_r, p_min_r, p_max_r, \
             p_mean_l, p_std_l, p_min_l, p_max_l

def calcEventFeatures(data, time_window):
    move_types = data[['move_type', 'move_type_id']]
        
    fix_num = move_types.loc[move_types['move_type'] == "Fixation"]
    sacc_num = move_types.loc[move_types['move_type'] == "Saccade"]

    fix_num = len(fix_num.groupby('move_type_id').count().index)
    sacc_num = len(sacc_num.groupby('move_type_id').count().index)

    duration_sec = time_window / 1000
    fix_freq = fix_num / duration_sec
    sacc_freq = sacc_num / duration_sec

    return fix_freq, sacc_freq