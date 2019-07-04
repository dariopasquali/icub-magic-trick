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

def extract_target_nontarget(features, subject, cols, nonTarget, target):
    data = features[cols].loc[features['subject'] == subject]
    nt = data.loc[features['label'] == 0].drop(['card_class'], axis=1)
    t = data.loc[features['label'] == 1]

    for col in nt.columns:
        nt[col] = nt[col].mean(skipna=True)
    nt = nt.head(1)
    nt['card_class'] = "avg"
    
    nonTarget = nonTarget.append(nt, ignore_index=True)
    target = target.append(t, ignore_index=True)
    
    return nonTarget, target

def aggregate_target_nontarget(features, cols):
    nonTargets = pd.DataFrame(columns=cols)
    targets = pd.DataFrame(columns=cols)
    TnT = pd.DataFrame(columns=cols)

    subjects = features.groupby('subject').count().index.values


    for sub in subjects:
        nonTargets, targets = extract_target_nontarget(
            features, sub, cols, nonTargets, targets)

    TnT.append(targets, ignore_index=True)
    TnT.append(nonTargets, ignore_index=True)

    return nonTargets, targets, TnT

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


def coumpute_TnT_scores(features, cols, score_col, norm_to_point=False, abs_ratio=False):

    nonTargets, targets, TnT = aggregate_target_nontarget(features, cols)

    subjects = features.groupby('subject').count().index.values
    score_columns = ['subject', 'point_ratio', 'react_ratio', 'descr_ratio']

    scores = pd.DataFrame(columns=score_columns)

    for sub in subjects:
        nT = nonTargets.loc[nonTargets['subject'] == sub]
        T = targets.loc[targets['subject'] == sub]

        nT_point = nT["point_" + score_col].values[0]
        T_point = T["point_" + score_col].values[0]
        
        nT_react = nT["react_" + score_col].values[0]
        T_react = T["react_" + score_col].values[0]

        nT_descr = nT["descr_" + score_col].values[0]
        T_descr = T["descr_" + score_col].values[0]

        point_ratio = T_point - nT_point
        react_ratio = T_react - nT_react
        descr_ratio = T_descr - nT_descr

        # absolute difference
        if(abs_ratio):
            point_ratio = np.abs(point_ratio)
            react_ratio = np.abs(react_ratio)
            descr_ratio = np.abs(descr_ratio)

        # report point to 0
        if(norm_to_point):
            react_ratio = react_ratio - point_ratio
            descr_ratio = descr_ratio - point_ratio
            point_ratio = point_ratio - point_ratio

        scr = pd.DataFrame(data=[[sub, point_ratio, react_ratio, descr_ratio]], columns=score_columns)
        scores = scores.append(scr, ignore_index=True)

    return scores, subjects
