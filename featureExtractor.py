import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

from time_series_extractor import *
from SubjectMagicFeatures import *
from LieFeatures import *
from plot_tools import *
from lie_plot_tools import *


column_names = [
        'subject','source','duration','card_class','show_order',
        'fix_freq','sacc_freq',
        'pd_right_mean','pd_right_std','pd_right_min','pd_right_max',
        'pd_left_mean','pd_left_std','pd_left_min','pd_left_max',
        'sre_fix_freq','sre_sacc_freq',
        'sre_pd_right_mean','sre_pd_right_std',
        'sre_pd_right_min','sre_pd_right_max',
        'sre_pd_left_mean','sre_pd_left_std',
        'sre_pd_left_min','sre_pd_left_max',
        'srl_fix_freq','srl_sacc_freq',
        'srl_pd_right_mean','srl_pd_right_std',
        'srl_pd_right_min','srl_pd_right_max',
        'srl_pd_left_mean','srl_pd_left_std',
        'srl_pd_left_min','srl_pd_left_max',
        'label'
    ]

lie_column_names = [
        'subject',
        'source',
        'card_class',
        'show_order',
        'duration',
        'react_dur',
        'point_react_dur',
        'descr_dur',
        'fix_freq','sacc_freq',
        'right_mean','right_std','right_min','right_max',
        'left_mean','left_std','left_min','left_max',
        'react_fix_freq', 'react_sacc_freq',
        'react_right_mean', 'react_right_std', 'react_right_min', 'react_right_max',
        'react_left_mean', 'react_left_std', 'react_left_min', 'react_left_max',
        'point_react_fix_freq', 'point_react_sacc_freq',
        'point_react_right_mean', 'point_react_right_std','point_react_right_min','point_react_right_max',
        'point_react_left_mean','point_react_left_std','point_react_left_min','point_react_left_max',
        'descr_fix_freq','descr_sacc_freq',
        'descr_right_mean','descr_right_std','descr_right_min','descr_right_max',
        'descr_left_mean','descr_left_std','descr_left_min','descr_left_max',
        'label'
    ]

feat_cols = [
        'duration',
        'fix_freq',
        'sacc_freq',
        'pd_right_mean',
        'pd_right_std',
        'pd_right_min',
        'pd_right_max',
        'pd_left_mean',
        'pd_left_std',
        'pd_left_min',
        'pd_left_max',
        'sre_fix_freq',
        'sre_sacc_freq',
        'sre_pd_right_mean',
        'sre_pd_right_std',
        'sre_pd_right_min',
        'sre_pd_right_max',
        'sre_pd_left_mean',
        'sre_pd_left_std',
        'sre_pd_left_min',
        'sre_pd_left_max',
        'srl_fix_freq',
        'srl_sacc_freq',
        'srl_pd_right_mean',
        'srl_pd_right_std',
        'srl_pd_right_min',
        'srl_pd_right_max',
        'srl_pd_left_mean',
        'srl_pd_left_std',
        'srl_pd_left_min',
        'srl_pd_left_max',
        'label'
    ]


sr_window = 1500
card_names = ['unicorn', 'pepper', 'minion', 'pig', 'hedge', 'aliens']

def extractLieFeatures(cols=lie_column_names, cards=card_names, source="frontiers", ref_to_base=True, mode="sub"):

    features = pd.DataFrame(columns=cols)
    subjects = extractMinSubjectSet(source, annot_path=annotations_lie_in_temp)

    baseline_right = []
    baseline_left = []

    for sub in subjects:

        eye_df, annot_dfs, baseline, overall_eye_df, filtered_interaction_dfs = \
                loadLieTimeSeries(sub, card_names, source="frontiers", clean=True, clean_mode="MAD", smooth=False) 
        
        mf = LieFeatures(sub, "frontiers",
                            annot_dfs,
                            filtered_interaction_dfs,
                            refer_to_baseline=True,
                            baseline=baseline,
                            refer_method=mode
                        )
        baseline_right.append(baseline['diam_right'].mean(skipna = True))
        baseline_left.append(baseline['diam_left'].mean(skipna = True)) 
        features = features.append(mf.getDataFrame(), ignore_index=True)
        
    return features, baseline_right, baseline_left


def extractFeatures(cols, cards, source="frontiers", short_resp=1500, ref_to_base=True, mode="sub"):
    
    features = pd.DataFrame(columns=cols)
    subjects = extractMinSubjectSet(source)

    baseline_right = []
    baseline_left = []
    
    for sub in subjects:
        eye_df, annot_dfs, baseline, overall_eye_df, \
            cards_eye_dfs, early_sr_dfs, late_sr_dfs = \
                loadTimeSeries(sub, cards, source=source, clean=True, clean_mode="MAD", smooth=False) 
        
        subject_card = getSubjectCard(sub, source, subject_cards_file)
        
        mf = SubjectMagicFeatures(sub, source, subject_card,
                              annot_dfs,
                              cards_eye_dfs, early_sr_dfs, late_sr_dfs,
                              short_resp,
                              cols=cols,
                              refer_to_baseline=ref_to_base,
                              baseline=baseline,
                              refer_method=mode
                             )

        baseline_right.append(baseline['diam_right'].mean(skipna = True))
        baseline_left.append(baseline['diam_left'].mean(skipna = True))        
        features = features.append(mf.getDataFrame(), ignore_index=True)
        
    return features, baseline_right, baseline_left

"""
def extractAndSaveAll(column_names=column_names, card_names=card_names, out_file="features/all.csv", ref_to_base=True, mode="sub", short_resp=1500):
    frontiers = extractFeatures(column_names, card_names, source="frontiers", ref_to_base=ref_to_base, mode=mode, short_resp=short_resp)
    pilot = extractFeatures(column_names, card_names, source="pilot", ref_to_base=ref_to_base, mode=mode, short_resp=short_resp)
    pilot['subject'] = pilot['subject'] + 100
    feats = frontiers.append(pilot, ignore_index=True)

    feats.to_csv(out_file, columns=column_names, sep='\t', index=False)
    pilot.to_csv("features/all_pilot.csv", columns=column_names, sep='\t', index=False)
    frontiers.to_csv("features/all_fontiers.csv", columns=column_names, sep='\t', index=False)

    return feats, frontiers, pilot
"""
