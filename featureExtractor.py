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
from evaluation import *

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
        'point_dur',
        'descr_dur',
        'fix_freq','sacc_freq',
        'right_mean','right_std','right_min','right_max',
        'left_mean','left_std','left_min','left_max',
        'react_fix_freq', 'react_sacc_freq',
        'react_right_mean', 'react_right_std', 'react_right_min', 'react_right_max',
        'react_left_mean', 'react_left_std', 'react_left_min', 'react_left_max',
        'point_fix_freq', 'point_sacc_freq',
        'point_right_mean', 'point_right_std','point_right_min','point_right_max',
        'point_left_mean','point_left_std','point_left_min','point_left_max',
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

def extractLieFeatures(subjects=[], cols=lie_column_names, cards=card_names, source="frontiers", ref_to_base="time", mode="sub", plot=False, print_pupil=False):

    # refer_to_base = [time, feat, None]

    features = pd.DataFrame(columns=cols)

    if(len(subjects) == 0):
        subjects = extractMinSubjectSet(source, annot_path=annotations_lie_in_temp)

    baseline_right = []
    baseline_left = []

    ref_time = ref_to_base == 'time'
    ref_features = ref_to_base == 'feat'

    for sub in subjects:

        eye_df, annot_dfs, baseline, overall_eye_df, filtered_interaction_dfs = \
                loadLieTimeSeries(
                    sub, card_names, source="frontiers", 
                    refer_to_baseline=ref_time, refer_mode=mode,
                    clean=True, clean_mode="MAD", smooth=False) 
        
        mf = LieFeatures(sub, "frontiers",
                            annot_dfs,
                            filtered_interaction_dfs)

        baseline_right.append(baseline['diam_right'].mean(skipna = True))
        baseline_left.append(baseline['diam_left'].mean(skipna = True)) 

        sub_features = mf.getDataFrame()

        if(ref_features):
            print(">> refer features to baseline")

            lie_right_feats = [
                    'right_mean', 'right_min','right_max',
                    'react_right_mean', 'react_right_min', 'react_right_max',
                    'point_right_mean', 'point_right_min','point_right_max',
                    'descr_right_mean', 'descr_right_min','descr_right_max'
                ]

            lie_left_feats = [
                    'left_mean', 'left_min','left_max',
                    'react_left_mean', 'react_left_min', 'react_left_max',
                    'point_left_mean', 'point_left_min','point_left_max',
                    'descr_left_mean', 'descr_left_min','descr_left_max'
                ]

            sub_features = referEyeFeaturesToBaseline(sub_features, baseline, lie_right_feats, lie_left_feats)

        
        if(print_pupil):
            print(annot_dfs[0])
            for card in card_names:
                print("card {} {}".format(card, sub_features.loc[sub_features['card_class'] == card]['descr_right_mean'].values[0]))
            print("=======================================")
        

        features = features.append(sub_features, ignore_index=True)

        if(plot):
            lie_plotTimeSeries(sub, cards, annot_dfs, overall_eye_df, filtered_interaction_dfs)
            #, baseline)            


    return features, baseline_right, baseline_left

def extractFeatures(subject=None, cols=column_names, cards=card_names, source="frontiers", short_resp=1500, ref_to_base=True, mode="sub", plot=False):
    
    features = pd.DataFrame(columns=cols)

    subjects = []

    if(subject != None):
        subjects = [subject]
    else:
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
                              cols=cols
                            )

        baseline_right.append(baseline['diam_right'].mean(skipna = True))
        baseline_left.append(baseline['diam_left'].mean(skipna = True))
        features = features.append(mf.getDataFrame(), ignore_index=True)

        if(plot):
            plotPupilDilationTimeSeries(sub, subject_card, cards, overall_eye_df, cards_eye_dfs)
        
    return features, baseline_right, baseline_left




mode = "sub"
print("================== MODE {} =====================".format(mode))
lie_features, lie_base_right, lie_base_left = \
    extractLieFeatures(subjects=[], mode=mode, ref_to_base="time", plot=False, print_pupil=False)
#lie_features.to_csv("lie_features.csv", index=False)

#lie_plotComparBars(lie_features, lie_base_right, lie_base_left, save=False)
#lie_plotBySubject(lie_features, mode=mode, save=False)
#lie_plotTnTratioMean(lie_features, save=False)
#lie_plotTnTstem(lie_features, feature="left_mean", save=False)

paired_t_test(lie_features, lie_column_names, print_result=True)

#%matplotlib notebook
plt.show()
 