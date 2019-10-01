import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

from quest_reader import *
from time_series_extractor import *
from LieFeatures import *
from lie_plot_tools import *
from evaluation import *
from Machine_Learning import *

import sys

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
        #'pd_right_std',
        #'pd_right_min',
        #'pd_right_max',
        'pd_left_mean',
        #'pd_left_std',
        #'pd_left_min',
        #'pd_left_max',
        'sre_fix_freq',
        'sre_sacc_freq',
        'sre_pd_right_mean',
        #'sre_pd_right_std',
        #'sre_pd_right_min',
        #'sre_pd_right_max',
        'sre_pd_left_mean',
        #'sre_pd_left_std',
        #'sre_pd_left_min',
        #'sre_pd_left_max',
        'srl_fix_freq',
        'srl_sacc_freq',
        'srl_pd_right_mean',
        #'srl_pd_right_std',
        #'srl_pd_right_min',
        #'srl_pd_right_max',
        'srl_pd_left_mean',
        #'srl_pd_left_std',
        #'srl_pd_left_min',
        #'srl_pd_left_max',
        #'label'
    ]

lie_ML_cols = [
        'right_mean',
        'left_mean',
        'react_right_mean', 'react_right_std',
        'react_left_mean', 'react_left_std',
        'descr_right_mean','descr_right_std',
        'descr_left_mean','descr_left_std',
    ]

lie_feat_cols = [
        'subject',
        'card_class',
        'label',
        'duration',
        'react_dur',
        'point_dur',
        'descr_dur',
        'fix_freq',
        'sacc_freq',
        'right_mean',
        'right_std',
        'right_min',
        'right_max',
        'left_mean',
        'left_std',
        'left_min',
        'left_max',
        'react_fix_freq',
        'react_sacc_freq',
        'react_right_mean',
        'react_right_std',
        'react_right_min',
        'react_right_max',
        'react_left_mean',
        'react_left_std',
        'react_left_min',
        'react_left_max',
        'point_fix_freq',
        'point_sacc_freq',
        'point_right_mean',
        'point_right_std',
        'point_right_min',
        'point_right_max',
        'point_left_mean',
        'point_left_std',
        'point_left_min',
        'point_left_max',
        'descr_fix_freq',
        'descr_sacc_freq',
        'descr_right_mean',
        'descr_right_std',
        'descr_right_min',
        'descr_right_max',
        'descr_left_mean',
        'descr_left_std',
        'descr_left_min',
        'descr_left_max',
    ]

tobii_quality_columns = [
    'subject',
    'subject_card',
    'card',
    'whole_right',
    'point_right',
    'react_right',
    'point_react_right',
    'descr_right',
    'whole_left',
    'point_left',
    'react_left',
    'point_react_left',
    'descr_left',
    'whole_dur',
    'point_dur',
    'react_dur',
    'point_react_dur',
    'descr_dur',
]


significant_cols = [
        'duration',
        'react_dur',
        'point_dur',
        'descr_dur',
        'right_mean',
        'right_std',
        'right_min',
        'right_max',
        'left_mean',
        'left_std',
        'left_min',
        'left_max',
        'react_right_mean',
        'react_right_std',
        'react_right_min',
        'react_right_max',
        'react_left_mean',
        'react_left_std',
        'react_left_min',
        'react_left_max',
        'point_right_mean',
        'point_right_std',
        'point_right_min',
        'point_right_max',
        'point_left_mean',
        'point_left_std',
        'point_left_min',
        'point_left_max',
        'descr_right_mean',
        'descr_right_std',
        'descr_right_min',
        'descr_right_max',
        'descr_left_mean',
        'descr_left_std',
        'descr_left_min',
        'descr_left_max',
    ]

reduced_significant_cols = [
    'descr_right_mean',
    'descr_left_mean',
    'react_right_mean',
    'react_left_mean',
    ]

tt_sign_cols = [
    'left_max',
    'left_std',
    'react_mean_pupil',
    'react_left_mean',
    'descr_left_max',
    'right_mean',
    'left_mean',
    'descr_right_max',
    'descr_left_mean',
    'descr_right_mean',
    'descr_mean_pupil'
]

tt_sign_cols_clear = [
    'left_max',
    'descr_left_max',
    'react_right_mean',
    'react_mean_pupil',
    'descr_right_max',
    'react_left_mean',
    'right_mean',
    'left_mean',
    'descr_right_mean',
    'descr_left_mean',
    'descr_mean_pupil',
]

tt_sign_cols_all = [
    'left_max',
    'left_std',
    'right_mean',
    'left_mean',
    'react_right_mean',
    'react_left_mean',
    'react_mean_pupil',
    'descr_left_mean',
    'descr_left_max',
    'descr_mean_pupil',
    'descr_right_max',
    'descr_right_mean',
    'descr_mean_pupil',   
]

descr_react_col_set = [
    'descr_right_mean',
    'descr_right_std',
    'descr_right_min',
    'descr_right_max',
    'descr_left_mean',
    'descr_left_std',
    'descr_left_min',
    'descr_left_max',
    'descr_mean_pupil',
    'react_right_mean',
    'react_right_std',
    'react_right_min',
    'react_right_max',
    'react_left_mean',
    'react_left_std',
    'react_left_min',
    'react_left_max', 
    'react_mean_pupil'    
]






sr_window = 1500
card_names = ['unicorn', 'pepper', 'minion', 'pig', 'hedge', 'aliens']

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def extractLieFeatures(subjects=[], subject_to_exclude=[], cols=lie_column_names, cards=card_names, source="frontiers", ref_to_base="time", mode="sub", plot=False, print_pupil=False, save_root="plots/V2/pupils_{}.png", save=False):

    # refer_to_base = [time, feat, None]

    features = pd.DataFrame(columns=cols)

    if(len(subjects) == 0):
        subjects = extractMinSubjectSet(source, annot_path=annotations_lie_in_temp)

    baseline_right = []
    baseline_left = []

    ref_time = ref_to_base == 'time'
    ref_features = ref_to_base == 'feat'

    subjects = [s for s in subjects if s not in subject_to_exclude]

    tobii_data_stats = []

    for sub in subjects:

        eye_df, annot_dfs, baseline, overall_eye_df, filtered_interaction_dfs, filtered_interaction__size = \
                loadLieTimeSeries(
                    sub, card_names, source="frontiers", 
                    refer_to_baseline=ref_time, refer_mode=mode,
                    clean=True, clean_mode="MAD", smooth=False)

        tobii_data_stats = tobii_data_stats + filtered_interaction__size
        
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
                print("card {} {}".format(card, sub_features.loc[sub_features['card_class'] == card]['descr_left_mean'].values[0]))
            print("=======================================")
        

        features = features.append(sub_features, ignore_index=True)
        features['subject'] = features.index // 6

        if(plot):
            fig = lie_plotTimeSeries(sub, cards, annot_dfs, overall_eye_df, filtered_interaction_dfs)
            if(save):
                fig.savefig(save_root.format(sub))
                  

    return features, baseline_right, baseline_left, tobii_data_stats

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


def calc_tobii_quality_features(tobii_features, threshold=50.0):

    df = pd.DataFrame(data=tobii_features, columns=tobii_quality_columns)

    # Expected points at 100Hz
    df['whole_exp'] = df['whole_dur'] / 10 #/1000 to seconds * 100 samples per second
    df['point_exp'] = df['point_dur'] / 10
    df['react_exp'] = df['react_dur'] / 10
    df['point_react_exp'] = df['point_react_dur'] / 10
    df['descr_exp'] = df['descr_dur'] / 10

    # Percentage of points resect to expected
    df['whole_perc_right_exp'] = (df['whole_right'] / df['whole_exp']) * 100
    df['point_perc_right_exp'] = (df['point_right'] / df['point_exp']) * 100
    df['react_perc_right_exp'] = (df['react_right'] / df['react_exp']) * 100
    df['point_react_perc_right_exp'] = (df['point_react_right'] / df['point_react_exp']) * 100
    df['descr_perc_right_exp'] = (df['descr_right'] / df['descr_exp']) * 100

    df['whole_perc_left_exp'] = (df['whole_left'] / df['whole_exp']) * 100
    df['point_perc_left_exp'] = (df['point_left'] / df['point_exp']) * 100
    df['react_perc_left_exp'] = (df['react_left'] / df['react_exp']) * 100
    df['point_react_perc_left_exp'] = (df['point_react_left'] / df['point_react_exp']) * 100
    df['descr_perc_left_exp'] = (df['descr_left'] / df['descr_exp']) * 100

    # ====================================================================================

    # Expected points at 100Hz
    df['whole_right_mean'] = df['whole_right'].mean()
    df['point_right_mean'] = df['point_right'].mean()
    df['react_right_mean'] = df['react_right'].mean()
    df['point_right_react_mean'] = df['point_react_right'].mean()
    df['descr_right_mean'] = df['descr_right'].mean()

    df['whole_left_mean'] = df['whole_left'].mean()
    df['point_left_mean'] = df['point_left'].mean()
    df['react_left_mean'] = df['react_left'].mean()
    df['point_left_react_mean'] = df['point_react_left'].mean()
    df['descr_left_mean'] = df['descr_left'].mean()

    # Percentage of points resect to mean
    df['whole_perc_right_mean'] = (df['whole_right'] / df['whole_right_mean']) * 100
    df['point_perc_right_mean'] = (df['point_right'] / df['whole_right_mean']) * 100
    df['react_perc_right_mean'] = (df['react_right'] / df['whole_right_mean']) * 100
    df['point_react_perc_right_mean'] = (df['point_react_right'] / df['whole_right_mean']) * 100
    df['descr_perc_right_mean'] = (df['descr_right'] / df['whole_right_mean']) * 100

    df['whole_perc_left_mean'] = (df['whole_left'] / df['descr_left_mean']) * 100
    df['point_perc_left_mean'] = (df['point_left'] / df['descr_left_mean']) * 100
    df['react_perc_left_mean'] = (df['react_left'] / df['descr_left_mean']) * 100
    df['point_react_perc_left_mean'] = (df['point_react_left'] / df['descr_left_mean']) * 100
    df['descr_perc_left_mean'] = (df['descr_left'] / df['descr_left_mean']) * 100

    # ============================================================================

    return df


def filter_by_quality(df, thresh_target, thresh_other):

    #df_filtered = df.query("(subject_card == card & (point_perc_right_exp < @thresh_target | react_perc_right_exp < @thresh_target | descr_perc_right_exp < @thresh_target |   point_perc_left_exp < @thresh_target | react_perc_left_exp < @thresh_target | descr_perc_left_exp < @thresh_target)) | (subject_card == card & (point_perc_right_exp < @thresh_other | react_perc_right_exp < @thresh_other | descr_perc_right_exp < @thresh_other |   point_perc_left_exp < @thresh_other | react_perc_left_exp < @thresh_other | descr_perc_left_exp < @thresh_other))")


    df_filtered = df.query("descr_perc_right_exp < @thresh_target |  descr_perc_left_exp < @thresh_target")

    #df['exclude'] = df['lower_exp_right'] or df['lower_exp_left'] or df['lower_mean_right'] or df['lower_mean_left']
        
    sub_to_exclude = df_filtered.groupby('subject').max().index.values

    return sub_to_exclude


#lie_plotComparBars(lie_features, save=True)
#lie_plotBySubject(lie_features, mode=mode, save=True)
#lie_plotTnTratioMean(lie_features, save=False)

#quest_ans = preprocessQuest("data/personality_raw.csv")
#tnt_scores, subjects = coumpute_TnT_scores(lie_features, lie_feat_cols, "right_mean", abs_ratio=False)
#quest_ans = quest_ans.loc[quest_ans['subject'].isin(subjects)]

#tnt_scores = tnt_scores.reset_index()
#quest_ans = quest_ans.reset_index()
#quest_ans = quest_ans.drop(['index', 'subject'], axis=1)

#qa = quest_ans[['histrionic', 'extraversion','agreeableness ','conscientiousness ','neuroticism ','openness','narci' ]]
#qa = quest_ans[['histrionic']]
#qa = quest_ans[['NARS_S1','NARS_S2','NARS_S3','NARS_TOTAL']]
#qa = quest_ans[['MAC_Negative_Interpersonal_Tactics','MAC_Positive_Interpersonal_Tactics','MAC_Cynical_view_human_nature','MAC_Positive_view_human_nature','MAC_TOTAL']]
#qa =quest_ans[['PSY_Primary_Psychopathy','PSY_Secondary_Psychopathy','PSY_TOTAL']]

"""
goods = []

for sc in ["premed_index", 'descr_ratio', 'react_ratio']:
    for q in quest_ans.columns:
        qa = quest_ans[[q]]
        intersect, slope, adj, p = linear_regression_quest_scores(tnt_scores, qa, score=sc)
        if(p < 0.05):
            goods.append((intersect, slope, q, sc, adj, p))
            #print("{} {} {} {}".format(q, sc, adj, p))

for (intersect, slope, q, sc, adj, p) in goods:
    print("{} {} {} {} {}".format(q, sc, slope[1], adj, p))
"""

"""
for col in quest_ans.columns:
    print("================================= {} ==================================".format(col))
    qa = quest_ans[[col]]
    linear_regression_quest_scores(tnt_scores, qa, score="premed_index")
"""
"""
qa = quest_ans[['subject', 'histrionic']]
intersect, slope, adj, p = linear_regression_quest_scores(tnt_scores, qa, score="descr_ratio")
lie_plotQuestRegression(lie_features, qa, slope[1], intersect, index_col="descr_ratio")

for col in ['right_mean']:
    #lie_plotTnTratioBySubject(lie_features, feature=col, save=True)
    lie_plotTnTPremedIndex(lie_features, feature=col, save=True)

"""

#%matplotlib notebook
plt.show()
