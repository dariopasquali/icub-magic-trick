import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

from eye_feature_tools import *
from evaluation import *

lie_feat_cols = [
        'subject',
        #'source',
        'card_class',
        #'show_order',
        #'duration',
        #'react_dur',
        #'point_dur',
        #'descr_dur',
        #'fix_freq',
        #'sacc_freq',
        'right_mean',
        #'right_std',
        #'right_min',
        #'right_max',
        'left_mean',
        #'left_std',
        #'left_min',
        #'left_max',
        #'react_fix_freq',
        #'react_sacc_freq',
        'react_right_mean',
        #'react_right_std',
        #'react_right_min',
        #'react_right_max',
        'react_left_mean',
        #'react_left_std',
        #'react_left_min',
        #'react_left_max',
        #'point_fix_freq',
        #'point_sacc_freq',
        'point_right_mean',
        #'point_right_std',
        #'point_right_min',
        #'point_right_max',
        'point_left_mean',
        #'point_left_std',
        #'point_left_min',
        #'point_left_max',
        #'descr_fix_freq',
        #'descr_sacc_freq',
        'descr_right_mean',
        #'descr_right_std',
        #'descr_right_min',
        #'descr_right_max',
        'descr_left_mean',
        #'descr_left_std',
        #'descr_left_min',
        #'descr_left_max',
        'label'
    ]

def lie_plotBySubject(features, mode, feat_cols=lie_feat_cols, save_root="plots/LIE/points_{}.png", save=True):
    
    markers=[',','1','v','^','<','>','8','s','p','P','*','h','H','+','x',
        'X','D','d','|','_','o','2','3','4','.']

    aggrZeros, aggrOnes, TnT = aggregate_target_nontarget(features, lie_feat_cols)

    sub_z = aggrZeros['subject'].values
    sub_o = aggrOnes['subject'].values

    subjects = [int(s) for s in sub_z if s in sub_o]
    subjects
    
    plot_cols = feat_cols.copy()
    plot_cols.remove('subject')
    plot_cols.remove('card_class')

    for f in plot_cols:
        fig, axs = plt.subplots(1, figsize=(9, 9), num="x{}".format(f))
        labels = []

        pallX = aggrZeros[f].mean(skipna=True)
        pallX_ste = aggrZeros[f].sem(skipna=True)    
        pallY = aggrOnes[f].mean(skipna=True)
        pallY_ste = aggrOnes[f].sem(skipna=True)

        for i, sub in enumerate(subjects):

            axs.set_title("{}".format(f))
            axs.set_xlabel("Not Target avg")
            axs.set_ylabel("Target")
            axs.set_label("{}".format(sub))

            if(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "pig"):
                size=200
                color = "r"
                labels.append("{}".format(sub))
            elif(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "unicorn"):
                size=100
                color = "b"
                labels.append("{}".format(sub))
            elif(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "pepper"):
                size=100
                color = "k"
                labels.append("{}".format(sub))
            elif(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "minion"):
                size=100
                color = "g"
                labels.append("{}".format(sub))
            elif(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "hedge"):
                size=100
                color = "c"
                labels.append("{}".format(sub))
            elif(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "aliens"):
                size=100
                color = "y"
                labels.append("{}".format(sub))

            axs.scatter(aggrZeros.loc[aggrZeros['subject'] == sub][f],
                        aggrOnes.loc[aggrOnes['subject'] == sub][f],
                        s=size,
                        c=color,
                        marker=markers[i]) 


        minX = aggrZeros[f].min()
        maxX = aggrZeros[f].max()        
        minY = aggrOnes[f].min()
        maxY = aggrOnes[f].max()       
        minXY = min(minX, minY)
        maxXY = max(maxX, maxY)        

        mn = (maxXY + minXY) / 20

        minXY = minXY - mn
        maxXY = maxXY + mn

        lims = [minXY, maxXY]

        axs.set_xlim(minXY, maxXY)
        axs.set_ylim(minXY, maxXY)
        axs.set_aspect('equal')        

        plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))    

        axs.scatter(pallX, pallY, s=200, alpha=0.7)
        axs.errorbar(pallX, pallY,
                xerr=pallX_ste,
                yerr=pallY_ste)

        axs.plot(lims, lims, alpha=0.75, zorder=100)
        if(save):
            fig.savefig(save_root.format(f))

def lie_plotTnTratioMean(features, save=True):

    feat_comparison = [
        #('duration', 'react_dur', 'point_dur', 'descr_dur'),
        #('fix_freq', 'point_fix_freq', 'react_fix_freq', 'descr_fix_freq'),
        #('sacc_freq', 'point_sacc_freq', 'react_sacc_freq', 'descr_sacc_freq'),
        ('right_mean', 'point_right_mean', 'react_right_mean', 'descr_right_mean'),
        #('right_std', 'point_right_std', 'react_right_std', 'descr_right_std'),
        #('right_min', 'point_right_min', 'react_right_min', 'descr_right_min'),
        #('right_max', 'point_right_max', 'react_right_max', 'descr_right_max'),
        ('left_mean', 'point_left_mean', 'react_left_mean', 'descr_left_mean'),
        #('left_std', 'point_left_std', 'react_left_std', 'descr_left_std'),
        #('left_min', 'point_left_min', 'react_left_min', 'descr_left_min'),
        #('left_max', 'point_left_max', 'react_left_max', 'descr_left_max')
    ]

    nonTargets = pd.DataFrame(columns=lie_feat_cols)
    targets = pd.DataFrame(columns=lie_feat_cols)

    subjects = features.groupby('subject').count().index.values

    for sub in subjects:
        nonTargets, targets = extract_target_nontarget(features, sub, lie_feat_cols, nonTargets, targets)    

    labels = ['point', 'reaction', 'description']
    x_pos = np.arange(len(labels))

    fig, axs = plt.subplots(1, figsize=(15, 10), num='TnT ratio')

    for (feat_name, point, react, descr) in feat_comparison:

        nonTarget_react_mean = nonTargets[react].mean(skipna=True)
        nonTarget_point_mean = nonTargets[point].mean(skipna=True)
        nonTarget_descr_mean = nonTargets[descr].mean(skipna=True)

        target_react_mean = targets[react].mean(skipna=True)
        target_point_mean = targets[point].mean(skipna=True)
        target_descr_mean = targets[descr].mean(skipna=True)
        
        tnt_point = np.abs(target_point_mean - nonTarget_point_mean)
        tnt_react = np.abs(target_react_mean - nonTarget_react_mean)
        tnt_descr = np.abs(target_descr_mean - nonTarget_descr_mean)

        #tnt_point = (target_point_mean - nonTarget_point_mean)
        #tnt_react = (target_react_mean - nonTarget_react_mean)
        #tnt_descr = (target_descr_mean - nonTarget_descr_mean)

        y_values = [ tnt_point, tnt_react, tnt_descr]

        
        axs.plot(labels, y_values, '-o', label=feat_name)

    axs.legend()
    return fig

def lie_plotTnTratioBySubject(features, feature, save_root="plots/LIE/profile_{}.png", save=True):

    #feat_comparison_R = (0, 'right_mean', 'point_right_mean', 'react_right_mean', 'descr_right_mean')

    fig, axs = plt.subplots(2, figsize=(15, 10), num='TnT {}'.format(feature))
    labels = ['point', 'reaction', 'description']

    tnt_scores, subjects = coumpute_TnT_scores(features, lie_feat_cols, feature, abs_ratio=False)
    tnt_scores_norm, subjects = coumpute_TnT_scores(features, lie_feat_cols, feature, abs_ratio=False, norm_to_point=True)

    for sub in subjects:
        score = tnt_scores.loc[tnt_scores['subject'] == sub]
        score_norm = tnt_scores_norm.loc[tnt_scores_norm['subject'] == sub]
        
        point = score["point_ratio"].values[0]
        react = score["react_ratio"].values[0]
        descr = score["descr_ratio"].values[0]

        point_norm = score_norm["point_ratio"].values[0]
        react_norm = score_norm["react_ratio"].values[0]
        descr_norm = score_norm["descr_ratio"].values[0]

        y_values = [ point, react, descr]
        y_values_norm = [ point_norm, react_norm, descr_norm]

        axs[0].plot(labels, y_values, '-o', label=sub)
        axs[1].plot(labels, y_values_norm, '-o', label=sub)

    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #axs[1].legend(loc='center right', bbox_to_anchor=(1, 0.5))
    axs[0].set_title("Diff {} T and nonT".format(feature))
    axs[1].set_title("Diff {} norm to POINT".format(feature))

    if(save):
        fig.savefig(save_root.format(feature))
    
    return fig

def lie_plotTnTstem(features, feature, save_root="plots/LIE/Stem_{}.png", save=True):

    fig, axs = plt.subplots(2, figsize=(15, 10), num='STEM TnT {}'.format(feature))

    tnt_scores, subjects = coumpute_TnT_scores(features, lie_feat_cols, feature, abs_ratio=False)
    tnt_scores_norm, subjects = coumpute_TnT_scores(features, lie_feat_cols, feature, abs_ratio=False, norm_to_point=True)
    

    tnt_scores = tnt_scores.sort_values(by=['descr_ratio'])
    tnt_scores_norm = tnt_scores_norm.sort_values(by=['descr_ratio'])

    labels = [str(lab) for lab in tnt_scores['subject'].values]
    labels_norm = [str(lab) for lab in tnt_scores_norm['subject'].values]

    x_pos = np.arange(len(labels))
    x_pos_norm = np.arange(len(labels_norm))

    axs[0].stem(x_pos, tnt_scores['descr_ratio'], linefmt='green', markerfmt='X', label="DESCR")
    axs[0].stem(x_pos, tnt_scores['react_ratio'], linefmt='grey', markerfmt='D', label="REACT")
    axs[0].set_xticks(x_pos)
    axs[0].set_xticklabels(labels)

    axs[1].stem(x_pos_norm, tnt_scores_norm['descr_ratio'], linefmt='green', markerfmt='X', label="DESCR")
    axs[1].stem(x_pos_norm, tnt_scores_norm['react_ratio'], linefmt='grey', markerfmt='D', label="REACT")
    axs[1].set_xticks(x_pos_norm)
    axs[1].set_xticklabels(labels_norm)

    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("Diff {} T and nonT".format(feature))
    axs[1].set_title("Diff {} norm to POINT".format(feature))
    #axs[1].legend(loc='center right', bbox_to_anchor=(1, 0.5))

    if(save):
        fig.savefig(save_root.format(feature))
    
    return fig

# Average Histogram of Target (old) vs nonTarget (new) for each feature
def lie_plotComparBars(features, save_root="plots/LIE/Bars_{}.png", save=True):

    feat_comparison = [
        #('duration', 'react_dur', 'point_dur', 'descr_dur'),
        #('fix_freq', 'point_fix_freq', 'react_fix_freq', 'descr_fix_freq'),
        #('sacc_freq', 'point_sacc_freq', 'react_sacc_freq', 'descr_sacc_freq'),
        ('right_mean', 'point_right_mean', 'react_right_mean', 'descr_right_mean'),
        #('right_std', 'point_right_std', 'react_right_std', 'descr_right_std'),
        #('right_min', 'point_right_min', 'react_right_min', 'descr_right_min'),
        #('right_max', 'point_right_max', 'react_right_max', 'descr_right_max'),
        ('left_mean', 'point_left_mean', 'react_left_mean', 'descr_left_mean'),
        #('left_std', 'point_left_std', 'react_left_std', 'descr_left_std'),
        #('left_min', 'point_left_min', 'react_left_min', 'descr_left_min'),
        #('left_max', 'point_left_max', 'react_left_max', 'descr_left_max')
    ]

    nonTargets, targets, TnT = aggregate_target_nontarget(features, lie_feat_cols)

    bar_width = 0.25   
    labels = ['point', 'react', 'descr']
    x_pos = np.arange(len(labels))

    for (whole, point, react, descr) in feat_comparison:

        fig, axs = plt.subplots(1, figsize=(15, 10), num='{}'.format(whole))
        fig.suptitle('{}'.format(whole), fontsize=16)

        nT_react_mean = nonTargets[react].mean(skipna=True)
        nT_point_mean = nonTargets[point].mean(skipna=True)
        nT_descr_mean = nonTargets[descr].mean(skipna=True)
        
        nT_react_ste = nonTargets[react].sem(skipna=True)
        nT_point_ste = nonTargets[point].sem(skipna=True)
        nT_descr_ste = nonTargets[descr].sem(skipna=True)

        T_react_mean = targets[react].mean(skipna=True)
        T_point_mean = targets[point].mean(skipna=True)
        T_descr_mean = targets[descr].mean(skipna=True)
        
        T_react_ste = targets[react].sem(skipna=True)
        T_point_ste = targets[point].sem(skipna=True)
        T_descr_ste = targets[descr].sem(skipna=True)        


        y_values_T = [ T_point_mean, T_react_mean, T_descr_mean]
        y_values_nT = [ nT_point_mean, nT_react_mean, nT_descr_mean]
        T_errors = [ T_point_ste, T_react_ste, T_descr_ste]
        nT_errors = [ nT_point_ste, nT_react_ste, nT_descr_ste] 
            
        x_pos = np.arange(len(labels))
        axs.bar(x_pos, y_values_T,
                yerr=T_errors,
                color='g', align='center',
                width=bar_width, label='Target (OLD)', alpha=0.8)

        axs.bar(x_pos+bar_width, y_values_nT,
                yerr=nT_errors,
                color='b', align='center',
                width=bar_width, label='nonTarget (NEW)', alpha=0.8)
        axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs.set_xticks(x_pos)
        axs.set_xticklabels(labels)
        axs.set_title('{}'.format(whole))

        if(save):
            fig.savefig(save_root.format(whole))

def lie_plotTimeSeries(subject, card_names, annotations, overall_eye, filtered_inter_dfs, baseline=None):
    
    subject_card = annotations[0]['subject_card'].values[0]
    cards_plot_features = []

    for i in range(1, len(annotations)):    
        cards_plot_features.append([
            
            annotations[i]['start_p'].min(), #start pointing
            annotations[i]['stop_p'].max(), #touch card
            annotations[i]['start_d'].min(), #start descr
            annotations[i]['stop_d'].max(), #stop descr
            annotations[i]['card'].values[0]
            
        ])

    cards_plot_features.sort() 

    fig, ass = plt.subplots(2, figsize=(15, 10))
        
    ass[0].set_title("RIGHT {}".format(subject))
    ass[1].set_title("LEFT {}".format(subject))
        
    ass[0].scatter(overall_eye['timestamp'],overall_eye['diam_right'], s=1, c='r')
    ass[1].scatter(overall_eye['timestamp'],overall_eye['diam_left'], s=1, c='b')

    if(baseline != None):
        start_base = baseline['timestamp'].min()
        stop_base = baseline['timestamp'].max() 
        # plot baseline
        ass[0].scatter(baseline['timestamp'],baseline['diam_right'], s=1, c='r')
        ass[1].scatter(baseline['timestamp'],baseline['diam_left'], s=1, c='b')


    for sections in cards_plot_features:    
        alp = 0.2
        color = 'green'
        
        if(sections[4] == subject_card):
            alp = 0.3
            color = 'yellow'
                
        ass[0].axvspan(sections[0], sections[1],
                        linewidth=1, color=color, alpha=alp)
        ass[0].axvspan(sections[2], sections[3],
                        linewidth=1, color=color, alpha=alp)
            
        ass[1].axvspan(sections[0], sections[1],
                        linewidth=1, color=color, alpha=alp)
        ass[1].axvspan(sections[2], sections[3],
                        linewidth=1, color=color, alpha=alp)

        if(baseline != None):
            #color the baseline
            ass[0].axvspan(start_base, stop_base,
                            linewidth=1, color="red", alpha=0.1)
            ass[1].axvspan(start_base, stop_base,
                            linewidth=1, color="red", alpha=0.1)
    
    
    return fig


