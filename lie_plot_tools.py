import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

from eye_feature_tools import *

lie_feat_cols = [
        'subject',
        #'source',
        'card_class',
        #'show_order',
        'duration',
        'react_dur',
        'point_react_dur',
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
        'point_react_fix_freq',
        'point_react_sacc_freq',
        'point_react_right_mean',
        'point_react_right_std',
        'point_react_right_min',
        'point_react_right_max',
        'point_react_left_mean',
        'point_react_left_std',
        'point_react_left_min',
        'point_react_left_max',
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
        'label'
    ]

def lie_plotBySubject(features, mode, feat_cols=lie_feat_cols, save=True):
    
    aggrZeros = pd.DataFrame(columns=lie_feat_cols)
    aggrOnes = pd.DataFrame(columns=lie_feat_cols)

    markers=[',','1','v','^','<','>','8','s','p','P','*','h','H','+','x',
        'X','D','d','|','_','o','2','3','4','.']

    subjects = features.groupby('subject').count().index.values

    for sub in subjects:
        aggrZeros, aggrOnes = aggregate_target_nontarget(features, sub, feat_cols, aggrZeros, aggrOnes)    

    sub_z = aggrZeros['subject'].values
    sub_o = aggrOnes['subject'].values

    subjects = [int(s) for s in sub_z if s in sub_o]
    subjects
    
    plot_cols = feat_cols.copy()
    plot_cols.remove('subject')
    plot_cols.remove('card_class')

    for f in plot_cols:
        fig, axs = plt.subplots(1, figsize=(9, 9))
        labels = []

        pallX = aggrZeros[f].mean()
        pallX_ste = aggrZeros[f].sem()    
        pallY = aggrOnes[f].mean()
        pallY_ste = aggrOnes[f].sem()

        for i, sub in enumerate(subjects):

            if(sub == 4):
                continue

            axs.set_title("{}".format(f))
            axs.set_xlabel("Not Target avg")
            axs.set_ylabel("Target")
            axs.set_label("{}".format(sub))

            if(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "pig"):
                size=200
                color = "r"
                labels.append("pig{}".format(sub))
            elif(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "unicorn"):
                size=100
                color = "b"
                labels.append("uni{}".format(sub))
            elif(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "pepper"):
                size=100
                color = "k"
                labels.append("pep{}".format(sub))
            elif(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "minion"):
                size=100
                color = "g"
                labels.append("min{}".format(sub))
            elif(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "hedge"):
                size=100
                color = "c"
                labels.append("hed{}".format(sub))
            elif(aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0] == "aliens"):
                size=100
                color = "y"
                labels.append("ali{}".format(sub))

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
            fig.savefig("plots/{}/subject/nTvsT_{}".format(mode, f))


    plt.show()


# Average Histogram of Target (old) vs nonTarget (new) for each feature
def lie_plotComparBars(features, baseline_right, baseline_left, mode, save=True):

    feat_comparison = [
        ('fix_freq', 'react_fix_freq', 'point_react_fix_freq', 'descr_fix_freq'),
        ('sacc_freq', 'react_sacc_freq', 'point_react_sacc_freq', 'descr_sacc_freq'),
        ('right_mean', 'react_right_mean', 'point_react_right_mean', 'descr_right_mean'),
        ('right_std', 'react_right_std', 'point_react_right_std', 'descr_right_std'),
        ('right_min', 'react_right_min', 'point_react_right_min', 'descr_right_min'),
        ('right_max', 'react_right_max', 'point_react_right_max', 'descr_right_max'),
        ('left_mean', 'react_left_mean', 'point_react_left_mean', 'descr_left_mean'),
        ('left_std', 'react_left_std', 'point_react_left_std', 'descr_left_std'),
        ('left_min', 'react_left_min', 'point_react_left_min', 'descr_left_min'),
        ('left_max', 'react_left_max', 'point_react_left_max', 'descr_left_max')
    ]

    labels = ['all', 'react', 'point_react', 'descr']
    bar_width = 0.25

    aggrZeros = pd.DataFrame(columns=lie_feat_cols)
    aggrOnes = pd.DataFrame(columns=lie_feat_cols)

    subjects = features.groupby('subject').count().index.values

    for sub in subjects:
        aggrZeros, aggrOnes = aggregate_target_nontarget(features, sub, lie_feat_cols, aggrZeros, aggrOnes)    

    plot_cols = lie_feat_cols.copy()
    plot_cols.remove('subject')
    plot_cols.remove('card_class')

    br_mean = np.mean(baseline_right)

    for (whole, react, point_react, descr) in feat_comparison:

        fig, axs = plt.subplots(1, figsize=(10, 5))

        new_all_mean = aggrZeros[whole].mean()
        new_react_mean = aggrZeros[react].mean()
        new_point_react_mean = aggrZeros[point_react].mean()
        new_descr_mean = aggrZeros[descr].mean()
        
        new_all_ste = aggrZeros[whole].sem(axis = 0)
        new_react_ste = aggrZeros[react].sem(axis = 0)
        new_point_react_ste = aggrZeros[point_react].sem(axis = 0)
        new_descr_ste = aggrZeros[descr].sem(axis = 0)

        old_all_mean = aggrOnes[whole].mean()
        old_react_mean = aggrOnes[react].mean()
        old_point_react_mean = aggrOnes[point_react].mean()
        old_descr_mean = aggrOnes[descr].mean()
        
        old_all_ste = aggrOnes[whole].sem(axis = 0)
        old_react_ste = aggrOnes[react].sem(axis = 0)
        old_point_react_ste = aggrOnes[point_react].sem(axis = 0)
        old_descr_ste = aggrOnes[descr].sem(axis = 0)

        x_pos = np.arange(len(labels))

        br_mean = np.mean(baseline_right)
        bl_mean = np.mean(baseline_left)

        if(mode=="none" and "right_mean" in whole):
            y_values_old = [br_mean, old_all_mean, old_react_mean, old_point_react_mean, old_descr_mean]
            y_values_new = [br_mean, new_all_mean, new_react_mean, new_point_react_mean, new_descr_mean]
            old_errors = [0, old_all_ste, old_react_ste, old_point_react_ste, old_descr_ste]
            new_errors = [0, new_all_ste, new_react_ste, new_point_react_ste, new_descr_ste]

            labels = ['baseline', 'all', 'react', 'point_react', 'descr']

        
        elif(mode=="none" and "left_mean" in whole):
            y_values_old = [bl_mean, old_all_mean, old_react_mean, old_point_react_mean, old_descr_mean]
            y_values_new = [bl_mean, new_all_mean, new_react_mean, new_point_react_mean, new_descr_mean]
            old_errors = [0, old_all_ste, old_react_ste, old_point_react_ste, old_descr_ste]
            new_errors = [0, new_all_ste, new_react_ste, new_point_react_ste, new_descr_ste]

            labels = ['baseline', 'all', 'react', 'point_react', 'descr']

        else:
            y_values_old = [ old_all_mean, old_react_mean, old_point_react_mean, old_descr_mean]
            y_values_new = [ new_all_mean, new_react_mean, new_point_react_mean, new_descr_mean]
            old_errors = [old_all_ste, old_react_ste, old_point_react_ste, old_descr_ste]
            new_errors = [new_all_ste, new_react_ste, new_point_react_ste, new_descr_ste]

            labels = ['all', 'react', 'point_react', 'descr']
            
        x_pos = np.arange(len(labels))
        axs.bar(x_pos, y_values_old,
                yerr=old_errors,
                color='g', align='center',
                width=bar_width, label='Target (OLD)', alpha=0.8)

        axs.bar(x_pos+bar_width, y_values_new,
                yerr=new_errors,
                color='b', align='center',
                width=bar_width, label='nonTarget (NEW)', alpha=0.8)
        axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs.set_xticks(x_pos)
        axs.set_xticklabels(labels)


        axs.set_title('{}'.format(whole))
        if(save):
            fig.savefig("plots/XXX/hist/{}".format(whole))


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
