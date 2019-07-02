import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

from eye_feature_tools import *

plot_column_names = [
        'subject',
        'duration',
        'card_class',
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


def plotPupilDilationTimeSeries(subject, card_names, subject_card, overall_eye, cards_eye):
    
    colors = ['green', 'yellow', 'red', 'purple', 'grey', 'pink']
    cards_plot_features = []
    
    for i in range(0, len(cards_eye)):    
        cards_plot_features.append([
            cards_eye[i]['timestamp'].min(),
            cards_eye[i]['timestamp'].max(),
            card_names[i]
        ])    
        
    cards_plot_features.sort()    
    fig, ass = plt.subplots(3, figsize=(15, 10))
    
    ass[0].set_title("RIGHT {}".format(subject))
    ass[1].set_title("LEFT {}".format(subject))
    
    ass[0].scatter(overall_eye['timestamp'],overall_eye['diam_right'], s=1, c='r')
    ass[1].scatter(overall_eye['timestamp'],overall_eye['diam_left'], s=1, c='b')

    ass[2].scatter(overall_eye['timestamp'],overall_eye['diam_right'], s=1, c='r')
    ass[2].scatter(overall_eye['timestamp'],overall_eye['diam_left'], s=1, c='b')

    for i in range(0, len(cards_eye)):    
        alp = 0.2
        color = 'green'
        if(cards_plot_features[i][2] == subject_card):
            alp = 0.3
            color = 'yellow'
            
        ass[0].axvspan(cards_plot_features[i][0],
                       cards_plot_features[i][1],
                       linewidth=1, color=color, alpha=alp)
        
        ass[1].axvspan(cards_plot_features[i][0],
                       cards_plot_features[i][1],
                       linewidth=1, color=color, alpha=alp)

    return fig

def plotBySubject(features, mode, feat_cols=plot_column_names, save=True):
    
    plot_cols = feat_cols[1:]

    aggrZeros = pd.DataFrame(columns=plot_cols)
    aggrOnes = pd.DataFrame(columns=plot_cols)

    markers=[',','1','v','^','<','>','8','s','p','P','*',
    'h','H','+','x','X','D','d','|','_','o','2','3','4','.']
    subjects = features.groupby('subject').count().index.values

    for sub in subjects:
        aggrZeros, aggrOnes = aggregate_target_nontarget(features, sub, plot_cols, aggrZeros, aggrOnes)    

    sub_z = aggrZeros['subject'].values
    sub_o = aggrOnes['subject'].values

    subjects = [int(s) for s in sub_z if s in sub_o]
    subjects
    
    for f in feat_cols:
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

def aggregateByCard(features, card, aggrZeros, aggrOnes):
    data = features[plot_column_names].loc[features['card_class'] == card]
    
    zeros = data.loc[features['label'] == 0].drop(['card_class'], axis=1)
    ones = data.loc[features['label'] == 1].drop(['card_class'], axis=1)

    w = len(ones.index)
    
    for col in zeros.columns:
        zeros[col] = zeros[col].mean()
        ones[col] = ones[col].mean()
        
    zeros = zeros.head(1)
    zeros['card_class'] = card
    
    ones = ones.head(1)
    ones['card_class'] = card

    aggrZeros = aggrZeros.append(zeros, ignore_index=True)
    aggrOnes = aggrOnes.append(ones, ignore_index=True)
    
    return aggrZeros, aggrOnes, w
    
def plotByCards(features, feat_cols, card_names, mode):
    
    markers=[',','1','v','^','<','>','8','s','p','P','*','h','H','+','x','X','D','d','|','_','o','2','3','4','.']
    
    cZ = pd.DataFrame(columns=plot_column_names)
    cO = pd.DataFrame(columns=plot_column_names)
    weights = []

    for card in card_names:
        cZ, cO, w= aggregateByCard(features, card, cZ, cO)
        weights.append(w)

    card_weights = dict(zip(card_names, weights))

    restrict_card_names = [c for c in cZ['card_class'].values if c in cO['card_class'].values]
    card_names = restrict_card_names
    
    for f in feat_cols:
        fig2, axs2 = plt.subplots(1, figsize=(9, 9))
        labels = []

        for c, card in enumerate(card_names):

            if(card == "pig"):
                color = "r"
            elif(card == "unicorn"):
                color = "b"
            elif(card == "minion"):
                color = "g"
            elif(card == "aliens"):
                color = "y"
            elif(card == "pepper"):
                color = "k"
            elif(card == "hedge"):
                color = "c"

            axs2.set_title("{}".format(f))
            axs2.set_xlabel("Not Target avg")
            axs2.set_ylabel("Target")
            #axs2.set_label("{}".format(sub))

            size = card_weights[card] * 50            
            axs2.scatter(cZ.loc[cZ['card_class'] == card][f],
                        cO.loc[cO['card_class'] == card][f],
                        s=size,
                        c=color,
                        marker=markers[c])    

        minX = cZ[f].min()
        maxX = cZ[f].max()        
        minY = cO[f].min()
        maxY = cO[f].max()       
        minXY = min(minX, minY)
        maxXY = max(maxY, maxY)        

        mn = (maxXY + minXY) / 50

        minXY = minXY - mn
        maxXY = maxXY + mn

        lims = [minXY, maxXY]

        axs2.set_xlim(minXY, maxXY)
        axs2.set_ylim(minXY, maxXY)
        axs2.set_aspect('equal')

        plt.legend(card_names, loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'large')

        axs2.plot(lims, lims, alpha=0.75, zorder=100)
        fig2.savefig("plots/{}/card/nTvsT_{}".format(mode, f))

# Average Histogram of Target (old) vs nonTarget (new) for each feature
def plotComparisonHistogram(features, mode, save=False):
    feat_comparison = [
        ('fix_freq', 'sre_fix_freq', 'srl_fix_freq'),
        ('sacc_freq', 'sre_sacc_freq', 'srl_sacc_freq'),
        ('pd_right_mean', 'sre_pd_right_mean', 'srl_pd_right_mean'),
        ('pd_right_std', 'sre_pd_right_std', 'srl_pd_right_std'),
        ('pd_right_min', 'sre_pd_right_min', 'srl_pd_right_min'),
        ('pd_right_max', 'sre_pd_right_max', 'srl_pd_right_max'),
        ('pd_left_mean', 'sre_pd_left_mean', 'srl_pd_left_mean'),
        ('pd_left_std', 'sre_pd_left_std', 'srl_pd_left_std'),
        ('pd_left_min', 'sre_pd_left_min', 'srl_pd_left_min'),
        ('pd_left_max', 'sre_pd_left_max', 'srl_pd_left_max')
    ]

    labels = ['all', 'early', 'late']
    bar_width = 0.35

    aggrZeros = pd.DataFrame(columns=plot_column_names)
    aggrOnes = pd.DataFrame(columns=plot_column_names)

    subjects = features.groupby('subject').count().index.values

    for sub in subjects:
        aggrZeros, aggrOnes = aggregate_target_nontarget(features, sub, plot_column_names, aggrZeros, aggrOnes)    


    for (whole, early, late) in feat_comparison:

        fig, axs = plt.subplots(1, figsize=(10, 5))

        new_all_mean = aggrZeros[whole].mean()
        new_early_mean = aggrZeros[early].mean()
        new_late_mean = aggrZeros[late].mean()

        new_all_ste = aggrZeros[whole].sem()
        new_early_ste = aggrZeros[early].sem()
        new_late_ste = aggrZeros[late].sem()

        old_all_mean = aggrOnes[whole].mean()
        old_early_mean = aggrOnes[early].mean()
        old_late_mean = aggrOnes[late].mean()

        old_all_ste = aggrOnes[whole].sem()
        old_early_ste = aggrOnes[early].sem()
        old_late_ste = aggrOnes[late].sem()

        x_pos = np.arange(len(labels))

        y_values_old = [old_all_mean, old_early_mean, old_late_mean]
        y_values_new = [new_all_mean, new_early_mean, new_late_mean]
        new_errors = [new_all_ste, new_early_ste, new_late_ste]
        old_errors = [old_all_ste, old_early_ste, old_late_ste]

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
            fig.savefig("plots/{}/hist/{}".format(mode, whole))
