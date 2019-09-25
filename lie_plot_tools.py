import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

from eye_feature_tools import *
from evaluation import *

markers=[".",",","o","v","^","<",">",
"1","2","3","4","8","s","p","P",
"*","h","H","+","x","X","D","d","|","_",]

card_color_map = {
    "unicorn" : "#1E90FF",
    "pepper" : "#FFD700",
    "minion" : "#FF0000",
    "aliens" : "#32CD32",
    "hedge" : "#4B0082",
    "pig" : "#FF8C00"
}

marker_map = {
0 : ( ".", "#696969" ), 
1 : ( ",", "#9A6324" ), 
2 : ( "o", "#808000" ), 
3 : ( "v", "#469990" ), 
4 : ( "^", "#000075" ), 
5 : ( "<", "#000000" ), 
6 : ( ">", "#e6194B" ), 
7 : ( "1", "#f58231" ), 
8 : ( "2", "#FFD700" ), 
9 : ( "3", "#bfef45" ), 
10 : ( "4", "#3cb44b" ), 
11 : ( "8", "#42d4f4" ), 
12 : ( "s", "#4363d8" ), 
13 : ( "p", "#911eb4" ), 
14 : ( "P", "#f032e6" ), 
15 : ( "*", "#a9a9a9" ), 
16 : ( "h", "#FFDA00" ), 
17 : ( "H", "#bfef55" ), 
18 : ( "+", "#3cb4ab" ),
19 : ( "x", "#a2d4f4" ), 
20 : ( "X", "#f363d8" ), 
21 : ( "D", "#a11eb4" ), 
22 : ( "d", "#f072e6" ), 
23 : ( "|", "#a9f9a9" ), 
24 : ( "_", "#00f000" ), 
}

label_font_size = 18
legend_prop_size = {'size': 18}


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


def TNT_points(lie_features, lie_feat_cols, feature, title, abs_ratio=False, norm_to_point=False):
    tnt_scores, sss = coumpute_TnT_scores(lie_features, lie_feat_cols, feature, abs_ratio=abs_ratio, norm_to_point=norm_to_point)
    
    print(sss)

    f, axs = plt.subplots(1, figsize=(10, 10), num="{}".format(feature))
    pallX = tnt_scores['react_ratio'].mean(skipna=True)
    pallX_ste = tnt_scores['react_ratio'].sem(skipna=True)    
    pallY = tnt_scores['descr_ratio'].mean(skipna=True)
    pallY_ste = tnt_scores['descr_ratio'].sem(skipna=True)

    labels = []
    size = 200
    label_font_size = 10
    legend_prop_size = {'size': 12}

    for i, sub in enumerate(sss):

        labels.append("{}".format(sub))
        marker = marker_map[sub][0]

        axs.set_xlabel("REACT ratio {}".format(title), fontsize=label_font_size)
        axs.set_ylabel("DESCR ratio {}".format(title), fontsize=label_font_size)
        axs.set_label("{}".format(sub))
        
        color = marker_map[sub][1]
        labels.append("{}".format(sub))
        marker = marker_map[sub][0]
        
        axs.scatter(tnt_scores.loc[tnt_scores['subject'] == sub]['react_ratio'],
                            tnt_scores.loc[tnt_scores['subject'] == sub]['descr_ratio'],
                            s=size,
                            c=color,
                            marker=marker)

    minX = tnt_scores['react_ratio'].min()
    maxX = tnt_scores['react_ratio'].max()        
    minY = tnt_scores['descr_ratio'].min()
    maxY = tnt_scores['descr_ratio'].max()       
    minXY = min(minX, minY)
    maxXY = max(maxX, maxY)

    mn = (maxXY + minXY) / 20

    minXY = minXY - mn
    maxXY = maxXY + mn
    
    lims = [minXY, maxXY]
    
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), prop=legend_prop_size)  
    axs.scatter(pallX, pallY, s=200, alpha=0.5)
    axs.errorbar(pallX, pallY,
                xerr=pallX_ste,
                yerr=pallY_ste)
    
    axs.plot(lims, lims, alpha=0.75, zorder=100)
    axs.axvline(color='black', alpha=0.2)
    axs.axhline(color='black', alpha=0.2)
    axs.tick_params(axis='both', which='major', labelsize=label_font_size)


def lie_plotPointsAllSubjects(features, mode, feat_cols=[], scale=4, save_root="plots/LIE/points_{}.png", save=True):
    
    label_font_size = 18 * scale
    legend_prop_size = {'size': 13 * scale}

    cols_to_aggregate = [col for (col, title) in feat_cols]
    aggrZeros, aggrOnes, TnT = aggregate_target_nontarget(features, cols_to_aggregate)

    sub_z = aggrZeros['subject'].values
    sub_o = aggrOnes['subject'].values

    subjects = [int(s) for s in sub_z if s in sub_o]
    subjects
    
    plot_cols = feat_cols.copy()
    plot_cols = [(f, t) for (f, t) in plot_cols if f not in ('subject', 'card_class', 'label')]

    #plot_cols = ["descr_right_mean"]

    for (f, title) in plot_cols:
        fig, axs = plt.subplots(1, figsize=(10 * scale, 10 * scale), num="x{}".format(f))
        print(f)
        labels = []

        pallX = aggrZeros[f].mean(skipna=True)
        pallX_ste = aggrZeros[f].sem(skipna=True)    
        pallY = aggrOnes[f].mean(skipna=True)
        pallY_ste = aggrOnes[f].sem(skipna=True)

        for i, sub in enumerate(subjects):
            #print(sub)

            #axs.set_xlabel("Right Mean Pupil Dilation Average Non Target", fontsize=label_font_size)
            #axs.set_ylabel("Right Mean Pupil Dilation Target", fontsize=label_font_size)
            axs.set_xlabel("{} Average Non Target".format(title), fontsize=label_font_size)
            axs.set_ylabel("{} Target".format(title), fontsize=label_font_size)
            axs.set_label("{}".format(sub))

            color = card_color_map[aggrOnes.loc[aggrOnes['subject'] == sub]["card_class"].values[0]]
            size = 1000 * scale
            labels.append("{}".format(sub))
            marker = marker_map[sub][0]
            
            axs.scatter(aggrZeros.loc[aggrZeros['subject'] == sub][f],
                        aggrOnes.loc[aggrOnes['subject'] == sub][f],
                        s=size,
                        c=color,
                        marker=marker, linewidth=10) 


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

        plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), prop=legend_prop_size)    

        axs.scatter(pallX, pallY, s=1000*scale, alpha=0.5, linewidth=10)
        axs.errorbar(pallX, pallY,
                xerr=pallX_ste,
                yerr=pallY_ste, linewidth=10)

        axs.plot(lims, lims, alpha=0.75, zorder=100, linewidth=10)
        axs.axvline(color='black', alpha=0.2, linewidth=10)
        axs.axhline(color='black', alpha=0.2, linewidth=10)

        axs.tick_params(axis='both', which='major', labelsize=label_font_size)

        if(save):
            fig.savefig(save_root.format(f), dpi=100)

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

    fig, axs = plt.subplots(1, figsize=(15, 10), num='TnT {}'.format(feature))
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

        marker = '-' + marker_map[sub][0]
        color = marker_map[sub][1]

        axs.plot(labels, y_values, marker, label=sub, markersize=15, color=color)
        #axs[1].plot(labels, y_values_norm, '-o', label=sub)

    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=legend_prop_size)
    #axs[1].legend(loc='center right', bbox_to_anchor=(1, 0.5))
    #axs.set_title("Diff {} T and nonT".format(feature))
    #axs[1].set_title("Diff {} norm to POINT".format(feature))

    axs.tick_params(axis='both', which='major', labelsize=label_font_size)

    if(save):
        fig.savefig(save_root.format(feature))
    
    return fig

def lie_stemPlot(features, feature, save_root="plots/LIE/Stem_{}.png", save=True):

    tnt_scores, subjects = coumpute_TnT_scores(features, lie_feat_cols, feature, abs_ratio=False)    
    tnt_scores = tnt_scores.sort_values(by=['descr_ratio'])
    labels = [str(lab) for lab in tnt_scores['subject'].values]

    print(tnt_scores)
    print(labels)

    pos = 0

    for sub in labels:

        sub = int(sub)

        val = tnt_scores.loc[tnt_scores['subject'] == sub]['descr_ratio']
        #pos = tnt_scores.loc[tnt_scores['subject'] == sub].index.values[0]

        print("sub {} pos {}".format(sub, pos))
        marker = marker_map[sub][0]
        color = marker_map[sub][1]
        (markers, stemlines, baseline) = plt.stem([pos],tnt_scores.loc[tnt_scores['subject'] == sub]['descr_ratio'],
         markerfmt=marker)

        plt.setp(markers, markersize=18, color=color)

        pos += 1

    x_pos = np.arange(len(labels))  
    plt.axhline(y=0, color="red")
    plt.xticks(ticks=x_pos, labels=labels)

    plt.xlabel("Subjects", fontsize=label_font_size)
    plt.ylabel("DESCR Lie Score", fontsize=label_font_size)
    plt.tick_params(axis='both', which='major', labelsize=label_font_size)

def lie_premedPlot(features, feature, save_root="plots/LIE/Premed_{}.png", save=True):


    tnt_scores, subjects = coumpute_TnT_scores(features, lie_feat_cols, feature, abs_ratio=False)    
    tnt_scores = tnt_scores.sort_values(by=['descr_ratio'])
    labels = [str(lab) for lab in tnt_scores['subject'].values]

    pos = 0

    for sub in labels:

        sub = int(sub)
        #pos = tnt_scores.loc[tnt_scores['subject'] == sub].index.values[0]

        marker = marker_map[sub][0]
        color = marker_map[sub][1]
        (markers, stemlines, baseline) = plt.stem([pos],tnt_scores.loc[tnt_scores['subject'] == sub]['premed_index'],
         markerfmt=marker)

        plt.setp(markers, markersize=18, color=color)
        plt.setp(stemlines, linewidth=2)

        pos += 1

    x_pos = np.arange(len(labels))  
    plt.axhline(y=0, color="black")
    plt.axhline(y=75, color="green")
    plt.axhline(y=100, color="red")
    plt.xticks(ticks=x_pos, labels=labels)

    plt.xlabel("Subjects", fontsize=label_font_size)
    plt.ylabel("Premeditation Score", fontsize=label_font_size)
    plt.tick_params(axis='both', which='major', labelsize=label_font_size)

def lie_plotTnTPremedIndex(features, feature, save_root="plots/LIE/Premed_{}.png", save=True):

    fig0, axs0 = plt.subplots(1, figsize=(15, 10), num='STEM TnT {}'.format(feature))
    fig1, axs1 = plt.subplots(1, figsize=(15, 10), num='Premed Index TnT {}'.format(feature))

    tnt_scores, subjects = coumpute_TnT_scores(features, lie_feat_cols, feature, abs_ratio=False)    
    tnt_scores = tnt_scores.sort_values(by=['descr_ratio'])
    
    labels = [str(lab) for lab in tnt_scores['subject'].values]
    x_pos = np.arange(len(labels))    

    print(labels)

    axs0.stem(x_pos, tnt_scores['descr_ratio'], linefmt='green', markerfmt='X', label="DESCR")
    #axs0.stem(x_pos, tnt_scores['react_ratio'], linefmt='grey', markerfmt='D', label="REACT")
    axs0.set_xticks(x_pos)
    axs0.set_xticklabels(labels)

    #tnt_scores = tnt_scores.sort_values(by=['premed_index'])

    labels1 = [str(lab) for lab in tnt_scores['subject'].values]
    x_pos1 = np.arange(len(labels))

    axs1.bar(x_pos1, tnt_scores['premed_index'],
                color='dodgerblue', align='center',
                width=0.35, alpha=0.8)

    axs1.axhline(y=0)
    axs1.axhline(y=80, color="limegreen")
    axs1.set_xticks(x_pos1)
    axs1.set_xticklabels(labels1)

    axs1.set_xlabel("Subjects", fontsize=label_font_size)
    axs1.set_ylabel("Premeditation Score", fontsize=label_font_size)
    axs1.tick_params(axis='both', which='major', labelsize=label_font_size)

    if(save):
        fig1.savefig(save_root.format(feature))
    
    return fig0, fig1

# Average Histogram of Target (old) vs nonTarget (new) for each feature
def lie_plotComparBars(features, feat_cols=lie_feat_cols, scale=3, save_root="plots/LIE/Bars_{}.png", save=True):

    feat_comparison = [
        #('duration', 'react_dur', 'point_dur', 'descr_dur'),
        #('fix_freq', 'point_fix_freq', 'react_fix_freq', 'descr_fix_freq'),
        #('sacc_freq', 'point_sacc_freq', 'react_sacc_freq', 'descr_sacc_freq'),
        ('right_mean', 'point_right_mean', 'react_right_mean', 'descr_right_mean', "Right Mean Pupil Dilation"),
        #('right_std', 'point_right_std', 'react_right_std', 'descr_right_std', "Right STD Pupil Dilation"),
        #('right_min', 'point_right_min', 'react_right_min', 'descr_right_min', "Right Min Pupil Dilation"),
        #('right_max', 'point_right_max', 'react_right_max', 'descr_right_max', "Right Max Pupil Dilation"),
        ('left_mean', 'point_left_mean', 'react_left_mean', 'descr_left_mean', "Left Mean Pupil Dilation"),
        #('left_std', 'point_left_std', 'react_left_std', 'descr_left_std', "Left STD Pupil Dilation" ),
        #('left_min', 'point_left_min', 'react_left_min', 'descr_left_min',  "Left Min Pupil Dilation" ),
        #('left_max', 'point_left_max', 'react_left_max', 'descr_left_max',  "Left Max Pupil Dilation" )
    ]

    label_font_size = 18*scale
    legend_prop_size = {'size': 18*scale}

    nonTargets, targets, TnT = aggregate_target_nontarget(features, feat_cols)

    bar_width = 0.25   
    labels = ['POINT', 'REACT', 'DESCR']
    x_pos = np.arange(len(labels))

    for (whole, point, react, descr, title) in feat_comparison:

        fig, axs = plt.subplots(1, figsize=(15 * scale, 10 * scale), num='{}'.format(title))
        #fig.suptitle('{}'.format(whole), fontsize=16)

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
                color='#32CD32', 
                width=bar_width, label='{} Target'.format(title), linewidth=10)

        axs.bar(x_pos+bar_width, y_values_nT,
                yerr=nT_errors,
                color='#1E90FF', 
                width=bar_width, label='{} Non Target'.format(title), linewidth=10)

        axs.legend(loc="upper left", prop=legend_prop_size)

        axs.set_xticks(x_pos)
        axs.set_xticklabels(labels)
        axs.set_ylabel(title, fontsize=label_font_size)
        #axs.set_title('{}'.format(whole))

        axs.tick_params(axis='both', which='major', labelsize=label_font_size)

        if(save):
            fig.savefig(save_root.format(whole), dpi=100)

def lie_plotRLbars(features, feat_cols=lie_feat_cols, save_root="plots/LIE/Bars_{}.png", save=True):

    feat_comparison = [
        ('right_mean', 'point_right_mean', 'react_right_mean', 'descr_right_mean', "Right Mean Pupil Dilation"),
        ('left_mean', 'point_left_mean', 'react_left_mean', 'descr_left_mean', "Left Mean Pupil Dilation"),
    ]

    nonTargets, targets, TnT = aggregate_target_nontarget(features, feat_cols)

    bar_width = 0.25   
    labels = ['POINT', 'REACT', 'DESCR']
    x_pos = np.arange(len(labels))

    fig, axs = plt.subplots(1, figsize=(15, 10), num='{}'.format("Right and Left Mean Pupil Dilation"))

    x_pos = np.arange(len(labels))
    axs.set_xticks(x_pos)
    axs.set_xticklabels(labels)
    axs.tick_params(axis='both', which='major', labelsize=label_font_size)

    colors = ['#32CD32', '#1E90FF', '#FFEB3B', '#FFEB3B']

    for i, (whole, point, react, descr, title) in enumerate(feat_comparison):

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
        
        axs.bar(x_pos*i, y_values_T,
                yerr=T_errors,
                color=colors[i], 
                width=bar_width, label='{} Target'.format(title))

        axs.bar((x_pos+bar_width)*i, y_values_nT,
                yerr=nT_errors,
                color=colors[i+1], 
                width=bar_width, label='{} Non Target'.format(title))

        axs.legend(loc="upper left", prop=legend_prop_size)

    
    

    if(save):
        fig.savefig(save_root.format(whole), dpi=1200)



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

def lie_plotQuestRegression(lie_features, quest_ans, slope, intersect, index_col, TnT_score_feature="right_mean"):

    lie_subs = lie_features.groupby('subject').count().index.values
    quest_subs = quest_ans.groupby('subject').count().index.values

    tnt_scores, subjects = coumpute_TnT_scores(lie_features, lie_feat_cols, TnT_score_feature, abs_ratio=False) 

    subjects = [s for s in lie_subs if s in quest_subs]
    quest_cols = quest_ans.columns

    for col in quest_cols:
        
        fig, axs = plt.subplots(1, figsize=(9, 9), num="lr {}".format(col))
        labels = []

        for i, sub in enumerate(subjects):

            labels.append("{}".format(sub))

            axs.set_title("{} x {}".format(col, index_col))
            axs.set_ylabel("{}".format(index_col))
            axs.set_xlabel("{}".format(col))
            axs.set_label("{}".format(sub))

            axs.scatter(quest_ans.loc[quest_ans['subject'] == sub][col],
                        tnt_scores.loc[tnt_scores['subject'] == sub][index_col],
                        s=200,
                        marker=markers[i])
        
        plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))

        minX = quest_ans[col].min()
        maxX = quest_ans[col].max()
        minY = slope * minX + intersect
        maxY = slope * maxX + intersect        

        axs.plot([minX, maxX], [minY, maxY], alpha=0.75, zorder=100)
