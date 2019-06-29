import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

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