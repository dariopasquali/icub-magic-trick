import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

def stringToMSec(time):
    h, m, sms = time.split(':')
    s, ms = sms.split('.')
    return 1000 * (int(h)*3600 + int(m)*60 + int(s)) + int(ms)

def stringLieToMSec(time):
    m, tail = time.split('m')
    s, ms = tail.split('s')
    return 1000 * (int(m)*60 + int(s)) + int(ms)

def preprocessAnnotations(annotation_file, card_names):
    # read the eye file
    annot = pd.read_csv(annotation_file, sep='\t',
         names=['tire', 'remove', 'start', 'stop', 'duration', 'class'])

    annot['show_order'] = annot.index    
    annot['start_ms'] = annot['start'].apply(stringToMSec)
    annot['stop_ms'] = annot['stop'].apply(stringToMSec)
    annot['duration_ms'] = annot['duration'].apply(stringToMSec)
    annot = annot.drop(['start', 'stop', 'duration', 'remove'], axis=1)
    default = annot.loc[annot['tire'] == "default"]
    
    dfs = [default]    
    for card in card_names:
        dfs.append(
            annot.loc[(annot['tire'] == "per_card") & (annot['class'] == card)]
        )
        
    return dfs

def preprocessLieAnnotations(annotation_file, card_names):
    # read the eye file
    annot = pd.read_csv(annotation_file, sep=',', 
        names=['subject', 'condition', 'remove', 'start_p', 'stop_p', 'start_d', 'stop_d', 'card', 'label'])

    annot['show_order'] = annot.index

    annot['start_p'] = annot['start_p'].apply(stringLieToMSec)
    annot['stop_p'] = annot['stop_p'].apply(stringLieToMSec)
    annot['start_d'] = annot['start_d'].apply(stringLieToMSec)
    annot['stop_d'] = annot['stop_d'].apply(stringLieToMSec)

    """
    reaction_time = (while the subject is looking at the card) = start_descr - stop_point
    point_reaction_time = start_descr - start_point
    description_time = stop_descr - start_descr
    """
    #annot['point_time'] = annot['stop_p'] - annot['start_p']
    #annot['reaction_time'] = annot['start_d'] - annot['stop_p']
    #annot['point_reaction_time'] = annot['start_d'] - annot['start_p']
    #annot['description_time'] = annot['stop_d'] - annot['start_d']

    annot = annot.drop(['condition', 'remove'], axis=1)
    
    subID = annot.groupby('subject').count().index
    subID = int(subID[0].replace('s', ''))
    start = annot['start_p'].min()
    stop = annot['stop_d'].max()
    card = annot.loc[annot["label"] == 1]['card'].values[0]

    overall = pd.DataFrame(
        data=[[subID, card, start, stop]],
        columns=["subject", "subject_card", "start", "stop"]
        ) 
    
    dfs = [overall] 
    starts = annot.groupby('start_p').count().index.values

    for s in starts:
        dfs.append(
            annot.loc[(annot['start_p'] == s)]
        )

    #for card in card_names:
    #    dfs.append(
    #        annot.loc[(annot['card'] == card)]
    #    )
        
    return dfs


#card_names = ["unicorn", "pepper", "hedge", "pig", "minion", "aliens"]
#fin = "data/annotations_lie/s8.csv"
#annot = preprocessLieAnnotations(fin, card_names)
#print(annot)