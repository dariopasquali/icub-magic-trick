import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

def stringToMSec(time):
    h, m, sms = time.split(':')
    s, ms = sms.split('.')
    return 1000 * (int(h)*3600 + int(m)*60 + int(s)) + int(ms)

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