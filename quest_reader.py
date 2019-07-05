import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

def parseSubID(s_num):
    id = int(s_num.replace('s', ''))
    return id

def preprocessQuest(quest_file):

    quest = pd.read_csv(quest_file, sep=';') 

    quest_cols = ['id','subject','eta','sex','extraversion','agreeableness ','conscientiousness ','neuroticism ','openness','NARS_S1','NARS_S2','NARS_S3','NARS_TOTAL','histrionic','narci','MAC_Negative_Interpersonal_Tactics','MAC_Positive_Interpersonal_Tactics','MAC_Cynical_view_human_nature','MAC_Positive_view_human_nature','MAC_TOTAL','PSY_Primary_Psychopathy','PSY_Secondary_Psychopathy','PSY_TOTAL']

    quest.columns = quest_cols
    quest['subject'] = quest['subject'].apply(parseSubID)
    quest = quest.drop(['id', 'sex', 'eta'], axis=1)

    return quest