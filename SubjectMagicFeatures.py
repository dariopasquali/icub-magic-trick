import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

from eye_feature_tools import *

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

class SubjectMagicFeatures:
    
    def __init__(self,
                 subject,
                 source,
                 subject_card,
                 annot_dfs,
                 card_eye_dfs,
                 card_eye_sr_early_dfs,
                 card_eye_sr_late_dfs,
                 sr_window,
                 cols=column_names):
        
        # ==== INIT AGGREGATOR =============
           
        self.features = pd.DataFrame(columns=cols)
        
        # Process Each card and compose the dataframe
        for c in range(0, len(card_eye_dfs)):
                   
            # ==== TASK FEATURES =============
            
            self.subject = subject
            self.source = source
            self.duration = annot_dfs[c+1]['duration_ms'].iloc[0]
            self.card_class = card_eye_dfs[c]['class'].iloc[0]
            
            if(self.card_class == subject_card):
                self.is_subject_one = 1
            else:
                self.is_subject_one = 0
                
            self.show_order = annot_dfs[c+1]['show_order'].iloc[0]        
        
            # ==== EXTRACT CARDS ==========
            
            card = \
                card_eye_dfs[c][['diam_right', 'diam_left', 'move_type', 'move_type_id']]

            sr_early_card = \
                card_eye_sr_early_dfs[c][['diam_right', 'diam_left', 'move_type', 'move_type_id']]      
            sr_late_card = \
                card_eye_sr_late_dfs[c][['diam_right', 'diam_left', 'move_type', 'move_type_id']]  
        
            # ==== EVENTS =============
            
            # Whole card
            self.fix_freq, self.sacc_freq = calcEventFeatures(card, self.duration)

            # Early
            self.sre_fix_freq, self.sre_sacc_freq = calcEventFeatures(sr_early_card, sr_window)

            # Late
            self.srl_fix_freq, self.srl_sacc_freq = calcEventFeatures(sr_late_card, sr_window)
             
            # ==== SINGLE CARD PUPIL FEATURES =============

            self.right_mean, self.right_std, \
                self.right_min, self.right_max, \
                    self.left_mean, self.left_std, \
                        self.left_min, self.left_max \
                             = computePupilFeatures(card)

            # ==== EARLY SHORT RESPONSE =============
            
            self.sre_right_mean, self.sre_right_std, \
                self.sre_right_min, self.sre_right_max, \
                    self.sre_left_mean, self.sre_left_std, \
                        self.sre_left_min, self.sre_left_max \
                             = computePupilFeatures(sr_early_card)
            
            # ==== LATE SHORT RESPONSE =============
            
            self.srl_right_mean, self.srl_right_std, \
                self.srl_right_min, self.srl_right_max, \
                    self.srl_left_mean, self.srl_left_std, \
                        self.srl_left_min, self.srl_left_max \
                             = computePupilFeatures(sr_late_card)


            # ==== AGGREGATE =============

            self.column_names = cols            
            self.features = self.features.append(self.aggregate(), ignore_index=True)
        

    def getDataFrame(self):
        return self.features
    
    def aggregate(self):
        return pd.DataFrame(
            data=[[
                self.subject,
                self.source,
                self.duration,
                self.card_class,
                self.show_order,
                self.fix_freq,
                self.sacc_freq,
                self.right_mean,
                self.right_std,
                self.right_min,
                self.right_max,
                self.left_mean,
                self.left_std,
                self.left_min,
                self.left_max,
                self.sre_fix_freq,
                self.sre_sacc_freq,
                self.sre_right_mean,
                self.sre_right_std,
                self.sre_right_min,
                self.sre_right_max,
                self.sre_left_mean,
                self.sre_left_std,
                self.sre_left_min,
                self.sre_left_max,
                self.srl_fix_freq,
                self.srl_sacc_freq,
                self.srl_right_mean,
                self.srl_right_std,
                self.srl_right_min,
                self.srl_right_max,
                self.srl_left_mean,
                self.srl_left_std,
                self.srl_left_min,
                self.srl_left_max,
                self.is_subject_one,
            ]],
            columns=self.column_names
        )