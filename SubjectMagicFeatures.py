import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

def referToBaseline(data, baseline, mode="sub"):

    base_right_mean = baseline['diam_right'].mean(skipna = True)
    base_left_mean = baseline['diam_left'].mean(skipna = True)

    if(mode == 'sub'):
        data['diam_right'] = data['diam_right'] - base_right_mean
        data['diam_left'] = data['diam_left'] - base_left_mean
                    
    if(mode == 'div'):
        data['diam_right'] = data['diam_right'] / base_right_mean
        data['diam_left'] = data['diam_left'] / base_left_mean

    return data 

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
                 cols,
                 refer_to_baseline=False,
                 baseline=None,
                 refer_method='sub'):
        
        # ==== CALCULATE THE BASELINE =============
           
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
                card_eye_dfs[c][['diam_right', 'diam_left', 'move_type']]

            sr_early_card = \
                card_eye_sr_early_dfs[c][['diam_right', 'diam_left', 'move_type']]      
            sr_late_card = \
                card_eye_sr_late_dfs[c][['diam_right', 'diam_left', 'move_type']]  
        
            # ==== EVENTS =============
            
            # Whole card
            self.fix_freq, self.sacc_freq = \
                self.calcEventFeatures(card, self.duration)

            # Early
            self.sre_fix_freq, self.sre_sacc_freq = \
                self.calcEventFeatures(sr_early_card, sr_window)

            # Late
            self.srl_fix_freq, self.srl_sacc_freq = \
                self.calcEventFeatures(sr_late_card, sr_window)

            # ==== RESCALE DATA TO BASELINE =============
            if(refer_to_baseline):
                card = referToBaseline(card, baseline, refer_method)
                sr_early_card = referToBaseline(sr_early_card, baseline, refer_method)
                sr_late_card = referToBaseline(sr_late_card , baseline, refer_method)
             
            # ==== SINGLE CARD PUPIL FEATURES =============

            self.right_mean, self.right_std, \
                self.right_min, self.right_max, \
                    self.left_mean, self.left_std, \
                        self.left_min, self.left_max \
                             = self.computePupilFeatures(card)

            # ==== EARLY SHORT RESPONSE =============
            
            self.sre_right_mean, self.sre_right_std, \
                self.sre_right_min, self.sre_right_max, \
                    self.sre_left_mean, self.sre_left_std, \
                        self.sre_left_min, self.sre_left_max \
                             = self.computePupilFeatures(sr_early_card)
            
            # ==== LATE SHORT RESPONSE =============
            
            self.srl_right_mean, self.srl_right_std, \
                self.srl_right_min, self.srl_right_max, \
                    self.srl_left_mean, self.srl_left_std, \
                        self.srl_left_min, self.srl_left_max \
                             = self.computePupilFeatures(sr_late_card)


            # ==== AGGREGATE =============

            self.column_names = cols            
            self.features = self.features.append(self.aggregate(), ignore_index=True)
            

    def computePupilFeatures(self, data):
        # Right Eye
        p_mean_r = data["diam_right"].mean(skipna = True)
        p_std_r = data["diam_right"].std(skipna = True)
        p_max_r = data["diam_right"].min(skipna = True)
        p_min_r = data["diam_right"].max(skipna = True)

        # Left Eye
        p_mean_l = data["diam_left"].mean(skipna = True)
        p_std_l = data["diam_left"].std(skipna = True)
        p_max_l = data["diam_left"].min(skipna = True)
        p_min_l = data["diam_left"].max(skipna = True)

        return p_mean_r, p_std_r, p_max_r, p_min_r, \
                 p_mean_l, p_std_l, p_max_l, p_min_l

    def calcEventFeatures(self, data, time_window):
        move_types = data[['move_type']]
        move_types['count'] = 1
        move_types = move_types.groupby('move_type').count()
            
        fix_num = move_types.loc[move_types.index == "Fixation"]
        sacc_num = move_types.loc[move_types.index == "Saccade"]
            
        # Check if not empty
        if not fix_num.empty:
            fix_num = fix_num.iloc[0][0]
        else:
            fix_num = 0
                
        if not sacc_num.empty:
            sacc_num = sacc_num.iloc[0][0]
        else:
            sacc_num = 0
            
        duration_sec = time_window / 1000            
        fix_freq = fix_num / duration_sec
        sacc_freq = sacc_num / duration_sec

        return fix_freq, sacc_freq

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