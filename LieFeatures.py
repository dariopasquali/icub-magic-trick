import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'


lie_feat_cols = [
        'subject',
        'source',
        'card_class',
        'show_order',
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

class LieFeatures:
    
    def __init__(self,
                 subject,
                 source,
                 annot_dfs,
                 filtered_interaction_dfs,
                 cols=lie_feat_cols,
                 refer_to_baseline=False,
                 baseline=None,
                 refer_method='sub'):
        
        # ==== CALCULATE THE BASELINE =============
           
        self.features = pd.DataFrame(columns=cols)
        
        # Process Each card and compose the dataframe
        for c, (whole, reaction, point_reaction, description) in enumerate(filtered_interaction_dfs):
        
            annot = annot_dfs[c+1]

            # ==== TASK FEATURES =============
            
            self.subject = subject
            self.source = source

            # ==== TIMING AND ANNOTATION FEATURES =============

            self.duration = (annot['stop_d'] - annot['start_p']).iloc[0]
            self.point_dur = (annot['stop_p'] - annot['start_p']).iloc[0]
            self.description_dur = (annot['stop_d'] - annot['start_d']).iloc[0]
            self.reaction_dur = (annot['start_d'] - annot['stop_p']).iloc[0]
            self.point_reaction_dur = (annot['start_d'] - annot['start_p']).iloc[0]
            self.card_class = annot['card'].iloc[0]
            self.label = annot['label'].iloc[0]
                
            self.show_order = annot['show_order'].iloc[0]        
        
            # ==== EXTRACT TIME INTERVAL ==========
            
            whole = whole[['diam_right', 'diam_left', 'move_type', 'move_type_id']]
            reaction = reaction[['diam_right', 'diam_left', 'move_type', 'move_type_id']]
            point_reaction = point_reaction[['diam_right', 'diam_left', 'move_type', 'move_type_id']]
            description = description[['diam_right', 'diam_left', 'move_type', 'move_type_id']]  
        
            # ==== EVENTS =============
            
            # Reaction Interval
            self.fix_freq, self.sacc_freq = self.calcEventFeatures(whole, self.reaction_dur)

            # Reaction Interval
            self.react_fix_freq, self.react_sacc_freq = self.calcEventFeatures(reaction, self.reaction_dur)

            # Point + Reaction Interval
            self.point_react_fix_freq, self.point_react_sacc_freq = self.calcEventFeatures(point_reaction, self.point_reaction_dur)

            # Description Interval
            self.descr_fix_freq, self.descr_sacc_freq = self.calcEventFeatures(description, self.description_dur)

            # ==== RESCALE DATA TO BASELINE =============
            if(refer_to_baseline):
                whole = referToBaseline(whole, baseline, refer_method)
                reaction = referToBaseline(reaction, baseline, refer_method)
                point_reaction = referToBaseline(point_reaction, baseline, refer_method)
                description = referToBaseline(description , baseline, refer_method)
            
            # ==== WHOLE PUPIL FEATURES =============

            self.right_mean, self.right_std, \
                self.right_min, self.right_max, \
                    self.left_mean, self.left_std, \
                        self.left_min, self.left_max \
                             = self.computePupilFeatures(whole)


            # ==== REACTION PUPIL FEATURES =============

            self.react_right_mean, self.react_right_std, \
                self.react_right_min, self.react_right_max, \
                    self.react_left_mean, self.react_left_std, \
                        self.react_left_min, self.react_left_max \
                             = self.computePupilFeatures(reaction)

            # ==== POINT REACTION PUPIL FEATURES =============
            
            self.point_react_right_mean, self.point_react_right_std, \
                self.point_react_right_min, self.point_react_right_max, \
                    self.point_react_left_mean, self.point_react_left_std, \
                        self.point_react_left_min, self.point_react_left_max \
                             = self.computePupilFeatures(point_reaction)
            
            # ==== DESCRIPTION PUPIL FEATURESE =============
            
            self.descr_right_mean, self.descr_right_std, \
                self.descr_right_min, self.descr_right_max, \
                    self.descr_left_mean, self.descr_left_std, \
                        self.descr_left_min, self.descr_left_max \
                             = self.computePupilFeatures(description)


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
        move_types = data[['move_type', 'move_type_id']]
        
        fix_num = move_types.loc[move_types['move_type'] == "Fixation"]
        sacc_num = move_types.loc[move_types['move_type'] == "Saccade"]

        fix_num = len(fix_num.groupby('move_type_id').count().index)
        sacc_num = len(sacc_num.groupby('move_type_id').count().index)
        
        #print("fix {}, sacc {}".format(fix_num, sacc_num))

        duration_sec = time_window / 1000

        #print("duration_sec {}".format(duration_sec))

        fix_freq = fix_num / duration_sec
        sacc_freq = sacc_num / duration_sec

        return fix_freq, sacc_freq

        #return fix_num, sacc_num

    def getDataFrame(self):
        return self.features
    
    def aggregate(self):
        return pd.DataFrame(
            data=[[
                self.subject,
                self.source,
                self.card_class,
                self.show_order,
                self.duration,
                self.reaction_dur,
                self.point_reaction_dur,
                self.description_dur,
                self.react_fix_freq,
                self.sacc_freq,
                self.right_mean,
                self.right_std,
                self.right_min,
                self.right_max,
                self.left_mean,
                self.left_std,
                self.left_min,
                self.left_max,
                self.react_fix_freq,
                self.react_sacc_freq,
                self.react_right_mean,
                self.react_right_std,
                self.react_right_min,
                self.react_right_max,
                self.react_left_mean,
                self.react_left_std,
                self.react_left_min,
                self.react_left_max,
                self.point_react_fix_freq,
                self.point_react_sacc_freq,
                self.point_react_right_mean,
                self.point_react_right_std,
                self.point_react_right_min,
                self.point_react_right_max,
                self.point_react_left_mean,
                self.point_react_left_std,
                self.point_react_left_min,
                self.point_react_left_max,
                self.descr_fix_freq,
                self.descr_sacc_freq,
                self.descr_right_mean,
                self.descr_right_std,
                self.descr_right_min,
                self.descr_right_max,
                self.descr_left_mean,
                self.descr_left_std,
                self.descr_left_min,
                self.descr_left_max,
                self.label,
            ]],
            columns=self.column_names
        )