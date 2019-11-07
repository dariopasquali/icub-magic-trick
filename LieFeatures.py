import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

from eye_feature_tools import *

lie_feat_cols = [
        'subject',
        'source',
        'card_class',
        'show_order',
        'duration',
        'react_dur',
        'point_dur',
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
        'point_fix_freq',
        'point_sacc_freq',
        'point_right_mean',
        'point_right_std',
        'point_right_min',
        'point_right_max',
        'point_left_mean',
        'point_left_std',
        'point_left_min',
        'point_left_max',
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





class RealTimeHeuristicFeatures:

    def __init__(self, subject, annot_dfs, filtered_interaction_dfs, cols):

        self.features = pd.DataFrame(columns=cols)

        for c, (robot_inter, sub_inter) in enumerate(filtered_interaction_dfs):

            annot = annot_dfs[c+1]

            self.subject = subject
            self.card_class = annot['card'].iloc[0]
            self.label = annot['label'].iloc[0]

            sub_inter = sub_inter[['diam_right', 'diam_left']]

            self.right_mean = sub_inter['diam_right'].mean(skipna = True)
            self.left_mean = sub_inter['diam_left'].mean(skipna = True)
            self.mean = (self.right_mean + self.left_mean) / 2.0

            self.column_names = cols            
            self.features = self.features.append(self.aggregate(), ignore_index=True)
            

    def getDataFrame(self):
        return self.features
    
    def aggregate(self):
        return pd.DataFrame(
            data=[[
                self.subject,
                self.card_class,
                self.right_mean,
                self.left_mean,
                self.mean,
                self.label,
            ]],
            columns=self.column_names
        )



class LieFeatures:
    
    def __init__(self,
                 subject,
                 source,
                 annot_dfs,
                 filtered_interaction_dfs,
                 cols=lie_feat_cols):
        
        # ==== INIT AGGREGATOR =============
           
        self.features = pd.DataFrame(columns=cols)
        
        # Process Each card and compose the dataframe
        for c, (whole, point, reaction, point_reaction, description) in enumerate(filtered_interaction_dfs):
        
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
            point = point[['diam_right', 'diam_left', 'move_type', 'move_type_id']]
            description = description[['diam_right', 'diam_left', 'move_type', 'move_type_id']]  
        
            # ==== EVENTS =============
            
            # Reaction Interval
            self.fix_freq, self.sacc_freq = calcEventFeatures(whole, self.duration)

            # Reaction Interval
            self.react_fix_freq, self.react_sacc_freq = calcEventFeatures(reaction, self.reaction_dur)

            # Point + Reaction Interval
            self.point_fix_freq, self.point_sacc_freq = calcEventFeatures(point, self.point_reaction_dur)

            # Description Interval
            self.descr_fix_freq, self.descr_sacc_freq = calcEventFeatures(description, self.description_dur)

           
            # ==== WHOLE PUPIL FEATURES =============

            self.right_mean, self.right_std, \
                self.right_min, self.right_max, \
                    self.left_mean, self.left_std, \
                        self.left_min, self.left_max \
                             = computePupilFeatures(whole)


            # ==== REACTION PUPIL FEATURES =============

            self.react_right_mean, self.react_right_std, \
                self.react_right_min, self.react_right_max, \
                    self.react_left_mean, self.react_left_std, \
                        self.react_left_min, self.react_left_max \
                             = computePupilFeatures(reaction)

            # ==== POINT REACTION PUPIL FEATURES =============
            
            self.point_right_mean, self.point_right_std, \
                self.point_right_min, self.point_right_max, \
                    self.point_left_mean, self.point_left_std, \
                        self.point_left_min, self.point_left_max \
                             = computePupilFeatures(point)
            
            # ==== DESCRIPTION PUPIL FEATURESE =============
            
            self.descr_right_mean, self.descr_right_std, \
                self.descr_right_min, self.descr_right_max, \
                    self.descr_left_mean, self.descr_left_std, \
                        self.descr_left_min, self.descr_left_max \
                             = computePupilFeatures(description)


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
                self.point_fix_freq,
                self.point_sacc_freq,
                self.point_right_mean,
                self.point_right_std,
                self.point_right_min,
                self.point_right_max,
                self.point_left_mean,
                self.point_left_std,
                self.point_left_min,
                self.point_left_max,
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