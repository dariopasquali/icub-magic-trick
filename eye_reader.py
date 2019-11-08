import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

def col_var_init():
    cols_frontiers = [
            'timestamp','prj_name','exp_date','name',
            'rec_name','rec_date','start_time','duration',
            'rec_fix_filter_name',
            'event','event_val',
            'gaze_x','gaze_y','gaze_3d_x','gaze_3d_y','gaze_3d_z',
            'gaze_dir_right_x','gaze_dir_right_y','gaze_dir_right_z',
            'gaze_dir_left_x','gaze_dir_left_y','gaze_dir_left_z',
            'pupil_pos_left_x','pupil_pos_left_y','pupil_pos_left_y',
            'pupil_pos_right_x','pupil_pos_right_y','pupil_pos_right_z',
            'diam_left', 'diam_right',
            'rec_media_name', 'rec_media_width', 'rec_media_heigh',
            'move_type','event_dur','move_type_id','fix_x','fix_y',
            'gyro_x','gyro_y', 'gyro_z','acc_x', 'acc_y', 'acc_z',
            'ID'
            ]

    cols_pilot = [
            'timestamp','prj_name','exp_date','name',
            'rec_name','rec_date','start_time','duration',
            'rec_fix_filter_name',
            'event','event_val',
            'gaze_x','gaze_y','gaze_3d_x','gaze_3d_y','gaze_3d_z',
            'gaze_dir_right_x','gaze_dir_right_y','gaze_dir_right_z',
            'gaze_dir_left_x','gaze_dir_left_y','gaze_dir_left_z',
            'pupil_pos_left_x','pupil_pos_left_y','pupil_pos_left_y',
            'pupil_pos_right_x','pupil_pos_right_y','pupil_pos_right_z',
            'diam_left', 'diam_right',
            'move_type','event_dur','move_type_id','fix_x','fix_y',
            'ID'
            ]

    cols_to_drop_frontiers = ['ID','prj_name','exp_date', 'rec_name', 'duration',
                    'rec_fix_filter_name','gaze_x','gaze_y',
                    'gaze_3d_x','gaze_3d_y','gaze_3d_z',
                    'gaze_dir_right_x','gaze_dir_right_y','gaze_dir_right_z','gaze_dir_left_x','gaze_dir_left_y', 'gaze_dir_left_z','pupil_pos_left_x','pupil_pos_left_y','pupil_pos_left_y', 'fix_x','fix_y',
                    'pupil_pos_right_x','pupil_pos_right_y','pupil_pos_right_z',
                    'rec_media_name', 'rec_media_width', 'rec_media_heigh',
                    'gyro_x','gyro_y', 'gyro_z','acc_x', 'acc_y', 'acc_z'
                    ]

    cols_to_drop_pilot = ['ID','prj_name','exp_date', 'rec_name', 'duration',
                    'rec_fix_filter_name','gaze_x','gaze_y',
                    'gaze_3d_x','gaze_3d_y','gaze_3d_z',
                    'gaze_dir_right_x','gaze_dir_right_y','gaze_dir_right_z','gaze_dir_left_x','gaze_dir_left_y', 'gaze_dir_left_z','pupil_pos_left_x','pupil_pos_left_y','pupil_pos_left_y', 'fix_x','fix_y',
                    'pupil_pos_right_x','pupil_pos_right_y','pupil_pos_right_z'
                    ]

    cols = cols_frontiers
    cols_to_drop = cols_to_drop_frontiers

    return cols, cols_to_drop, cols_frontiers, \
         cols_pilot, cols_to_drop_frontiers, cols_to_drop_pilot
             

def clearFloats(x):
    if(type(x) == str):
        x = x.replace('\U00002013', '-').replace(',', '.')
    return x   


def preprocess_streaming_eye_data(eye_file, source="frontiers"):

    eye_data = preprocessEye(eye_file, source)

    # Select relevant columns
    eye_data = eye_data[['timestamp', 'name', 'rec_date', 'start_time', 'diam_left', 'diam_right']]
    
    eye_data.to_csv("./RT/temp_tobii_stream.csv", index=False, header=False)
    return open("./RT/temp_tobii_stream.csv", "r")


def preprocessEye(eye_file, source="frontiers"):
    
    cols, cols_to_drop, cols_frontiers, \
         cols_pilot, cols_to_drop_frontiers, cols_to_drop_pilot = col_var_init()

    if(source == "pilot"):
        cols = cols_pilot
        cols_to_drop = cols_to_drop_pilot
    
    # read the eye file    
    df = pd.read_csv(eye_file, encoding='utf-16', sep='\t')    
    # rename columns
    df.columns = cols
        
    # remove useless columns
    df = df.drop(cols_to_drop, axis=1)
    
    # remove useless event rows (empty)
    df = df[df.move_type != "EyesNotFound"]
    df = df[df.event != "RecordingStart"]
    df = df[df.event != "RecordingEnd"]
    df = df[df.event != "SyncPortOutHigh"]
    df = df[df.event != "SyncPortOutLow"]
    df = df.drop(['event', 'event_val'], axis=1)
    
    # set datatypes
    df = df.applymap(clearFloats)
    dfe = df.apply(pd.to_numeric, errors='ignore')

    
    dfe['dt_index'] = pd.to_datetime(dfe['start_time'], \
                                     format="%H:%M:%S.%f") + \
                                    pd.to_timedelta(dfe['timestamp'], unit='ms')
    # set datetime index
    dfe = dfe.set_index(pd.DatetimeIndex(dfe['dt_index']))    

    return dfe

