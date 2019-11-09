import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
from statsmodels.robust.scale import mad

def normalizeWithinSubject(features, cols, mode="minmax"):

    subjects = features.groupby('subject').count().index.values

    for sub in subjects:
        for col in cols:
            sub_col_feat = features.loc[features['subject'] == sub][col]

            if(mode == "minmax"):
               
                max_c = sub_col_feat.max()
                min_c = sub_col_feat.min()

                features.loc[features['subject'] == sub, [col]] = (sub_col_feat - min_c) / (max_c - min_c)
                
            else:

                mean = sub_col_feat.mean()
                std = sub_col_feat.std()

                features.loc[features['subject'] == sub, [col]] = (sub_col_feat - mean) / std

    return features

# refer an array of eye data to the mean of the baseline
def referToBaseline(data, baseline, mode="sub", source_cols=['diam_right'], baseline_col='diam_right'):

    # this way I can use it in both contexts

    baseline_mean = baseline[baseline_col].mean(skipna = True)
    for c in source_cols:
        if(mode == 'sub'):
            data[c] = data[c] - baseline_mean
                        
        if(mode == 'div'):
            data[c] = data[c] / baseline_mean

    return data 

def cleanZscore(time_series, tresh=5, col="diam_right"):
    # Calc Zscore params

    #print("clean Z")

    #print("Outlier Treshold {}".format(tresh))
    mean = time_series[col].mean(skipna = True)
    std = time_series[col].std(skipna = True)      
    #Filter Outliers    
    time_series.loc[(np.abs((time_series[col] - mean) / std) >= tresh), col] = np.NaN

    return time_series

def cleanMAD(time_series, tresh=3.5, col="diam_right"):
    
    #print("clean MAD")
    #https://medium.com/james-blogs/outliers-make-us-go-mad-univariate-outlier-detection-b3a72f1ea8c7
    
    df = time_series[[col]]
    median = df[col].median()
    df['dist'] = abs(df[col] - median)
    mad = df['dist'].median()
    df['modZ'] = (0.6754 * (df[col] - median)) / mad

    df.loc[abs(df['modZ']) >= 3.5] = np.NaN    
    time_series[col] = df[col]
    
    return time_series

def resampleAndFill(time_series, col="diam_right", resample=True, fill=True, smooth=True):
    # RESAMPLE
    if(resample):
        time_series[col] = time_series[[col]].resample("ms").mean() 
    
    # Fill NaN
    if(fill):
        time_series[col] = time_series[[col]].fillna(method="ffill")
 
    # Smooth
    if(smooth): 
        time_series[col] = time_series[[col]].rolling(window=150).mean()

    return time_series

def dataCleaner(eyeDF, clean_mode="MAD", clean=True, smooth=False):
    cleaner = None
    if(clean_mode == "MAD"):
        cleaner = cleanMAD
    else:
        cleaner = cleanZscore

    # Clean the outliers
    if(clean):
        eyeDF = cleaner(eyeDF, col="diam_right")
        eyeDF = cleaner(eyeDF, col="diam_left")

    # Resample, FillNaN and Smooth
    if(smooth):
        eyeDF = resampleAndFill(eyeDF, col="diam_right")
        eyeDF = resampleAndFill(eyeDF, col="diam_left")

    return eyeDF

def vad_rt_data_filtering(eyeDF, sound_annot, clean=True, clean_mode="MAD", smooth=False):

    (start_p, stop_p, start_d, stop_d) = sound_annot

    robot_time = eyeDF.loc[
        (eyeDF['timestamp'] >= start_p) & (eyeDF['timestamp'] <= stop_p)
    ]

    subject_time = eyeDF.loc[
        (eyeDF['timestamp'] >= start_d) & (eyeDF['timestamp'] <= stop_d)
    ]

    if(clean or smooth):
        robot_time = dataCleaner(robot_time, clean, clean_mode, smooth)
        subject_time = dataCleaner(subject_time, clean, clean_mode, smooth)

    return robot_time, subject_time

def rt_windowed_filtering(eyeDF, annot, annot_1, window_size=1000, clean=True, clean_mode="MAD", smooth=False):

    annot = annot.reset_index()
    start_point_0 = annot.at[0, 'start_p']
    stop_point_0 = annot.at[0, 'stop_p']
    stop_descr_0 = annot.at[0, 'stop_d']

    # Find the temporal interval
    if(not annot_1.empty):
        annot_1 = annot_1.reset_index()
        start_point_1 = annot_1.at[0, 'start_p']
    else:
        start_point_1 = stop_descr_0

    # Extract the intervals
    robot_time = eyeDF.loc[
        (eyeDF['timestamp'] >= start_point_0) & (eyeDF['timestamp'] <= stop_point_0)
    ]

    subject_time = eyeDF.loc[
        (eyeDF['timestamp'] >= stop_point_0) & (eyeDF['timestamp'] <= start_point_1)
    ]

    # Clean
    if(clean or smooth):
        robot_time = dataCleaner(robot_time, clean, clean_mode, smooth)
        subject_time = dataCleaner(subject_time, clean, clean_mode, smooth)
    
    # Filter the Windows
    subject_windows_dfs = []
    window_start = stop_point_0
    window_stop = min(stop_point_0 + window_size, start_point_1)

    while(window_start < start_point_1):
        window = subject_time.loc[
            (subject_time['timestamp'] >= window_start) & (subject_time['timestamp'] <= window_stop)
        ]

        if(not window.empty):
            subject_windows_dfs.append(window)

        window_start = window_stop
        window_stop = min(window_start + window_size, start_point_1)

    return robot_time, subject_windows_dfs

    


def rt_data_filtering(eyeDF, annot, annot_1, clean=True, clean_mode="MAD", smooth=False):

    annot = annot.reset_index()
    start_point_0 = annot.at[0, 'start_p']
    stop_point_0 = annot.at[0, 'stop_p']
    stop_descr_0 = annot.at[0, 'stop_d']


    if(not annot_1.empty):
        annot_1 = annot_1.reset_index()
        start_point_1 = annot_1.at[0, 'start_p']

    robot_time = eyeDF.loc[
        (eyeDF['timestamp'] >= start_point_0) & (eyeDF['timestamp'] <= stop_point_0)
    ]

    if(not annot_1.empty):
        subject_time = eyeDF.loc[
            (eyeDF['timestamp'] >= stop_point_0) & (eyeDF['timestamp'] <= start_point_1)
        ]
    else:
        subject_time = eyeDF.loc[
            (eyeDF['timestamp'] >= stop_point_0) & (eyeDF['timestamp'] <= stop_descr_0)
        ]

    if(clean or smooth):
        robot_time = dataCleaner(robot_time, clean, clean_mode, smooth)
        subject_time = dataCleaner(subject_time, clean, clean_mode, smooth)

    return robot_time, subject_time

def lieDataFiltering(eyeDF, annot, clean=True, clean_mode="MAD", smooth=False):

    # 'start_p', 'stop_p', 'start_d', 'stop_d'
    annot = annot.reset_index()
    start_point = annot.at[0, 'start_p']
    stop_point = annot.at[0, 'stop_p']
    start_descr = annot.at[0, 'start_d']
    stop_descr = annot.at[0, 'stop_d']

    """
    start_point = icub moves the eyes to point
    stop_point = the subject touches the card
    start_descr = the subject starts to talk
    stop_descr = the subject finish to talk

    point_time = stop_point - start_point
    reaction_time = (while the subject is looking at the card) = start_descr - stop_point
    point_reaction_time = start_descr - start_point
    description_time = stop_descr - start_descr
    """
    whole_time = eyeDF.loc[
        (eyeDF['timestamp'] >= start_point) & (eyeDF['timestamp'] <= stop_descr)
    ]

    point_time = eyeDF.loc[
        (eyeDF['timestamp'] >= start_point) & (eyeDF['timestamp'] <= stop_point)
    ]

    reaction_time = eyeDF.loc[
        (eyeDF['timestamp'] >= stop_point) & (eyeDF['timestamp'] <= start_descr)
    ]

    point_reaction_time = eyeDF.loc[
        (eyeDF['timestamp'] >= start_point) & (eyeDF['timestamp'] <= start_descr)
    ]

    description_time = eyeDF.loc[
        (eyeDF['timestamp'] >= start_descr) & (eyeDF['timestamp'] <= stop_descr)
    ]

    if(clean or smooth):
        point_time = dataCleaner(point_time, clean, clean_mode, smooth)
        whole_time = dataCleaner(whole_time, clean, clean_mode, smooth)
        reaction_time = dataCleaner(reaction_time, clean, clean_mode, smooth)
        point_reaction_time = dataCleaner(point_reaction_time, clean, clean_mode, smooth)
        description_time = dataCleaner(description_time, clean, clean_mode, smooth)

    return whole_time, point_time, reaction_time, point_reaction_time, description_time

def filterEyeData(eyeDF, annot, clean=True, start_col="start_ms", stop_col="stop_ms", clean_mode="MAD", smooth=False):
    annot = annot.reset_index()
    start = annot.at[0, start_col]
    stop = annot.at[0, stop_col]
    novelty = eyeDF.loc[
        (eyeDF['timestamp'] >= start) & (eyeDF['timestamp'] <= stop)
    ]

    if(clean or smooth):
        novelty = dataCleaner(novelty, clean, clean_mode, smooth)

    return novelty

def filterShortEyeResponse(eyeDF, annot, 
    after_start=False, before_end=False, window=1500,
    start_col="start_ms", stop_col="stop_ms",
    clean=True, clean_mode="MAD", smooth=False):

    annot = annot.reset_index()
    start = annot.at[0, start_col]
    stop = annot.at[0, stop_col]    

    if(after_start):
        # After Start
        short_stop = start + window
        #look to a window of 1.5 seconds after the stimulus    
        stop = min(short_stop, stop)    
        
    if(before_end):
        # Before End
        short_start = stop - window
        start = max(short_start, start)
    
    short_response = \
        eyeDF.loc[(eyeDF['timestamp'] >= start) & (eyeDF['timestamp'] <= stop)]

    if(clean or smooth):
        short_response = dataCleaner(short_response, clean, clean_mode, smooth)

    return short_response

def filterBaseline(eyeDF, annot, window=5000, start_col="start_ms", stop_col="stop_ms", clean=True, clean_mode="MAD", smooth=False):
    annot = annot.reset_index()
    start = annot.at[0, start_col]
    stop = annot.at[0, stop_col]
    
    early_start = start - window 
    baseline = eyeDF.loc[(eyeDF['timestamp'] >= early_start) & (eyeDF['timestamp'] <= start)]

    if(clean or smooth):
        baseline = dataCleaner(baseline, clean, clean_mode, smooth)

    return baseline