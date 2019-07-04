import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
from statsmodels.robust.scale import mad

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
        time_series[col] = time_series[[col]].resample("ms").mean(skipna=True) 
    
    # Fill NaN
    if(fill):
        time_series[col] = time_series[[col]].fillna(method="ffill")
 
    # Smooth
    if(smooth): 
        time_series[col] = time_series[[col]].rolling(window=150).mean(skipna=True)

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