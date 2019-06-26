import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

def cleanTimeSeries(time_series, tresh=5, col_right="diam_right", col_left="diam_left"):
    # Calc Zscore params
    righ_mean = time_series[col_right].mean(skipna = True)
    righ_std = time_series[col_right].std(skipna = True)
    left_mean = time_series[col_left].mean(skipna = True)
    left_std = time_series[col_left].std(skipna = True)    
    
    #Filter Outliers    
    time_series.loc[(np.abs((time_series[col_right] - righ_mean) / righ_std) >= tresh), col_left] = np.NaN
    time_series.loc[(np.abs((time_series[col_right] - left_mean) / left_std) >= tresh), col_left] = np.NaN
    return time_series

def resampleAndFill(time_series, resample=True, fill=True, smooth=True):
    # RESAMPLE
    if(resample):
        time_series['diam_right'] = \
            time_series[['diam_right']].resample("ms").mean()
        time_series['diam_left'] = \
            time_series[['diam_left']].resample("ms").mean()   
    
    # Fill NaN
    if(fill):
        time_series["diam_right"] = time_series[["diam_right"]].fillna(method="ffill")
        time_series["diam_left"] = time_series[["diam_left"]].fillna(method="ffill")
 
    # Smooth
    if(smooth): 
        time_series["diam_right"] = time_series[['diam_right']].rolling(window=150).mean()
        time_series["diam_left"] = time_series[['diam_left']].rolling(window=150).mean() 

    return time_series

def filterEyeData(eyeDF, annot, clean=True, smooth=False):
    annot = annot.reset_index()
    start = annot.at[0, 'start_ms']
    stop = annot.at[0, 'stop_ms']
    novelty = eyeDF.loc[
        (eyeDF['timestamp'] >= start) & (eyeDF['timestamp'] <= stop)
    ]

    # Clean the outliers
    if(clean):
        novelty = cleanTimeSeries(novelty)

    # Resample, FillNaN and Smooth
    if(smooth):
        novelty = resampleAndFill(novelty) 
    
    return novelty


def filterShortEyeResponse(eyeDF, annot, 
    after_start=False, before_end=False, window=1500,
    clean=True, smooth=False):

    annot = annot.reset_index()
    start = annot.at[0, 'start_ms']
    stop = annot.at[0, 'stop_ms']    
    
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

    # Clean the outliers
    if(clean):
        novelty = cleanTimeSeries(novelty)

    # Resample, FillNaN and Smooth
    if(smooth):
        novelty = resampleAndFill(novelty) 

    return short_response

def filterBaseline(eyeDF, annot, window=5000, clean=True, smooth=False):
    annot = annot.reset_index()
    start = annot.at[0, 'start_ms']
    stop = annot.at[0, 'start_ms']
    
    early_start = start - window 
    baseline = eyeDF.loc[(eyeDF['timestamp'] >= early_start) & (eyeDF['timestamp'] <= stop)]

    # Clean the outliers
    if(clean):
        baseline = cleanTimeSeries(baseline)

    # Resample, FillNaN and Smooth
    if(smooth):
        baseline = resampleAndFill(baseline) 

    return baseline