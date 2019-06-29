import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
from statsmodels.robust.scale import mad


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

def filterEyeData(eyeDF, annot, clean=True, clean_mode="MAD", smooth=False):
    annot = annot.reset_index()
    start = annot.at[0, 'start_ms']
    stop = annot.at[0, 'stop_ms']
    novelty = eyeDF.loc[
        (eyeDF['timestamp'] >= start) & (eyeDF['timestamp'] <= stop)
    ]

    cleaner = None
    if(clean_mode == "MAD"):
        cleaner = cleanMAD
    else:
        cleaner = cleanZscore

    # Clean the outliers
    if(clean):
        novelty = cleaner(novelty, col="diam_right")
        novelty = cleaner(novelty, col="diam_left")

    # Resample, FillNaN and Smooth
    if(smooth):
        novelty = resampleAndFill(novelty, col="diam_right")
        novelty = resampleAndFill(novelty, col="diam_left") 
    
    return novelty


def filterShortEyeResponse(eyeDF, annot, 
    after_start=False, before_end=False, window=1500,
    clean=True, clean_mode="MAD", smooth=False):

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

    cleaner = None
    if(clean_mode == "MAD"):
        cleaner = cleanMAD
    else:
        cleaner = cleanZscore

    # Clean the outliers
    if(clean):
        short_response = cleaner(short_response, col="diam_right")
        short_response = cleaner(short_response, col="diam_left")

    # Resample, FillNaN and Smooth
    if(smooth):
        short_response = resampleAndFill(short_response, col="diam_right")
        short_response = resampleAndFill(short_response, col="diam_left") 

    return short_response

def filterBaseline(eyeDF, annot, window=5000, clean=True,clean_mode="MAD", smooth=False):
    annot = annot.reset_index()
    start = annot.at[0, 'start_ms']
    stop = annot.at[0, 'start_ms']
    
    early_start = start - window 
    baseline = eyeDF.loc[(eyeDF['timestamp'] >= early_start) & (eyeDF['timestamp'] <= stop)]

    cleaner = None
    if(clean_mode == "MAD"):
        cleaner = cleanMAD
    else:
        cleaner = cleanZscore

    # Clean the outliers
    if(clean):
        baseline = cleaner(baseline, col="diam_right")
        baseline = cleaner(baseline, col="diam_left")

    # Resample, FillNaN and Smooth
    if(smooth):
        baseline = resampleAndFill(baseline, col="diam_right")
        baseline = resampleAndFill(baseline, col="diam_left") 

    return baseline