import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

from time_series_extractor import *
from SubjectMagicFeatures import *

subject = 25
card_names = ['unicorn', 'pepper', 'minion', 'pig', 'hedge', 'aliens']

eye_df, annot_dfs, baseline, overall_eye_df, \
            cards_eye_dfs, early_sr, late_sr = \
                loadTimeSeries(subject, card_names)

subject_card = getSubjectCard(subject, subject_cards_file)
#fig = plotPupilDilationTimeSeries(subject, card_names, overall_eye_df, cards_eye_dfs)
print(eye_df.columns)