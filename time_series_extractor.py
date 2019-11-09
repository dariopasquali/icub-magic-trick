import pandas as pd
import numpy as np
from scipy import stats
import glob 
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

# PROJECT
from data_cleaning import  *
from annotation_reader import *
from sound_reader import *
from eye_reader import *


annotations_in_temp = "data/{}annotations/s{}.csv"
annotations_lie_in_temp = "data/{}annotations_lie/s{}.csv"
tobii_in_temp = "data/{}tobii/s{}.csv"
subject_cards_file = "data/{}cards.csv"

sound_in_temp = "data/{}sounds/s{}.wav"

def annot_var_init():
    root_pilot = "PILOT/"
    root_front = ""
    root = root_front

    return root, root_pilot, root_front

def extractMinSubjectSet(source="frontiers",
                         annot_path=annotations_in_temp,
                         eye_data_path=tobii_in_temp):

    root, root_pilot, root_front = annot_var_init()
    root = root_front
    if(source == "pilot"):
        root = root_pilot
    
    annot_path = annot_path.format(root, "*")
    eye_path = eye_data_path.format(root, "*")

    annot_glob = glob.glob(annot_path)
    eye_glob = glob.glob(eye_path)

    annot_subjects = []
    eye_subjects = []
    
    for fn in annot_glob:
        id = fn.split("/")[-1]
        id = id.split(".")[0].replace('s', '')
        annot_subjects.append(int(id))

    for fn in eye_glob:
        id =fn.split("/")[-1].split(".")[0].replace('s', '')
        eye_subjects.append(int(id))

    subjects = [s for s in annot_subjects if s in eye_subjects]
    subjects.sort()
    
    return subjects

def getSubjectCard(subject, source="frontiers", cards_file=subject_cards_file):

    root, root_pilot, root_front = annot_var_init()
    root = root_front
    if(source == "pilot"):
        root = root_pilot

    s_cards = pd.read_csv(cards_file.format(root), sep=';')
    sub_name = "s" + str(subject)    
    card = s_cards.loc[s_cards['subject'] == sub_name]
    return card['card'].values[0]

def loadTimeSeries(subject, card_names,
                    source="frontiers",
                    tobii_input_template=tobii_in_temp,
                    annot_input_template=annotations_in_temp,
                    sr_window=1500,
                    clean_mode="MAD",
                    clean=True,
                    smooth=False):
    
    root, root_pilot, root_front = annot_var_init()
    print("LOAD s{} from {}".format(subject, source))

    root = root_front
    if(source == "pilot"):
        root = root_pilot

    # Complete the files
    eye_in = tobii_input_template.format(root, subject)
    annot_in = annot_input_template.format(root, subject)
    
    # vector with temporal filtered data for each card
    cards = []
    cards_sr_early = []
    cards_sr_late = []
    
    # load tobii data
    eye = preprocessEye(eye_in, source)
    
    # load novelty annotations
    annotations = preprocessAnnotations(annot_in, card_names)
    
    # filter eye data relative to overall novelty phase
    overall = filterEyeData(eye, annotations[0], clean=clean, clean_mode=clean_mode, smooth=smooth)
    
    # filter the baseline to rescale the pupil dilation data
    baseline = filterBaseline(eye, annotations[0], window=5000)
    
    # single cards eye data overall and in the 1.5 sec after the stimulus
    for i in range(1, len(annotations)):
        c = filterEyeData(overall, annotations[i], clean=False, smooth=False)
        
        early = filterShortEyeResponse(overall,
                                         annotations[i],
                                         after_start=True,
                                         before_end=False,
                                         window=sr_window,
                                         clean=False, smooth=False)
        
        late = filterShortEyeResponse(overall,
                                         annotations[i],
                                         after_start=False,
                                         before_end=True,
                                         window=sr_window,
                                         clean=False, smooth=False)
        
        c['class'] = annotations[i]['class'].iloc[0]
        cards.append(c)
        cards_sr_early.append(early)
        cards_sr_late.append(late)
    
    return eye, annotations, baseline, overall, cards, cards_sr_early, cards_sr_late

def load_rt_annotations(subject, card_names,
                    source="frontiers",
                    annot_input_template=annotations_lie_in_temp):


    root, root_pilot, root_front = annot_var_init()
    print("LOAD s{} from {}".format(subject, source))

    root = root_front
    if(source == "pilot"):
        root = root_pilot

    # Complete the files
    annot_in = annot_input_template.format(root, subject)

    # Load annotations
    annotations = preprocessLieAnnotations(annot_in, card_names)

    return annotations

def create_rt_eye_streaming(subject, source="frontiers", tobii_input_template=tobii_in_temp):

    root, root_pilot, root_front = annot_var_init()
    print("LOAD s{} from {}".format(subject, source))

    root = root_front
    if(source == "pilot"):
        root = root_pilot

    # Complete the files
    eye_in = tobii_input_template.format(root, subject)

    # load tobii data
    eye_streaming = preprocess_streaming_eye_data(eye_in, source)

    return eye_streaming


def load_rt_windowed_features(subject, card_names,
                    source="frontiers",
                    tobii_input_template=tobii_in_temp,
                    annot_input_template=annotations_lie_in_temp,
                    refer_to_baseline=True, refer_mode='sub',
                    clean_mode="MAD",
                    clean=True,
                    smooth=False,
                    window_size=1000):

    root, root_pilot, root_front = annot_var_init()
    print("LOAD s{} from {}".format(subject, source))

    root = root_front
    if(source == "pilot"):
        root = root_pilot

    # Complete the files
    eye_in = tobii_input_template.format(root, subject)
    annot_in = annot_input_template.format(root, subject)
    
    # vector with temporal filtered data for each card
    filtered_interaction_dfs = []
    
    # load tobii data
    eye = preprocessEye(eye_in, source)

    # Load annotations
    annotations = preprocessLieAnnotations(annot_in, card_names)


    # filter eye data relative to the entire phase
    overall = filterEyeData(eye, annotations[0], \
        start_col="start", stop_col="stop", \
        clean=clean, clean_mode=clean_mode, smooth=smooth)

    # Extract the baseline to rescale the pupil dilation data
    baseline = filterBaseline(eye, annotations[0], \
        start_col="start", stop_col="stop", \
        clean=clean, clean_mode=clean_mode, smooth=smooth, window=5000)

    if(refer_to_baseline):
        # Refer the Pupil data to the baseline
        overall = referToBaseline(overall, baseline, mode=refer_mode,
        source_cols=['diam_right'], baseline_col='diam_right')

        overall = referToBaseline(overall, baseline, mode=refer_mode,
        source_cols=['diam_left'], baseline_col='diam_left')

    for i in range(1, len(annotations)):        

        if(i < (len(annotations)-1)):
            robot_interaction, subject_interaction_windows = rt_windowed_filtering(overall, annotations[i], annotations[i+1], window_size, clean=False, smooth=False)
        else:
            robot_interaction, subject_interaction_windows = rt_windowed_filtering(overall, annotations[i], pd.DataFrame(), window_size, clean=False, smooth=False)
            
        
        robot_interaction['card'] = annotations[i]['card'].iloc[0]

        for w in subject_interaction_windows:
            w['card'] = annotations[i]['card'].iloc[0]
            w['label'] = annotations[i]['label'].iloc[0]
            filtered_interaction_dfs.append(
                (robot_interaction, w)
            )

    return eye, annotations, baseline, overall, filtered_interaction_dfs



def load_rt_lie_timeseries(subject, card_names,
                    source="frontiers",
                    tobii_input_template=tobii_in_temp,
                    annot_input_template=annotations_lie_in_temp,
                    refer_to_baseline=True, refer_mode='sub',
                    clean_mode="MAD",
                    clean=True,
                    smooth=False,
                    with_vad=False,
                    sound_input_template=sound_in_temp):

    root, root_pilot, root_front = annot_var_init()
    print("LOAD s{} from {}".format(subject, source))

    root = root_front
    if(source == "pilot"):
        root = root_pilot

    # Complete the files
    eye_in = tobii_input_template.format(root, subject)
    annot_in = annot_input_template.format(root, subject)
    sound_in = sound_input_template.format(root, subject)
    
    # vector with temporal filtered data for each card
    filtered_interaction_dfs = []
    
    # load tobii data
    eye = preprocessEye(eye_in, source)

    # Load annotations
    annotations = preprocessLieAnnotations(annot_in, card_names)

    if(with_vad):
        sound_annotations = preprocess_vad_annotations(sound_in, annotations)


    # filter eye data relative to the entire phase
    overall = filterEyeData(eye, annotations[0], \
        start_col="start", stop_col="stop", \
        clean=clean, clean_mode=clean_mode, smooth=smooth)

    # Extract the baseline to rescale the pupil dilation data
    baseline = filterBaseline(eye, annotations[0], \
        start_col="start", stop_col="stop", \
        clean=clean, clean_mode=clean_mode, smooth=smooth, window=5000)

    if(refer_to_baseline):
        # Refer the Pupil data to the baseline
        overall = referToBaseline(overall, baseline, mode=refer_mode,
        source_cols=['diam_right'], baseline_col='diam_right')

        overall = referToBaseline(overall, baseline, mode=refer_mode,
        source_cols=['diam_left'], baseline_col='diam_left')


    for i in range(1, len(annotations)):        

        if(with_vad):
            robot_interaction, subject_interaction = vad_rt_data_filtering(overall, sound_annotations[i-1], clean=False, smooth=False)
        else:
            if(i < (len(annotations)-1)):
                robot_interaction, subject_interaction = rt_data_filtering(overall, annotations[i], annotations[i+1], clean=False, smooth=False)
            else:
                robot_interaction, subject_interaction = rt_data_filtering(overall, annotations[i], pd.DataFrame(), clean=False, smooth=False)
        
        robot_interaction['card'] = annotations[i]['card'].iloc[0]
        subject_interaction['card'] = annotations[i]['card'].iloc[0]

        filtered_interaction_dfs.append(
            (robot_interaction, subject_interaction)
        )

    return eye, annotations, baseline, overall, filtered_interaction_dfs



def loadLieTimeSeries(subject, card_names,
                    source="frontiers",
                    tobii_input_template=tobii_in_temp,
                    annot_input_template=annotations_lie_in_temp,
                    refer_to_baseline=True, refer_mode='sub',
                    clean_mode="MAD",
                    clean=True,
                    smooth=False):
    
    root, root_pilot, root_front = annot_var_init()
    print("LOAD s{} from {}".format(subject, source))

    root = root_front
    if(source == "pilot"):
        root = root_pilot

    # Complete the files
    eye_in = tobii_input_template.format(root, subject)
    annot_in = annot_input_template.format(root, subject)
    
    # vector with temporal filtered data for each card
    filtered_interaction_dfs = []
    filtered_interaction__size = []
    
    # load tobii data
    eye = preprocessEye(eye_in, source)
    
    # load novelty annotations
    # Use Lie One
    annotations = preprocessLieAnnotations(annot_in, card_names)
    
    # filter eye data relative to the entire phase
    overall = filterEyeData(eye, annotations[0], \
        start_col="start", stop_col="stop", \
        clean=clean, clean_mode=clean_mode, smooth=smooth)

    # Extract the baseline to rescale the pupil dilation data
    baseline = filterBaseline(eye, annotations[0], \
        start_col="start", stop_col="stop", \
        clean=clean, clean_mode=clean_mode, smooth=smooth, window=5000)

    if(refer_to_baseline):
        # Refer the Pupil data to the baseline
        overall = referToBaseline(overall, baseline, mode=refer_mode,
        source_cols=['diam_right'], baseline_col='diam_right')

        overall = referToBaseline(overall, baseline, mode=refer_mode,
        source_cols=['diam_left'], baseline_col='diam_left')
    
    for i in range(1, len(annotations)):

        whole_interval, point_interaction, reaction_interval, point_reaction_interval, description_interval = \
            lieDataFiltering(overall, annotations[i], clean=False, smooth=False)
        
        point_interaction['card'] = annotations[i]['card'].iloc[0]
        reaction_interval['card'] = annotations[i]['card'].iloc[0]
        point_reaction_interval['card'] = annotations[i]['card'].iloc[0]
        description_interval['card'] = annotations[i]['card'].iloc[0]

        filtered_interaction_dfs.append(
            (whole_interval, point_interaction, reaction_interval, point_reaction_interval, description_interval)
        )

        start_point = annotations[i]['start_p'].iloc[0]
        stop_point = annotations[i]['stop_p'].iloc[0]
        start_descr = annotations[i]['start_d'].iloc[0]
        stop_descr = annotations[i]['stop_d'].iloc[0]

        filtered_interaction__size.append(
            [
                subject,
                annotations[0]['subject_card'].iloc[0],
                annotations[i]['card'].iloc[0],
                whole_interval['diam_right'].count(),
                point_interaction['diam_right'].count(),
                reaction_interval['diam_right'].count(),
                point_reaction_interval['diam_right'].count(),
                description_interval['diam_right'].count(),
                whole_interval['diam_left'].count(),
                point_interaction['diam_left'].count(),
                reaction_interval['diam_left'].count(),
                point_reaction_interval['diam_left'].count(),
                description_interval['diam_left'].count(),
                (stop_descr - start_point), #whole
                (stop_point - start_point), #point
                (start_descr - stop_point), #reaction
                (start_descr - start_point), #point_reaction
                (stop_descr - start_descr) #description
            ]
        )
    
    return eye, annotations, baseline, overall, filtered_interaction_dfs, filtered_interaction__size