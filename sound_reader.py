import pandas as pd
import numpy as np
from pydub import AudioSegment
from inaSpeechSegmenter import Segmenter, seg2csv
import os

seg = Segmenter()
temp_audio_filename = "./VAD/temp.wav"

def preprocess_vad_annotations(sound_in, annotations):

    audio = AudioSegment.from_wav(sound_in)
    sounds = []

    for i in range(1, len(annotations)):

        # Extract audio interval
        annot = annotations[i]
        annot = annot.reset_index()
        start_point_0 = annot.at[0, 'start_p']
        stop_point_0 = annot.at[0, 'stop_p']
        stop_descr_0 = annot.at[0, 'stop_d']

        start_cut = stop_point_0

        if(i < len(annotations)-1):
            annot_1 = annotations[i+1]
            annot_1 = annot_1.reset_index()
            start_point_1 = annot_1.at[0, 'start_p']
            stop_cut = start_point_1
        else:
            stop_cut = stop_descr_0

        audio_cut = audio[start_cut:stop_cut]
        audio_cut.export(temp_audio_filename, format="wav")


        # Apply VAD
        segmentation = seg(temp_audio_filename)

        # Aggregate
        vad_descr_start = None
        vad_descr_stop = None

        for (label, start, stop) in segmentation:
            if(label is not 'noise'):
                if(stop - start >= 1.2):
                    if(vad_descr_start == None):
                        vad_descr_start = start
                
                    vad_descr_stop = stop

        # Shift
        vad_descr_start = (vad_descr_start * 1000) + start_point_0
        vad_descr_stop = (vad_descr_stop * 1000) + start_point_0

        # Append and clear
        sounds.append((start_point_0, stop_point_0, vad_descr_start, vad_descr_stop))
        os.remove(temp_audio_filename) 

    
    return sounds
        