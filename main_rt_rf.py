from featureExtractor import *
from numpy.random import seed
import time

import csv

seed(42)
mode = "sub"

#to_exclude = [5, 16, 19, 22, 24, 26]
to_exclude = []
output_filename="./RT/models/random_forest.joblib"
card_names = ['unicorn', 'pepper', 'minion', 'pig', 'hedge', 'aliens']

def train_random_forest_model(dataset, filename):

    random_forest_features = [
        'react_left_std',
        'react_left_mean',
        'react_right_min',
        'react_right_mean',
        'react_mean_pupil',        
        'right_max',
        'left_std',
        'left_mean',
        'right_mean',
        'descr_right_max',
        'descr_right_mean',
        'descr_mean_pupil',
        'descr_left_mean',
    ]

    model = train_random_forest(dataset, random_forest_features, output_filename=filename)
    return model

def load_model(filename):

    with open(filename, "rb") as input_file:
        random_forest = load(filename)
    
    return random_forest

def load_dataset(filename):
    lie_features = pd.read_csv(filename, sep=',')
    lie_features = lie_features.dropna()
    lie_features['descr_mean_pupil'] = (lie_features['descr_right_mean'] + lie_features['descr_left_mean'])/2
    lie_features['react_mean_pupil'] = (lie_features['react_right_mean'] + lie_features['react_left_mean'])/2
    lie_features['point_mean_pupil'] = (lie_features['point_right_mean'] + lie_features['point_left_mean'])/2
    lie_features['premed_score_right'] = lie_features['react_right_mean'] / lie_features['descr_right_mean']
    lie_features['premed_score_left'] = lie_features['react_left_mean'] / lie_features['descr_left_mean']

    return lie_features


class RealTimePredictionSimulator:

    def __init__(self, annotations, tobii_streaming, model, baseline_window_size_ms, window_size_ms):

        self.annotations = annotations
        self.tobii_streaming = tobii_streaming
        self.model = model
        self.window_size_ms = window_size_ms
        self.baseline_window_size_ms = baseline_window_size_ms

        # Pointing State ==================================
        self.start_p0 = None
        self.stop_p0 = None
        self.stop_d0 = None
        
        self.start_p1 = None

        # Bin State =======================================
        self.bin_start = None
        self.bin_end = None

        # Baseline =========================================
        self.baseline_left = 0.0
        self.baseline_right = 0.0
        self.baseline_done = False

        # Flow Control =========================================
        self.tobii_reader = csv.reader(self.tobii_streaming)
        self.keep_reading = True
        self.last_point_done = False
        self.is_first_pointing = True

        # Storage ==============================================
        self.tobii_data = pd.DataFrame(columns=['timestamp', 'name', 'rec_date', 'start_time', 'diam_left', 'diam_right'])
        self.bin = None
        
    def store_tobii_data(self, eye_data):        
        new_row = pd.DataFrame(data=[eye_data], columns=['timestamp', 'name', 'rec_date', 'start_time', 'diam_left', 'diam_right'])
        self.tobii_data = self.tobii_data.append(new_row, ignore_index=True)
        return True

    def store_tobii_data_bin(self, eye_data):
        new_row = pd.DataFrame(data=[eye_data], columns=['timestamp', 'name', 'rec_date', 'start_time', 'diam_left', 'diam_right'])
        self.bin = self.bin.append(new_row, ignore_index=True)
        return True

    def calc_baseline(self):
        # TODO
        return True

    def refer_to_baseline(self):
        # TODO
        return True

    def aggregate_bin(self):
        # TODO
        return True

    def predict_lable_and_store(self):
        # TODO
        return True

    def aggregate_and_store_predictions(self):
        # TODO
        return True

    
    def read_from_streaming(self):
        
        eye_data = None

        try:
            eye_data = next(self.tobii_reader)
        except:
            self.keep_reading = False
            return

        # Store new row
        self.store_tobii_data(eye_data)

        # Extract timestamp
        timestamp = int(eye_data[0])

        # if it's the first pointing I need to calculate the baseline
        if(self.is_first_pointing):
            if(not self.baseline_done):
                # But only if the fist pointing is running
                if(timestamp > self.start_p0):
                    self.calc_baseline()
                    self.baseline_done = True
                    return
        
        # If the first pointing is still running keep reading data
        if(timestamp < self.stop_p0):
            return

        # if I don't have a bin it means that it is the first bin
        if(self.bin_start == None):
            # Create an empty bin
            self.bin = pd.DataFrame(columns=['timestamp', 'name', 'rec_date', 'start_time', 'diam_left', 'diam_right'])
            # Init time limits
            self.bin_start = self.stop_p0
            self.bin_end = min(self.start_p1, self.bin_start + self.window_size_ms)

        # if the timestamp is inside the bin
        if(timestamp < self.bin_end):
            # Store the data in the bin
            self.store_tobii_data_bin(eye_data)
        else:
            # process the bin
            self.refer_to_baseline()
            self.aggregate_bin()
            self.predict_lable_and_store()

            # If I'm still in this pointing
            if(timestamp < self.start_p1):
                # Create an empty bin
                self.bin = pd.DataFrame(columns=['timestamp', 'name', 'rec_date', 'start_time', 'diam_left', 'diam_right'])
                # Init time limits
                self.bin_start = self.bin_end
                self.bin_end = min(self.start_p1, self.bin_start + self.window_size_ms)
            else:
                # I finished this pointing
                self.aggregate_and_store_predictions()
                self.bin_end = None
                self.bin_start = None
                self.keep_reading = False
            
            if(self.is_first_pointing):
                self.is_first_pointing = False

            return


    def simulate(self):

        for i, annot in enumerate(self.annotations):
            
            annot = annot.reset_index()
            self.start_p0 = annot.at[0, 'start_p']
            self.stop_p0 = annot.at[0, 'stop_p']
            self.stop_d0 = annot.at[0, 'stop_d']

            if(i < len(self.annotations)-1):
                annot_1 = self.annotations[i+1].reset_index()
                self.start_p1 = annot_1.at[0, 'start_p']
            else:
                self.start_p1 = self.stop_d0
            
            self.keep_reading = True

            while(self.keep_reading):
                self.read_from_streaming()

            if(self.last_point_done):
                self.tobii_streaming.close()





#dataset = load_dataset("lie_features_clear_whole_35.csv")
#dumped = train_random_forest_model(dataset, output_filename)
#print(dumped)

random_forest = load_model(output_filename)

annotations = load_rt_annotations(0, card_names)
with create_rt_eye_streaming(0) as eye_streaming:

    simulator = RealTimePredictionSimulator(annotations[1:], eye_streaming, random_forest, 5000, 1000)
    simulator.simulate()