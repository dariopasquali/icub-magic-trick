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

tobii_columns = ['timestamp', 'name', 'rec_date', 'start_time', 'diam_left', 'diam_right']
feature_columns = ['subject', 'card_class', 'label', 'right_mean', 'right_std', 'right_min', 'right_max', 'left_mean', 'left_std', 'left_min', 'left_max']
prediction_columns = ['subject', 'card_class', 'from', 'to', 'prediction', 'label']

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

    def __init__(self, subject, annotations, tobii_streaming, model, baseline_window_size_ms, window_size_ms):

        self.subject = subject
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

        self.predictions = pd.DataFrame(columns=prediction_columns)
        self.features = pd.DataFrame(columns=feature_columns)
        self.tobii_data = pd.DataFrame(columns=tobii_columns)
        self.bin = None
        
    def store_tobii_data(self, eye_data):

        t = int(eye_data[0])
        name = eye_data[1]
        sd = eye_data[2]
        st = eye_data[3]
        if(eye_data[4] == ''):
            left = 0.0
        else:
            left = float(eye_data[4])
        
        if(eye_data[5] == ''):
            right = 0.0
        else:
            right = float(eye_data[5])

        new_row = pd.DataFrame(data=[[t, name, sd, st, left, right]], columns=tobii_columns)
        self.tobii_data = self.tobii_data.append(new_row, ignore_index=True)


    def store_tobii_data_bin(self, eye_data):
        
        t = int(eye_data[0])
        name = eye_data[1]
        sd = eye_data[2]
        st = eye_data[3]
        if(eye_data[4] == ''):
            left = 0.0
        else:
            left = float(eye_data[4])
        
        if(eye_data[5] == ''):
            right = 0.0
        else:
            right = float(eye_data[5])        

        new_row = pd.DataFrame(data=[[t, name, sd, st, left, right]], columns=tobii_columns)
        self.bin = self.bin.append(new_row, ignore_index=True)
 
    def calc_baseline(self):
        
        start = self.start_p0 - self.baseline_window_size_ms
        stop = self.start_p0

        baseline = self.tobii_data.loc[(self.tobii_data['timestamp'] >= start) & (self.tobii_data['timestamp'] <= stop)]
        baseline = dataCleaner(baseline, clean=True, clean_mode="MAD", smooth=False)

        self.baseline_left = baseline["diam_left"].mean(skipna = True)
        self.baseline_right = baseline["diam_right"].mean(skipna = True)


    def refer_bin_to_baseline(self):
        self.bin['diam_left'] = self.bin['diam_left'] - self.baseline_left
        self.bin['diam_right'] = self.bin['diam_right'] - self.baseline_right


    def aggregate_bin(self, card, label):
        right_mean = self.bin['diam_right'].mean(skipna = True)
        right_std = self.bin['diam_right'].std(skipna = True)
        right_max = self.bin['diam_right'].max(skipna = True)
        right_min = self.bin['diam_right'].min(skipna = True)

        left_mean = self.bin['diam_left'].mean(skipna = True)
        left_std = self.bin['diam_left'].std(skipna = True)
        left_max = self.bin['diam_left'].max(skipna = True)
        left_min = self.bin['diam_left'].min(skipna = True)

        feats = [self.subject, card, label, right_mean, right_std, right_min, right_max, left_mean, left_std, left_min, left_max]
        feats_df = pd.DataFrame(data=[feats], columns=feature_columns)
        self.features = self.features.append(feats_df, ignore_index=True)

        return feats

    def predict_label_and_store(self, feats, card, label):
        pred = self.model.predict(feats)
        prediction = [self.subject, card, self.stop_p0, self.start_p1, label, pred]
        prediction_df = pd.DataFrame(data=[prediction], columns=prediction_columns)
        self.predictions = self.predictions.append(prediction_df, ignore_index=True)


    def aggregate_and_store_predictions(self):
        # TODO
        return True

    
    def read_from_streaming(self, card, label):
        
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
        #print(timestamp)

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
            self.refer_bin_to_baseline()
            self.aggregate_bin(card, label)
            #self.predict_label_and_store(feats)

            # If I'm still in this pointing
            if(timestamp < self.start_p1):
                # Create an empty bin
                self.bin = pd.DataFrame(columns=['timestamp', 'name', 'rec_date', 'start_time', 'diam_left', 'diam_right'])
                # Init time limits
                self.bin_start = self.bin_end
                self.bin_end = min(self.start_p1, self.bin_start + self.window_size_ms)
            else:
                # I finished this pointing interval
                # self.aggregate_and_store_predictions()
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

            card = annot['card'].iloc[0]
            label = annot['label'].iloc[0]

            if(i < len(self.annotations)-1):
                annot_1 = self.annotations[i+1].reset_index()
                self.start_p1 = annot_1.at[0, 'start_p']
            else:
                self.start_p1 = self.stop_d0
            
            self.keep_reading = True

            while(self.keep_reading):
                self.read_from_streaming(card, label)

        # Save the file
        filename = "./RT/features/features8_s{}_w{}.csv".format(self.subject, self.window_size_ms)
        self.features.to_csv(filename, index=False)





#dataset = load_dataset("lie_features_clear_whole_35.csv")
#dumped = train_random_forest_model(dataset, output_filename)
#print(dumped)

"""
random_forest = load_model(output_filename)

annotations = load_rt_annotations(0, card_names)
#with create_rt_eye_streaming(0) as eye_streaming:
with open("./RT/temp_tobii_stream.csv") as eye_streaming:
    simulator = RealTimePredictionSimulator(0, annotations[1:], eye_streaming, random_forest, 5000, 1000)
    simulator.simulate()
"""


#windowed_features = extract_rt_8_features_windowed(subjects=[], subject_to_exclude=to_exclude, mode=mode, ref_to_base="time", window_size=100)
#windowed_features.to_csv("features8_w100.csv", index=False)



rt_features = ['right_mean', 'right_std', 'right_min', 'right_max', 'left_mean', 'left_std', 'left_min', 'left_max', 'mean_pupil']
rt_heuristic_features = pd.read_csv("features8_w1000.csv", sep=',')

rt_heuristic_features['source'] = ""
rt_heuristic_features['show_order'] = 0

col_sets = {
    '8_features' : rt_features
}

sys.stdout = open("RT/reports/multiple_grid_search___window1000_8features_2.txt", "w")
gsEngine = GridSearchEngine()
#gsEngine.add_naive_bayes()
#gsEngine.add_knn()
#gsEngine.add_ada()
#gsEngine.add_svm()
#gsEngine.add_decision_tree()
gsEngine.add_random_forest()
gsEngine.add_mlp()

report = gsEngine.multiple_grid_search(rt_heuristic_features, col_sets=col_sets, norm_by_subject=True)
report.to_csv("RT/reports/MGS_report___window1000_8features_2.csv", sep='\t')