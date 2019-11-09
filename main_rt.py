from featureExtractor import *
from numpy.random import seed
import time
seed(42)
mode = "sub"

#to_exclude = [5, 16, 19, 22, 24, 26]
to_exclude = []

"""
rt_heuristic_features = extract_rt_lie_features( subjects=[], subject_to_exclude=to_exclude, mode=mode, ref_to_base="time", with_vad=True)

print("Save to CSV")
rt_heuristic_features.to_csv("lie_features_real_time_heuristic__vad_all.csv", index=False)
#rt_heuristic_features = pd.read_csv("lie_features_real_time_heuristic.csv", sep=',')



# ============= HEURISTIC AND STATISTIC TESTS ===========================
print("Evaluate Heuristic")
# TAKE MAX HEURISIC
sys.stdout = open("RT/reports/heuristic_annots__vad_all.txt", "w")
print("======================================================")
take_max_heuristic(rt_heuristic_features, ['rt_right_mean', 'rt_left_mean', 'rt_mean_pupil'], print_result=True, only_rel=True)
"""

"""
start = int(round(time.time() * 1000))
rt_heuristic_features = extract_rt_lie_features_single_subject(0, mode=mode, ref_to_base="time", with_vad=True)
print(int(round(time.time() * 1000)) - start)
"""




def predict_take_max(features, col):
        max_f = features[col].max()
        max_label = features.loc[features[col] == max_f]['label'].values[0]
        max_card = features.loc[features[col] == max_f]['card_class'].values[0]

        check = max_label == 1

        return max_card, check

"""
#subjects = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27]
subjects = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,17,18,20,21,25,27]

deltas = []

for i in range(1):
        print(i)
        for sub in subjects:

                start = int(round(time.time() * 1000))

                rt_heuristic_features = extract_rt_lie_features_single_subject(sub, mode=mode, ref_to_base="time", with_vad=True)
                predict_take_max(rt_heuristic_features, 'rt_left_mean')

                deltas.append((int(round(time.time() * 1000)) - start))


mean = 0
for d in deltas:
        mean += d

mean = mean/(len(subjects))

print ("Mean Time %d" % mean)
"""

"""
#rt_heuristic_features = extract_rt_lie_features(subjects=[], subject_to_exclude=to_exclude, mode=mode, ref_to_base="time", with_vad=False)
rt_heuristic_features.to_csv("lie_features_real_time___8_features.csv", index=False)

# ============= HEURISTIC AND STATISTIC TESTS ===========================
print("Evaluate Heuristic")
# TAKE MAX HEURISIC
sys.stdout = open("RT/reports/heuristic_annots___8features.txt", "w")
print("======================================================")
take_max_heuristic(rt_heuristic_features, \
        ['right_mean', 'right_std', 'right_min', 'right_max', 'left_mean', 'left_std', 'left_min', 'left_max', 'mean_pupil'],\
                 print_result=True, only_rel=True)
"""


rt_features = ['right_mean', 'right_std', 'right_min', 'right_max', 'left_mean', 'left_std', 'left_min', 'left_max', 'mean_pupil']
rt_heuristic_features = pd.read_csv("lie_features_real_time___8_features.csv", sep=',')

rt_heuristic_features['source'] = ""
rt_heuristic_features['show_order'] = 0

col_sets = {
    '8_features' : rt_features
}

sys.stdout = open("RT/reports/multiple_grid_search___whole_interv_8_features_nonorm.txt", "w")
gsEngine = GridSearchEngine()
gsEngine.add_naive_bayes()
gsEngine.add_knn()
gsEngine.add_ada()
gsEngine.add_svm()
gsEngine.add_decision_tree()
gsEngine.add_random_forest()
gsEngine.add_mlp()

report = gsEngine.multiple_grid_search(rt_heuristic_features, col_sets=col_sets, norm_by_subject=False)
report.to_csv("RT/reports/MGS_report___whole_interv_8_features_nonorm.csv", sep='\t')