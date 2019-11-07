from featureExtractor import *
from numpy.random import seed
import time
seed(42)
mode = "sub"

"""
rt_heuristic_features = extract_rt_lie_features( \
        subjects=[], subject_to_exclude=[5, 16, 19, 22, 24, 26],\
        mode=mode, ref_to_base="time", plot=False, save=False, save_root="plots/V2/pupils_{}.png")

#rt_heuristic_features.to_csv("lie_features_real_time_heuristic.csv", index=False)
#rt_heuristic_features = pd.read_csv("lie_features_real_time_heuristic.csv", sep=',')



# ============= HEURISTIC AND STATISTIC TESTS ===========================

# TAKE MAX HEURISIC
sys.stdout = open("RT/reports/heuristic_annots_all.txt", "w")
print("======================================================")
take_max_heuristic(rt_heuristic_features, ['rt_right_mean', 'rt_left_mean', 'rt_mean_pupil'], print_result=True, only_rel=True)
"""

start = int(round(time.time() * 1000))
rt_heuristic_features = extract_rt_lie_features_single_subject(3, mode=mode, ref_to_base="time", with_vad=True)
print(int(round(time.time() * 1000)) - start)





def predict_take_max(features, col):
        max_f = features[col].max()
        max_label = features.loc[features[col] == max_f]['label'].values[0]
        max_card = features.loc[features[col] == max_f]['card_class'].values[0]

        check = max_label == 1

        return max_card, check

"""
subjects = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27]
#subjects = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,17,18,20,21,25,27]

deltas = []

for i in range(10):
        print(i)
        for sub in subjects:

                start = int(round(time.time() * 1000))

                rt_heuristic_features = extract_rt_lie_features_single_subject(sub, mode=mode, ref_to_base="time")
                predict_take_max(rt_heuristic_features, 'rt_left_mean')

                deltas.append((int(round(time.time() * 1000)) - start))


mean = 0
for d in deltas:
        mean += d

mean = mean/(len(subjects)*10)

print ("Mean Time %d" % mean)
"""

