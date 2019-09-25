from featureExtractor import *
from numpy.random import seed
seed(42)


mode = "sub"
print("================== MODE SMOOTH {} =====================".format(mode))
#lie_features, lie_base_right, lie_base_left, tobii_stats = \
#    extractLieFeatures(subjects=[], subject_to_exclude=[5, 16, 19, 22, 24, 26], mode=mode, ref_to_base="time", plot=False, save=False, save_root="plots/V2/pupils_{}.png")
#lie_features.to_csv("lie_features_clear_whole_35.csv", index=False)

#quality = calc_tobii_quality_features(tobii_stats)
#quality.to_csv("quality_tobii_50.csv", index=False)
#quality = pd.read_csv("quality_tobii_50.csv", sep=',')
#subject_to_exclude = filter_by_quality(quality, 20.0, 20.0)

#print(subject_to_exclude)




#"""
#lie_features = pd.read_csv("lie_features_smooth.csv", sep=',')
lie_features = pd.read_csv("lie_features_clear_whole_35.csv", sep=',')
#lie_features = pd.read_csv("lie_features_clear.csv", sep=',')


lie_features = lie_features.dropna()
lie_features['descr_mean_pupil'] = (lie_features['descr_right_mean'] + lie_features['descr_left_mean'])/2
lie_features['react_mean_pupil'] = (lie_features['react_right_mean'] + lie_features['react_left_mean'])/2
lie_features['point_mean_pupil'] = (lie_features['point_right_mean'] + lie_features['point_left_mean'])/2

lie_feat_cols.append('descr_mean_pupil')
lie_feat_cols.append('react_mean_pupil')
lie_feat_cols.append('point_mean_pupil')

significant_cols.append('descr_mean_pupil')
significant_cols.append('react_mean_pupil')
significant_cols.append('point_mean_pupil')

#"""
lie_features['premed_score_right'] = lie_features['react_right_mean'] / lie_features['descr_right_mean']
lie_features['premed_score_left'] = lie_features['react_left_mean'] / lie_features['descr_left_mean']
lie_feat_cols.append('premed_score_right')
lie_feat_cols.append('premed_score_left')
significant_cols.append('premed_score_right')
significant_cols.append('premed_score_left')
#"""

# PAIRED T TEST
"""
right_left_means = [
    ('right_mean', 'left_mean'),
    ('point_right_mean', 'point_left_mean'),
    ('react_right_mean', 'react_left_mean'),
    ('descr_right_mean', 'descr_left_mean')
]
sys.stdout = open("V2_clear_35/reports/paired_t_test_pupil_dilation.txt", "w")
print("======================================================")
paired_t_test_pupil_dilation(lie_features, right_left_means, print_result=True)
"""


"""
print("PLOT COMPARE BARS =================================================")
lie_plotComparBars(lie_features, feat_cols=lie_feat_cols, save_root="V2_clear_35/plot/hd/bars_NONAN_{}.png", save=True)
print("PLOT POINTS FOR EACH SUBJECT =================================================")
lie_plotPointsAllSubjects(lie_features, feat_cols=points_cols, save_root="V2_clear_35/plot/hd/NONAN_{}.png", mode=mode, save=True)
"""

"""
# ============= NORMALITY TEST ==========================================
sys.stdout = open("V2_clear_35/reports/NONAN_normality.txt", "w")
print("============= NORMALITY TEST =========================")
normality_test(lie_features, 'right_mean', mode=0)
normality_test(lie_features, 'left_mean', mode=0)
normality_test(lie_features, 'descr_right_mean', mode=0)
normality_test(lie_features, 'descr_left_mean', mode=0)
normality_test(lie_features, 'react_right_mean', mode=0)
normality_test(lie_features, 'react_left_mean', mode=0)
sys.stdout = open("V2_clear_35/reports/NONAN_normality_1.txt", "w")
print("============= NORMALITY TEST =========================")
normality_test(lie_features, 'right_mean', mode=1)
normality_test(lie_features, 'left_mean', mode=1)
normality_test(lie_features, 'descr_right_mean', mode=1)
normality_test(lie_features, 'descr_left_mean', mode=1)
normality_test(lie_features, 'react_right_mean', mode=1)
normality_test(lie_features, 'react_left_mean', mode=1)
"""

"""
# ============= HEURISTIC AND STATISTIC TESTS ===========================

# TAKE MAX HEURISIC
sys.stdout = open("V2_clear_35/reports/NONAN_take_max_heuristic.txt", "w")
print("======================================================")
take_max_heuristic(lie_features, significant_cols, print_result=True, only_rel=True)
#"""

# ============= MAX BEST PUPIL HEURISTIC TESTS ===========================

"""
# TAKE MAX HEURISIC
sys.stdout = open("V2_clear_35/reports/NONAN_take_best_max_heuristic.txt", "w")
print("======================================================")
take_max_best_pupil_heuristic(lie_features, 'descr', print_result=True, only_rel=False)
take_max_best_pupil_heuristic(lie_features, 'react', print_result=True, only_rel=False)
take_max_best_pupil_heuristic(lie_features, 'point', print_result=True, only_rel=False)
"""

"""
# PAIRED T TEST
sys.stdout = open("V2_clear_35/reports/NONAN_paired_t_test.txt", "w")
print("======================================================")
tt_cols = [c for c in lie_feat_cols if c not in ('subject', 'card_class', 'label', 'source', 'show_order')]
paired_t_test(lie_features, lie_feat_cols, tt_cols, mode=0, print_result=True, only_rel=False)

sys.stdout = open("V2_clear_35/reports/NONAN_paired_t_test_wilcoxon.txt", "w")
print("======================================================")
tt_cols = [c for c in lie_feat_cols if c not in ('subject', 'card_class', 'label', 'source', 'show_order')]
paired_t_test(lie_features, lie_feat_cols, tt_cols, mode=1, print_result=True, only_rel=False)
"""


#"""
tt_sign_cols_35 = [
    'descr_left_max',
    'react_left_std',
    'react_left_max',
    'descr_right_max',
    'right_max',
    'left_std',
    'react_right_mean',
    'react_mean_pupil',
    'react_left_mean',
    'right_mean',
    'left_mean',
    'descr_right_mean',
    'descr_left_mean',
    'descr_mean_pupil',
]

tt_sign_cols_wilcoxon = [
    'react_left_std',
    'react_right_min',
    'right_max',
    'descr_right_max',
    'left_std',
    'react_right_mean',
    'react_mean_pupil',
    'left_mean',
    'react_left_mean',
    'right_mean',
    'descr_right_mean',
    'descr_mean_pupil',
    'descr_left_mean',
]

tt_sign_cols_35_premed = [
    'descr_left_max',
    'react_left_std',
    'react_left_max',
    'descr_right_max',
    'right_max',
    'left_std',
    'react_right_mean',
    'react_mean_pupil',
    'react_left_mean',
    'right_mean',
    'left_mean',
    'descr_right_mean',
    'descr_left_mean',
    'descr_mean_pupil',
    'premed_score_right',
    'premed_score_left'
]

tt_sign_cols_wilcoxon_premed = [
    'react_left_std',
    'react_right_min',
    'right_max',
    'descr_right_max',
    'left_std',
    'react_right_mean',
    'react_mean_pupil',
    'left_mean',
    'react_left_mean',
    'right_mean',
    'descr_right_mean',
    'descr_mean_pupil',
    'descr_left_mean',
    'premed_score_right',
    'premed_score_left'
]

col_sets = {
    'all_columns' : significant_cols,
    'reduced' : reduced_significant_cols,
    'tt_0' : tt_sign_cols_all,
    'tt_35' : tt_sign_cols_35,
    'tt_wilcoxon' : tt_sign_cols_wilcoxon,
    'tt_35_premed' : tt_sign_cols_35_premed,
    'tt_wilcoxon_premed' : tt_sign_cols_wilcoxon_premed,
}

sys.stdout = open("V2_clear_35/reports/multiple_grid_search_mean_std_seed_log.txt", "w")

gsEngine = GridSearchEngine()
gsEngine.add_naive_bayes()
gsEngine.add_knn()
gsEngine.add_ada()
gsEngine.add_svm()
gsEngine.add_decision_tree()
gsEngine.add_random_forest()
gsEngine.add_mlp()

report = gsEngine.multiple_grid_search(lie_features, col_sets=col_sets, norm_by_subject=True)
report.to_csv("V2_clear_35/reports/MGS_report_mean_std_seed.csv", sep='\t')
#"""
