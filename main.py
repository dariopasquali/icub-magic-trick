from featureExtractor import *


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


lie_features = lie_features.fillna(0)
lie_features['descr_mean_pupil'] = (lie_features['descr_right_mean'] + lie_features['descr_left_mean'])/2
lie_features['react_mean_pupil'] = (lie_features['react_right_mean'] + lie_features['react_left_mean'])/2
lie_features['point_mean_pupil'] = (lie_features['point_right_mean'] + lie_features['point_left_mean'])/2

lie_feat_cols.append('descr_mean_pupil')
lie_feat_cols.append('react_mean_pupil')
lie_feat_cols.append('point_mean_pupil')

significant_cols.append('descr_mean_pupil')
significant_cols.append('react_mean_pupil')
significant_cols.append('point_mean_pupil')

#points_cols.append('descr_mean_pupil')
#points_cols.append('react_mean_pupil')
#points_cols.append('point_mean_pupil')
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


#"""
print("PLOT COMPARE BARS =================================================")
lie_plotComparBars(lie_features, feat_cols=lie_feat_cols, save_root="V2_clear_35/plot/hd/bars_{}.png", save=True)
#print("PLOT POINTS FOR EACH SUBJECT =================================================")
#lie_plotPointsAllSubjects(lie_features, feat_cols=points_cols, save_root="V2_clear_35/plot/hd/{}.png", mode=mode, save=True)
#"""


"""
# ============= HEURISTIC AND STATISTIC TESTS ===========================

# TAKE MAX HEURISIC
sys.stdout = open("V2_clear_35/reports/take_max_heuristic.txt", "w")
print("======================================================")
take_max_heuristic(lie_features, significant_cols, print_result=True, only_rel=True)
"""

"""
# PAIRED T TEST
sys.stdout = open("V2_clear_35/reports/paired_t_test.txt", "w")
print("======================================================")
paired_t_test(lie_features, lie_feat_cols, significant_cols, print_result=True, only_rel=False)
"""

#lie_features['premed_score_right'] = lie_features['react_right_mean'] / lie_features#['descr_right_mean']
#lie_features['premed_score_left'] = lie_features['react_left_mean'] / lie_features['descr_left_mean']
#lie_feat_cols.append('premed_score_right')
#lie_feat_cols.append('premed_score_left')
#significant_cols.append('premed_score_right')
#significant_cols.append('premed_score_left')
#reduced_significant_cols.append('premed_score_right')
#reduced_significant_cols.append('premed_score_left')
#tt_sign_cols_2.append('premed_score_right')
#tt_sign_cols_2.append('premed_score_left')


"""
col_sets = {
    'all_columns' : significant_cols,
    'reduced' : reduced_significant_cols,
    'tt_0' : tt_sign_cols_all,
    'tt_35' : tt_sign_cols_35,
    #'tt_1' : tt_sign_cols_clear,
    #'descr_react' : descr_react_col_set
}

sys.stdout = open("V2_clear_35/reports/multiple_grid_search_NOnorm_log.txt", "w")

gsEngine = GridSearchEngine()
gsEngine.add_naive_bayes()
gsEngine.add_knn()
gsEngine.add_ada()
gsEngine.add_svm()
gsEngine.add_decision_tree()
gsEngine.add_random_forest()
gsEngine.add_mlp()

report = gsEngine.multiple_grid_search(lie_features, col_sets=col_sets, norm_by_subject=False)
report.to_csv("V2_clear_35/reports/MGS_NOnorm_report.csv", sep='\t')
"""
