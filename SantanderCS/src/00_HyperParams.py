# 모델별 최종 HyperParamers

# RF
rf_best_param = {
    "random_state": 23,
    "n_estimators": 390,
    "max_depth": 25,
    "class_weight": {0: 1, 1: 2},
    "min_samples_leaf": 1,
    "min_samples_split": 7,
    "n_jobs": -1
}

# XGB
xgb_best_params = {
    "random_state": 23,
    "n_estimators": 320,
    "colsample_bytree": 0.88,
    "gamma": 0.058,
    "learning_rate": 0.13,
    "max_depth": 6,
    "scale_pos_weight": 10,
    "min_child_weight": 2,
    "subsample": 0.85,
    "eval_metric": "auc",
    "use_label_encoder": False,
    "n_jobs": -1,
}

# LGBM
lgbm_best_param = {
    'random_state' : 23,
    'n_estimators' : 400,
    'num_leaves' : 36,
    'learning_rate' : 0.03,
    'subsample' : 0.9,
    'colsample_bytree' : 0.75,
    'reg_alpha' : 0.6,
    'reg_lambda' : 0.2,
    'class_weight' : {0:1, 1:10},
    'n_jobs' : -1    
}

# LR
meta_best_params = {
    "random_state": 23,
    'C': np.float64(0.040779926643605094), 
    'class_weight': None, 
    'max_iter': int(500.0), 
    'penalty': 'l2', 
    'solver': 'lbfgs',
    "n_jobs": -1
}
