# 모델별 최종 HyperParamers
import numpy as np 

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
xgb_best_params_santander = {
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

# XGB CreditCardFraud
xgb_best_params = { # 전처리 후 org data로 한 경우
    "colsample_bytree": 1.0,
    "gamma": 1.7089112007254605,
    "learning_rate": 0.038583724829444874,
    "max_depth": 3,
    "min_child_weight": 6,
    "n_estimators": 150,
    "reg_alpha": 0.5940048544595263,
    "reg_lambda": 0.007186370173139192,
    "scale_pos_weight": 82.0,
    "subsample": 0.6,
    "random_state": 23,
}

xgb_best_params_smote = {
    'colsample_bytree': 0.8,
    'gamma': 0.9657494843461217,
    'learning_rate': 0.07239930133055257,
    'max_depth': 4,
    'min_child_weight': 5,
    'n_estimators': 450,
    'reg_alpha': 0.7766700326389675,
    'reg_lambda': 8.599432523914023,
    'scale_pos_weight': 5.0,
    'subsample': 0.6000000000000001,
    "eval_metric": "auc",
    "use_label_encoder": False,
    "n_jobs": -1,
    'random_state': 23
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
    'C': 0.029, 
    'max_iter': 1000, 
    'penalty': 'l2', 
    'solver': 'lbfgs',
    'class_weight': 'balanced', 
    "n_jobs": -1     
}

# GradientBoostingClassifier 기본 파라미터
gb_basic_params = {
    'n_estimators': 100,      # 트리 개수 (더 많을수록 성능↑, 시간↑)
    'learning_rate': 0.1,     # 학습률 (0.01~0.3, 작을수록 안정적)
    'max_depth': 3,           # 개별 트리 깊이 (3~5 권장, 과적합 방지)
    'random_state': 23
}

# DecisionTreeClassifier 기본 파라미터
dt_basic_params = {
    'criterion': 'gini',      # 분할 기준 ('gini' 또는 'entropy')
    'max_depth': None,        # 트리 최대 깊이 (None=제한없음, 과적합 위험)
    'random_state': 23
}

# MLPClassifier 기본 파라미터
mlp_basic_params = {
    'hidden_layer_sizes': (64, 32),  # 은닉층 구조 (64개 노드 → 32개 노드)
    'activation': 'relu',             # 활성화 함수 (relu, tanh, logistic)
    'solver': 'adam',                 # 최적화 알고리즘 (adam, sgd, lbfgs)
    'alpha': 1e-4,                    # L2 정규화 파라미터 (과적합 방지)
    'batch_size': 256,                # 배치 크기 (메모리 허용 범위 내 크게)
    'learning_rate': 'adaptive',      # 학습률 스케줄 (adaptive, constant, invscaling)
    'max_iter': 50,                   # 최대 반복 횟수 (early stopping 권장)
    'random_state': 23
}

# SVM liner basic params
svc_liner_basic_params = {
  'C' : 1.0, 
  'kernel' : "linear", 
  'class_weight' : "balanced"
}

# RBF SVM basic params
svc_rbf_basic_params = {
  'C' : 1.0, 
  'kernel' : "rbf", 
  'gamma' : "scale",
  'class_weight' : "balanced"  
}

# SGD
sgd_best_params = {
    'alpha': 3.910947175889396e-05,
    'class_weight': None,  # 0 → None (첫 번째 옵션)
    'eta0': 0.0024094779453352057,
    'l1_ratio': 0.6092668391045077,
    'learning_rate': 'adaptive',  # 3 → 'adaptive' (네 번째 옵션)
    'loss': 'log_loss',  # 1 → 'log_loss' (두 번째 옵션)
    'max_iter': 6000,
    'penalty': 'elasticnet',  # 2 → 'elasticnet' (세 번째 옵션)
    'tol': 1.8099178032063352e-05,
    'random_state': 23
}