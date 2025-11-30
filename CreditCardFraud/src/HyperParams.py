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
    "n_jobs": -1,
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
xgb_best_params = {  # 전처리 후 org data로 한 경우
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
    "colsample_bytree": 0.8,
    "gamma": 0.9657494843461217,
    "learning_rate": 0.07239930133055257,
    "max_depth": 4,
    "min_child_weight": 5,
    "n_estimators": 450,
    "reg_alpha": 0.7766700326389675,
    "reg_lambda": 8.599432523914023,
    "scale_pos_weight": 5.0,
    "subsample": 0.6000000000000001,
    "eval_metric": "auc",
    "use_label_encoder": False,
    "n_jobs": -1,
    "random_state": 23,
}

# LGBM by YJH
lgbm_best_param = {
    "objective": "binary",  # 예시: 이진 분류 문제인 경우 설정
    "metric": "auc",  # 예시: 성능 평가 지표 설정
    "boosting_type": "gbdt",  # 예시: 기본 GBDT 방식 사용
    # 모델 구조 관련 파라미터
    "max_depth": 5,
    "num_leaves": 111,
    "min_data_in_leaf": 20,
    # 학습 속도 및 반복 횟수
    "learning_rate": 0.14435957617147147,
    "n_estimators": 100,
    # 과적합 방지 (규제 및 샘플링)
    "feature_fraction": 0.9900895952195329,  # Colsample_bytree
    "bagging_fraction": 0.8928311418280319,  # Subsample
    "bagging_freq": 0,
    "lambda_l1": 0.3045393816732618,
    "lambda_l2": 3.3928721015115566,
    # 클래스 불균형 처리
    "scale_pos_weight": 578,
    # 시스템/재현성 설정
    "random_state": 23,
    "n_jobs": -1,
}


# 좀더 세밀하게 HyperOpt 돌렸을때 값. search_space 하단 참조
lgbm_best_param2 = {
    # 모델 구조 관련 파라미터
    "max_depth": 5,  # 트리의 최대 깊이
    "num_leaves": 205,  # 하나의 트리가 가질 최대 리프 노드 수
    "min_data_in_leaf": 25,  # 리프 노드가 되기 위한 최소 데이터 수 (min_child_samples와 유사)
    "min_child_samples": 20,  # 리프 노드가 되기 위한 최소 데이터 수 (min_data_in_leaf와 유사)
    "min_gain_to_split": 0.002445,  # 분할(split)을 위한 최소 이득 (Gain)
    # 학습 속도 및 반복 횟수
    "learning_rate": 0.062768,  # 학습 속도 (이전보다 감소)
    "n_estimators": 450,  # 부스팅 단계 수 (트리 개수, 이전보다 증가)
    # 과적합 방지 (샘플링)
    "feature_fraction": 0.801203,  # 각 트리 학습 시 사용될 특성 샘플링 비율 (Colsample_bytree)
    "bagging_fraction": 0.825535,  # 각 트리 학습 시 사용될 데이터 샘플링 비율 (Subsample)
    "bagging_freq": 4,  # 배깅 수행 빈도 (4번의 이터레이션마다 배깅 수행)
    # 과적합 방지 (규제) - HyperOpt에서 탐색된 상세 값들
    "lambda_l1": 0.512937,  # L1 규제 (Lasso, reg_alpha와 유사)
    "lambda_l2": 1.092e-07,  # L2 규제 (Ridge, reg_lambda와 유사)
    "reg_alpha": 0.548894,  # L1 규제 (lambda_l1과 유사)
    "reg_lambda": 2.435e-08,  # L2 규제 (lambda_l2와 유사)
    # 클래스 불균형 처리
    # 'scale_pos_weight': 3,          # 클래스 불균형을 위한 양성 클래스 가중치 (이전보다 크게 감소)
    "scale_pos_weight": 578,  # 클래스 불균형을 위한 양성 클래스 가중치 (이전보다 크게 감소)
    # 시스템/재현성 설정
    "random_state": 23,  # 재현성을 위한 랜덤 시드
    # 'n_jobs': -1                  # HyperOpt 결과에 없지만 보통 추가하여 사용
}
"""
# 모델별 스페이스 생성 : LightGBM
lgbm_search_space = {
    # 학습률 (로그 스케일이 더 효과적)
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    
    # 트리 구조
    'num_leaves': hp.quniform('num_leaves', 31, 255, 1),  # 2^n-1 권장
    'max_depth': hp.quniform('max_depth', 3, 12, 1),  # 범위 확장
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
    
    # Feature sampling
    'feature_fraction': hp.uniform('feature_fraction', 0.6, 1.0),  # 0.5는 너무 낮음
    'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
    'bagging_freq': hp.quniform('bagging_freq', 1, 7, 1),  # 0 제외 (의미 없음)
    
    # 리프 제약
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 100, 5),  # 200은 너무 큼
    'min_child_samples': hp.quniform('min_child_samples', 5, 50, 5),  # 추가
    
    # 정규화
    'lambda_l1': hp.loguniform('lambda_l1', np.log(1e-8), np.log(10.0)),  # 로그 스케일
    'lambda_l2': hp.loguniform('lambda_l2', np.log(1e-8), np.log(10.0)),
    
    # 불균형 데이터 대응 (개선)
    'scale_pos_weight': hp.choice('scale_pos_weight', [
        1, 
        int(len(y_train) / sum(y_train)),  # 불균형 비율
        int(len(y_train) / sum(y_train)) * 0.5,  # 50%
        int(len(y_train) / sum(y_train)) * 1.5   # 150%
    ]),
    
    # 추가 파라미터 (성능 향상)
    'min_gain_to_split': hp.loguniform('min_gain_to_split', np.log(1e-5), np.log(1.0)),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(10.0)),  # L1
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0)),  # L2
}
"""

lgbm_best_param_santander = {
    "random_state": 23,
    "n_estimators": 400,
    "num_leaves": 36,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.6,
    "reg_lambda": 0.2,
    "class_weight": {0: 1, 1: 10},
    "n_jobs": -1,
}

# LR
meta_best_params = {
    "random_state": 23,
    "C": 0.029,
    "max_iter": 1000,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": "balanced",
    "n_jobs": -1,
}

# GradientBoostingClassifier 기본 파라미터
gb_basic_params = {
    "n_estimators": 100,  # 트리 개수 (더 많을수록 성능↑, 시간↑)
    "learning_rate": 0.1,  # 학습률 (0.01~0.3, 작을수록 안정적)
    "max_depth": 3,  # 개별 트리 깊이 (3~5 권장, 과적합 방지)
    "random_state": 23,
}

# DecisionTreeClassifier 기본 파라미터
dt_basic_params = {
    "criterion": "gini",  # 분할 기준 ('gini' 또는 'entropy')
    "max_depth": None,  # 트리 최대 깊이 (None=제한없음, 과적합 위험)
    "random_state": 23,
}

# MLPClassifier 기본 파라미터
mlp_basic_params = {
    "hidden_layer_sizes": (64, 32),  # 은닉층 구조 (64개 노드 → 32개 노드)
    "activation": "relu",  # 활성화 함수 (relu, tanh, logistic)
    "solver": "adam",  # 최적화 알고리즘 (adam, sgd, lbfgs)
    "alpha": 1e-4,  # L2 정규화 파라미터 (과적합 방지)
    "batch_size": 256,  # 배치 크기 (메모리 허용 범위 내 크게)
    "learning_rate": "adaptive",  # 학습률 스케줄 (adaptive, constant, invscaling)
    "max_iter": 50,  # 최대 반복 횟수 (early stopping 권장)
    "random_state": 23,
}

# SVM liner basic params
svc_liner_basic_params = {"C": 1.0, "kernel": "linear", "class_weight": "balanced"}

# RBF SVM basic params
svc_rbf_basic_params = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": "balanced",
}

# SGD
sgd_best_params = {
    "alpha": 3.910947175889396e-05,
    "class_weight": None,  # 0 → None (첫 번째 옵션)
    "eta0": 0.0024094779453352057,
    "l1_ratio": 0.6092668391045077,
    "learning_rate": "adaptive",  # 3 → 'adaptive' (네 번째 옵션)
    "loss": "log_loss",  # 1 → 'log_loss' (두 번째 옵션)
    "max_iter": 6000,
    "penalty": "elasticnet",  # 2 → 'elasticnet' (세 번째 옵션)
    "tol": 1.8099178032063352e-05,
    "random_state": 23,
}

# Catboost
cb_best_params = {
    "bagging_temperature": 0.5879957646583531,
    "border_count": 64,
    "depth": 4,
    "iterations": 550,
    "l2_leaf_reg": 7.548114015892266,
    "learning_rate": 0.017750712329165527,
    "random_strength": 1.722307004315998,
    "random_state": 23,
    "verbose": 0,
    "allow_writing_files": False,
}
