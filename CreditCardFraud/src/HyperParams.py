# 모델별 최종 HyperParamers
import numpy as np

# RF
rf_best_params_lsj = {
    'max_depth': 10,             # 트리의 최대 깊이 (규제와 성능 사이의 균형점)
    'max_features': 'log2',      # 각 분기에서 고려할 최대 특성 수 (특성 수의 log2만큼 사용)
    'min_samples_leaf': 5,       # 리프 노드가 되기 위한 최소 샘플 수
    'min_samples_split': 15,     # 노드를 분할하기 위한 최소 샘플 수
    'n_estimators': 450,         # 생성할 트리 개수 (충분히 많음)
    'random_state': 23,          # 재현성을 위한 랜덤 시드
    'n_jobs': -1                 # 병렬 처리 시 모든 CPU 코어 사용    
    # 참고: class_weight 파라미터가 포함되어 있지 않음. 
    #       신용카드 사기 검출 시 이 파라미터 추가 필요.
}
'''
캐글 신용카드 사기 검출 데이터셋 적용 분석
이 파라미터 세트는 이전 rf_best_params (max_depth 3)보다 훨씬 강력한 모델입니다.
1. 모델 복잡도: 중~고 수준
max_depth: 10
깊이 3은 너무 얕았지만, 깊이 10은 적절한 수준의 복잡성을 가집니다. 모델이 데이터의 복잡한 패턴을 학습할 수 있는 능력이 향상되었습니다.
n_estimators: 450
충분한 수의 트리를 사용하여 모델의 안정성과 성능을 높였습니다.
2. 규제 및 샘플링: 적절한 균형
max_features: 'log2'
특성 수의 로그2만큼의 특성만 고려하여 무작위성을 높이고 과적합을 방지합니다.
min_samples_leaf: 5 & min_samples_split: 15
노드 생성 조건을 까다롭게 하여 트리가 너무 깊고 복잡해지는 것을 막습니다.
3. 주요 고려사항: 클래스 불균형 처리 부재
제공된 rf_best_params2 딕셔너리에는 class_weight 파라미터가 포함되어 있지 않습니다.
캐글 신용카드 사기 검출과 같이 극심한 클래스 불균형이 있는 데이터셋에서는 이대로 학습할 경우 사기 거래 탐지율(Recall)이 매우 낮게 나올 것입니다.
결론 및 제안
이 파라미터 세트는 매우 유망한 성능을 보일 것으로 예상됩니다. 하지만 신용카드 사기 검출 데이터에 사용하기 위해서는 반드시 class_weight 파라미터를 추가해야 합니다.

'''

# --- Random Forest 최적 하이퍼파라미터 정의 ---
rf_best_params = {
    'bootstrap': True,          # 부트스트래핑 사용 여부 (True 권장)
    'class_weight': 'balanced', # 클래스 불균형 자동 처리 ('balanced' 사용)
    'criterion': 'entropy',     # 불순도 측정 기준: 엔트로피 사용 ('gini' 대신)
    'max_depth': 10, #3,         # 트리의 최대 깊이 (매우 얕게 설정됨)
    'max_features': None,       # 각 분기에서 고려할 최대 특성 수 (모든 특성 사용)
    'min_samples_leaf': 9,      # 리프 노드가 되기 위한 최소 샘플 수
    'min_samples_split': 3,     # 노드를 분할하기 위한 최소 샘플 수
    'n_estimators': 200,        # 생성할 트리 개수
    'random_state': 23,         # 재현성을 위한 랜덤 시드
    'n_jobs': -1                # 병렬 처리 시 모든 CPU 코어 사용
}
'''
### 결론 및 제안
이 파라미터 세트는 과적합 방지에 최적화되어 있으며, 학습된 모델이 매우 단순하고 해석하기 쉬울 것으로 예상됩니다.
장점: 안정적이며, 새로운 데이터(테스트/검증 데이터)에서 예상치 못한 성능 저하가 적을 것입니다.
단점: 트리의 깊이가 3으로 너무 얕아서 복잡한 사기 패턴을 놓칠 수 있으며, 이로 인해 LightGBM 같은 더 강력한 부스팅 모델보다 전체적인 성능(AUROC, F1-Score)이 낮을 수 있습니다.
추천:
이 파라미터 세트는 베이스라인 모델로 사용하기에 매우 적합합니다. 만약 이 모델의 성능이 기대보다 낮다면, max_depth를 10 또는 20 정도로 높여서 모델의 복잡도를 높이는 방향으로 튜닝을 시도해 볼 수 있습니다.
'''
rf_best_param_santander = {
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
# 원본 데이터(org data) 전처리 후 학습 시 사용된 XGBoost 최적 파라미터
xgb_best_params = {
    # 모델 복잡성 제어 (트리 구조 관련)
    "max_depth": 3,             # 트리의 최대 깊이 (매우 얕게 설정됨)
    "min_child_weight": 6,      # 리프 노드의 최소 가중치 합 (과적합 방지)
    "gamma": 1.7089112007254605, # 트리 분할에 필요한 최소 손실 감소량 (값이 클수록 규제 강함)

    # 학습률 및 부스팅 반복 횟수
    "learning_rate": 0.038583724829444874, # 학습 속도 (비교적 낮음)
    "n_estimators": 150,        # 부스팅 단계 수 (트리 개수)

    # 샘플링 및 피처 사용 비율 제어
    "subsample": 0.6,           # 각 트리 학습 시 사용될 데이터 샘플링 비율 (60% 사용)
    "colsample_bytree": 1.0,    # 각 트리 학습 시 사용될 특성(컬럼) 샘플링 비율 (모두 사용)

    # 규제 (L1/L2)
    "reg_alpha": 0.5940048544595263, # L1 규제 (Lasso)
    "reg_lambda": 0.007186370173139192, # L2 규제 (Ridge)

    # 클래스 불균형 처리 (원본 데이터셋 비율에 맞춤)
    "scale_pos_weight": 82.0,   # 양성 클래스(사기)에 부여하는 가중치

    # 시스템/재현성 설정
    "random_state": 23,         # 재현성을 위한 랜덤 시드
    # "objective": "binary:logistic", # 일반적으로 이진 분류 시 기본 설정됨
}
''' 
xgb_best_params 분석:
강력한 클래스 가중치: scale_pos_weight가 82로 설정되었습니다. 이는 데이터셋의 실제 불균형 비율(약 578:1)보다는 낮지만, 모델이 사기 거래 탐지에 집중하도록 유도합니다.
얕은 트리: max_depth가 3으로 매우 얕게 설정되어 과적합을 방지하고 모델 안정성을 추구합니다.
약한 규제 및 피처 사용: colsample_bytree가 1.0으로 모든 특성을 사용하며, 규제 강도(alpha, lambda)는 중간 정도입니다.
'''

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

# GradientBoostingClassifier by CreditFraud
gb_best_params = {
    'learning_rate': 0.02083271293464231, 
    'max_depth': 3, 
    'n_estimators': 800, 
    'subsample': 0.7076395576782778, 
    'random_state': 23
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
# --- 1. SGDClassifier 최적 하이퍼파라미터 정의 ---
sgd_best_params = {
    # 모델의 목적 (분류 문제에서 손실 함수 선택)
    "loss": "log_loss",  # 로지스틱 회귀를 위한 손실 함수 ('hinge'는 SVM)

    # 규제 (과적합 방지) 설정
    "penalty": "elasticnet",  # L1 (Lasso)과 L2 (Ridge) 규제를 혼합 사용
    "alpha": 3.910947175889396e-05, # 규제의 강도 (값이 낮을수록 규제 약함)
    "l1_ratio": 0.6092668391045077, # ElasticNet에서 L1 규제의 혼합 비율 (0: L2만, 1: L1만)

    # 학습률 (Learning Rate) 설정
    "learning_rate": "adaptive",  # 적응형 학습률 방식 사용 (성능 저하 시 eta0 자동 감소)
    "eta0": 0.0024094779453352057, # 초기 학습률 값 (learning_rate이 'constant', 'adaptive', 'invscaling' 일 때 사용)

    # 반복 학습 및 수렴 조건
    "max_iter": 6000, # 전체 데이터셋 반복 횟수 (에포크)
    "tol": 1.8099178032063352e-05, # 수렴 판단 임계값 (이 값보다 손실이 덜 감소하면 학습 중단)

    # 클래스 불균형 처리 (★ 중요: 캐글 데이터셋에 맞게 수정 필요)
    # 현재 값은 None이며, 아래 주석 처리된 옵션 중 하나를 선택하여 사용 권장
    "class_weight": None,
    # "class_weight": "balanced", # 데이터 불균형 시 자동 가중치 부여 옵션
    # "class_weight": {0: 1, 1: 578}, # 캐글 데이터셋 비율에 맞춘 수동 가중치 부여 옵션

    # 시스템/재현성 설정
    "random_state": 23, # 결과 재현성을 위한 랜덤 시드 설정
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
