import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import warnings

warnings.filterwarnings('ignore')

# --- 1. 데이터 로딩 및 전처리 ---
print("데이터를 로딩하고 전처리합니다...")

# 데이터 로딩
data_path_train = '../data/train.csv'
train_df = pd.read_csv(data_path_train)

# 불필요한 컬럼 제거
try:
    remove_cols_df = pd.read_csv('../doc/remove_cols.xls', header=0)
    remove_cols = remove_cols_df.squeeze()
    train_df.drop(columns=remove_cols, axis=1, inplace=True)
except FileNotFoundError:
    print("remove_cols.xls 파일을 찾을 수 없습니다. 모든 컬럼을 사용합니다.")
except Exception as e:
    print(f"컬럼 제거 중 오류 발생: {e}")


# 피처와 타겟 분리
X_features = train_df.drop(columns=['ID', 'TARGET'], axis=1)
y_labels = train_df['TARGET']

# 'var3' 컬럼의 이상치(-999999)를 최빈값(2)으로 대체
X_features['var3'] = X_features['var3'].replace(-999999, 2)

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_features,
    y_labels,
    test_size=0.3, # 더 많은 데이터로 검증하기 위해 test_size를 0.3으로 조정
    random_state=23,
    stratify=y_labels
)

print("데이터 준비 완료.")

# --- 2. Hyperparameter 탐색 공간 정의 ---

# XGBoost 탐색 공간
xgb_search_space = {
    'max_depth': hp.quniform('max_depth', 5, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.uniform('subsample', 0.7, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
    'gamma': hp.uniform('gamma', 0, 0.5)
}

# LightGBM 탐색 공간
lgbm_search_space = {
    'num_leaves': hp.quniform('num_leaves', 32, 128, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
    'subsample': hp.uniform('subsample', 0.7, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
}

# RandomForest 탐색 공간
rf_search_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 500, 10),
    'max_depth': hp.quniform('max_depth', 10, 30, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 8, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 12, 1)
}


# --- 3. 목적 함수 정의 ---

def xgb_objective(params):
    """XGBoost 목적 함수"""
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])

    xgb = XGBClassifier(
        **params,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=23
    )

    xgb.fit(
        X_train, y_train,
        early_stopping_rounds=30,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    roc_auc = roc_auc_score(y_val, xgb.predict_proba(X_val)[:, 1])
    
    return {'loss': -roc_auc, 'status': STATUS_OK}


def lgbm_objective(params):
    """LightGBM 목적 함수"""
    params['num_leaves'] = int(params['num_leaves'])
    params['n_estimators'] = int(params['n_estimators'])
    
    lgbm = LGBMClassifier(
        **params,
        random_state=23
    )
    
    lgbm.fit(
        X_train, y_train,
        early_stopping_rounds=30,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    roc_auc = roc_auc_score(y_val, lgbm.predict_proba(X_val)[:, 1])

    return {'loss': -roc_auc, 'status': STATUS_OK}

def rf_objective(params):
    """RandomForest 목적 함수"""
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['min_samples_split'] = int(params['min_samples_split'])

    rf = RandomForestClassifier(
        **params,
        random_state=23,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    roc_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])

    return {'loss': -roc_auc, 'status': STATUS_OK}


# --- 4. Hyperparameter 튜닝 실행 ---
import time

MAX_EVALS = 50 

# XGBoost 튜닝
print("\n--- XGBoost Hyperparameter Tuning ---")
start_time = time.time()
xgb_trials = Trials()
best_xgb = fmin(
    fn=xgb_objective,
    space=xgb_search_space,
    algo=tpe.suggest,
    max_evals=MAX_EVALS,
    trials=xgb_trials,
    rstate=pd.np.random.RandomState(23)
)
end_time = time.time()
print(f"튜닝 시간: {end_time - start_time:.2f}초")
print("최적 하이퍼파라미터:", best_xgb)
print("최고 AUC:", -xgb_trials.best_trial['result']['loss'])


# LightGBM 튜닝
print("\n--- LightGBM Hyperparameter Tuning ---")
start_time = time.time()
lgbm_trials = Trials()
best_lgbm = fmin(
    fn=lgbm_objective,
    space=lgbm_search_space,
    algo=tpe.suggest,
    max_evals=MAX_EVALS,
    trials=lgbm_trials,
    rstate=pd.np.random.RandomState(23)
)
end_time = time.time()
print(f"튜닝 시간: {end_time - start_time:.2f}초")
print("최적 하이퍼파라미터:", best_lgbm)
print("최고 AUC:", -lgbm_trials.best_trial['result']['loss'])


# RandomForest 튜닝
print("\n--- RandomForest Hyperparameter Tuning ---")
start_time = time.time()
rf_trials = Trials()
best_rf = fmin(
    fn=rf_objective,
    space=rf_search_space,
    algo=tpe.suggest,
    max_evals=MAX_EVALS,
    trials=rf_trials,
    rstate=pd.np.random.RandomState(23)
)
end_time = time.time()
print(f"튜닝 시간: {end_time - start_time:.2f}초")
print("최적 하이퍼파라미터:", best_rf)
print("최고 AUC:", -rf_trials.best_trial['result']['loss'])

print("\nHyperparameter 튜닝이 완료되었습니다.")
