import os
import sys

# 현재 작업 디렉토리 기준으로 상위 1단계 폴더를 루트로 설정
# current_dir  = os.getcwd()
# project_root = os.path.abspath(os.path.join(current_dir, '..'))
project_root = "c:/big20/git/big20-ML-project2-team3/SantanderCS"

# sys.path에 추가 (모듈 import용)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# print("프로젝트 루트로 설정된 경로:", project_root)
# print(sys.path)

import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK
from hyperopt import space_eval

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# import importlib

# from . import model_utils
# importlib.reload(model_utils)


# # 수정된 함수 불러오기
from utils.model_utils import save_model


def get_clf_eval(
    y_test,
    pred,
    pred_proba,
    model_name="model",
    folder="results",
    exec_time=None,
    HO_params=None,
):
    """
    분류 모델의 평가 지표를 계산하고 결과를 파일로 저장

    Parameters:
    -----------
    y_test : array-like
        실제 타겟 값
    pred : array-like
        예측 값
    pred_proba : array-like
        예측 확률 (Positive class)
    model_name : str
        모델 이름
    folder : str
        결과 저장 폴더명
    exec_time : float, optional
        실행 시간 (초)
    HO : dictionary
        하이퍼파라미터 값

    Returns:
    --------
    None
    """
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)

    result_text = (
        f"AUC: {roc_auc:.4f}, 정확도: {accuracy:.4f}, "
        f"정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}\n"
        f"오차행렬:\n{confusion}"
    )
    if exec_time is not None:
        result_text += f"\n실행 시간: {exec_time}"
    if HO_params is not None:
        result_text += f"\n하이퍼파라미터: {HO_params}"

    # 현재 작업 디렉토리 기준으로 상위 1단계 폴더를 루트로 설정
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    # 상위 폴더의 results 디렉토리 지정
    folder = os.path.abspath(os.path.join(os.getcwd(), "..", folder))
    print(f"folder = {folder}")
    os.makedirs(folder, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    filename = f"{model_name}_{today}.txt"
    save_path = os.path.join(folder, filename)

    print(result_text)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(result_text)


# --- eof ------------------


def get_model_train_eval(
    model, model_name, X_train=None, X_test=None, y_train=None, y_test=None
):
    """
    model별 HyperOpt수치, 학습, 예측값, 예측확율 구하기
    get_model_train_eval(lr_clf, 'model_name', X_train, X_test, y_train, y_test)

    """
    start_time = time.time()  # 시작 시간 기록
    model.fit(X_train, y_train)
    save_model(model, model_name)
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]  # Positive인 확률만 가져오기
    end_time = time.time()  # 종료 시간기록
    exec_time = end_time - start_time

    get_clf_eval(y_test, pred, pred_proba, model_name=model_name, exec_time=exec_time)


# -- eof ----------------------------------


def get_model_HO_train_eval(
    model,
    model_name,
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    hyperopt_params=None,
):
    """
     학습, 예측값, 예측확율, HyperOpt 파라미터 구하기
    get_model_train_eval(lr_clf, 'model_name', X_train, X_test, y_train, y_test, hyperopt_params)

    """
    start_time = time.time()  # 시작 시간 기록
    model.fit(X_train, y_train)
    save_model(model, model_name)
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]  # Positive인 확률만 가져오기
    end_time = time.time()  # 종료 시간기록
    exec_time = end_time - start_time

    get_clf_eval(
        y_test,
        pred,
        pred_proba,
        model_name=model_name,
        exec_time=exec_time,
        HO_params=hyperopt_params,
    )


# -- eof ----------------------------------


def get_preprocssed_df(df, columns):
    df_copy = df.copy()
    # 로그 변환
    amount_n = np.log1p(df_copy["Amount"])
    df_copy.insert(0, "Amount_Scaled", amount_n)
    df_copy.drop(["Time", "Amount"], axis=1, inplace=True)
    # 이상치 제거
    outlier_index = get_outlier(df=df_copy, columns=columns)
    df_copy.drop(outlier_index, axis=0, inplace=True)
    return df_copy


# 학습/테스트 분리
def get_train_test_dataset(
    df=None,
):  # df : 원본 받아서, df_copy로 사용 dataset대신 df가 더 어울리는데..
    df_copy = get_preprocssed_df(df)  # Time Feature drop

    # data and label seperate
    X_features = df_copy.iloc[:, :-1]
    y_target = df_copy.iloc[:, -1]

    # learn/test data sperate
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y_target,
        test_size=0.3,
        random_state=0,
        stratify=y_target,  # 불균형 데이터일 때 반드시 처리 필요!!! 중요해~
    )
    return X_train, X_test, y_train, y_test


def get_outlier(
    df, columns=None, weight=1.5
):  # weight=1.5 고정은 아니다! 존 튜키(John Tukey)

    # 25%, 75% 위치에 있는 값 구한다
    if columns:
        Q1 = df[df[columns]].loc["25%"]
        Q3 = df[df[columns]].loc["75%"]
    else:
        Q1 = df.loc["25%"]
        Q3 = df.loc["75%"]
    iqr = Q3 - Q1
    iqr_weight = iqr * weight
    lower_bound = Q1 - iqr_weight
    upper_bound = Q3 + iqr_weight

    # 이상치 마스킹 (컬럼별로)
    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.select_dtypes(include="number").columns:
        if col in lower_bound.index:
            lb = lower_bound[col]
            ub = upper_bound[col]
            outlier_mask[col] = (df[col] < lb) | (df[col] > ub)

    # 5. 이상치 기준 + 마스크를 outlier_bounds에 통합
    outlier_bounds = pd.DataFrame(
        {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": iqr,
            "LowerBound": lower_bound,
            "UpperBound": upper_bound,
            "OutlierCount": outlier_mask.sum(),
        }
    )

    return outlier_bounds


# -- eof -----
MAX_EVALS = 100


def lr_objective(params, X_train, y_train, X_val, y_val):
    """LogisticRegression 목적함수"""
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    roc_auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
    return {"loss": -roc_auc, "status": STATUS_OK}


def rf_objective(params, X_train, y_train, X_val, y_val):
    """RandomForest 목적 함수"""
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])
    params["min_samples_split"] = int(params["min_samples_split"])

    rf = RandomForestClassifier(**params, random_state=23, n_jobs=-1)

    rf.fit(X_train, y_train)
    roc_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])

    return {"loss": -roc_auc, "status": STATUS_OK}


def xgb_objective(params, X_train, y_train, X_val, y_val):
    """XGBoost 목적 함수"""
    params["max_depth"] = int(params["max_depth"])
    params["n_estimators"] = int(params["n_estimators"])
    params["min_child_weight"] = int(params["min_child_weight"])

    xgb = XGBClassifier(**params)

    xgb.fit(X_train, y_train)
    roc_auc = roc_auc_score(y_val, xgb.predict_proba(X_val)[:, 1])

    return {"loss": -roc_auc, "status": STATUS_OK}


def lgbm_objective(params, X_train, y_train, X_val, y_val):
    """LightGBM 목적 함수"""
    params["max_depth"] = int(params["max_depth"])
    params["n_estimators"] = int(params["n_estimators"])
    params["min_child_weight"] = int(params["min_child_weight"])
    params["num_leaves"] = int(params["num_leaves"])

    lgbm = LGBMClassifier(**params)

    lgbm.fit(X_train, y_train)
    roc_auc = roc_auc_score(y_val, lgbm.predict_proba(X_val)[:, 1])

    return {"loss": -roc_auc, "status": STATUS_OK}


def HyperOpt_Tune(model, X_train, y_train, X_val, y_val, search_space):
    if model.__class__.__name__ == "LogisticRegression":
        start_time = time.time()  # 시작 시간 기록
        lr_trials = Trials()
        best_lr = fmin(
            fn=lambda params: lr_objective(params, X_train, y_train, X_val, y_val),
            space=search_space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=lr_trials,
            rstate=np.random.default_rng(seed=42),
        )
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"튜닝 시간{exec_time:.2f}초")
        print("최적 하이퍼파라미터 : ", best_lr)

        best_lr = space_eval(search_space, best_lr)

        lr_best = LogisticRegression(**best_lr)

        get_model_train_eval(lr_best, "lr_HyperOpt_ejm", X_train, X_val, y_train, y_val)

    elif model.__class__.__name__ == "RandomForestClassifier":
        start_time = time.time()
        rf_trials = Trials()
        best_rf = fmin(
            fn=lambda params: rf_objective(params, X_train, y_train, X_val, y_val),
            space=search_space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=rf_trials,
            rstate=np.random.default_rng(seed=42),
        )
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"튜닝 시간{exec_time:.2f}초")
        print("최적 하이퍼파라미터 : ", best_rf)

        best_rf = space_eval(search_space, best_rf)

        best_rf["n_estimators"] = int(best_rf["n_estimators"])
        best_rf["max_depth"] = int(best_rf["max_depth"])
        best_rf["min_samples_leaf"] = int(best_rf["min_samples_leaf"])
        best_rf["min_samples_split"] = int(best_rf["min_samples_split"])

        rf_best = RandomForestClassifier(**best_rf, random_state=23, n_jobs=-1)

        get_model_train_eval(rf_best, "rf_HyperOpt_ejm", X_train, X_val, y_train, y_val)
    elif model.__class__.__name__ == "XGBClassifier":
        start_time = time.time()
        xgb_trials = Trials()
        best_xgb = fmin(
            fn=lambda params: xgb_objective(params, X_train, y_train, X_val, y_val),
            space=search_space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=xgb_trials,
            rstate=np.random.default_rng(seed=42),
        )
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"튜닝 시간{exec_time:.2f}초")
        print("최적 하이퍼파라미터 : ", best_xgb)

        best_xgb = space_eval(search_space, best_xgb)

        best_xgb["max_depth"] = int(best_xgb["max_depth"])
        best_xgb["n_estimators"] = int(best_xgb["n_estimators"])
        best_xgb["min_child_weight"] = int(best_xgb["min_child_weight"])

        xgb_best = XGBClassifier(**best_xgb)

        get_model_train_eval(
            xgb_best, "xgb_HyperOpt_ejm", X_train, X_val, y_train, y_val
        )
    elif model.__class__.__name__ == "LGBMClassifier":
        start_time = time.time()
        lgbm_trials = Trials()
        best_lgbm = fmin(
            fn=lambda params: lgbm_objective(params, X_train, y_train, X_val, y_val),
            space=search_space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=lgbm_trials,
            rstate=np.random.default_rng(seed=42),
        )
        end_time = time.time()  # 종료 시간기록
        exec_time = end_time - start_time
        print(f"튜닝 시간{exec_time:.2f}초")
        print("최적 하이퍼파라미터 : ", best_lgbm)

        best_lgbm = space_eval(search_space, best_lgbm)

        best_lgbm["max_depth"] = int(best_lgbm["max_depth"])
        best_lgbm["n_estimators"] = int(best_lgbm["n_estimators"])
        best_lgbm["min_child_weight"] = int(best_lgbm["min_child_weight"])
        best_lgbm["num_leaves"] = int(best_lgbm["num_leaves"])

        lgbm_best = LGBMClassifier(**best_lgbm)

        get_model_train_eval(
            lgbm_best, "lgbm_HyperOpt_ejm", X_train, X_val, y_train, y_val
        )


# -- eof -----
