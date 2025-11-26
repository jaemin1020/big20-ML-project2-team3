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

from functools import partial
from typing import Dict, Any, Callable
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK
from hyperopt import space_eval

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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
    model,
    model_name,
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    hyperopt_params=None,
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

    if hyperopt_params:
        get_clf_eval(
            y_test,
            pred,
            pred_proba,
            model_name=model_name,
            exec_time=exec_time,
            HO_params=hyperopt_params,
        )
    else:
        get_clf_eval(
            y_test, pred, pred_proba, model_name=model_name, exec_time=exec_time
        )


# -- eof ----------------------------------


def get_preprocssed_df(df, columns=None):
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


class HyperOptTuner:
    """HyperOpt을 사용한 하이퍼파라미터 튜닝 클래스
    사용법
    # 튜너 초기화
    tuner = HyperOptTuner(max_evals=100, metric='roc_auc', random_state=42)

    # RandomForest 예시
    rf_search_space = {
        'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)
    }

    rf = RandomForestClassifier()
    best_params, best_model, trials, exec_time = tuner.tune(
        rf, X_train, y_train, X_val, y_val, rf_search_space
    )
    get_model_train_eval(best_model, "rf_HyperOpt", X_train, X_val, y_train, y_val, best_params)
    """

    # 모델별 정수형 파라미터 정의
    INT_PARAMS = {
        "RandomForestClassifier": [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
        ],
        "XGBClassifier": ["max_depth", "n_estimators", "min_child_weight"],
        "LGBMClassifier": [
            "max_depth",
            "n_estimators",
            "min_child_weight",
            "num_leaves",
        ],
        "CatBoostClassifier": ["iterations", "depth", "min_data_in_leaf", "max_bin"],
    }

    # 모델별 고정 파라미터
    FIXED_PARAMS = {
        "RandomForestClassifier": {"random_state": 42, "n_jobs": -1},
        "XGBClassifier": {"random_state": 42},
        "LGBMClassifier": {"random_state": 42, "n_jobs": -1},
        "CatBoostClassifier": {
            "random_state": 42,
            "verbose": 0,
            "allow_writing_files": False,
        },
    }

    def __init__(
        self,
        max_evals: int = 100,
        metric: str = "recall",
        random_state: int = 42,
    ):
        """
        Args:
            max_evals: 최대 평가 횟수
            metric: 평가 지표 ('roc_auc', 'accuracy', 'f1' 등)
            random_state: 랜덤 시드
        """
        self.max_evals = max_evals
        self.metric = metric
        self.random_state = random_state

    @staticmethod
    def _convert_int_params(
        params: Dict[str, Any], int_param_names: list
    ) -> Dict[str, Any]:
        """정수형 파라미터 변환"""
        converted = params.copy()
        for param in int_param_names:
            if param in converted:
                converted[param] = int(converted[param])
        return converted

    def _get_metric_score(self, y_true, y_pred_proba):
        scores = {}
        y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
        """평가 지표 계산"""
        scores["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
        scores["f1"] = f1_score(y_true, y_pred)
        scores["precision"] = precision_score(y_true, y_pred)
        scores["recall"] = recall_score(y_true, y_pred)
        scores["accuracy"] = accuracy_score(y_true, y_pred)
        return scores

    def _objective(
        self, params: Dict[str, Any], model_class: type, X_train, y_train, X_val, y_val
    ) -> Dict[str, Any]:
        """통합 목적 함수"""
        try:
            model_name = model_class.__name__

            # 정수형 파라미터 변환
            if model_name in self.INT_PARAMS:
                params = self._convert_int_params(params, self.INT_PARAMS[model_name])

            # 고정 파라미터 추가
            if model_name in self.FIXED_PARAMS:
                params.update(self.FIXED_PARAMS[model_name])

            # 모델 학습
            model = model_class(**params)
            model.fit(X_train, y_train)

            # 평가
            y_pred_proba = model.predict_proba(X_val)
            scores = self._get_metric_score(y_val, y_pred_proba)

            # hyperopt는 loss를 최소화하는 방향으로 최적화합니다.
            loss = -scores[self.metric]

            # fmin은 loss, status 외의 다른 값들도 trials 객체에 저장합니다.
            return {"loss": loss, "status": STATUS_OK, "scores": scores}

        except Exception as e:
            print(f"Error in objective function: {str(e)}")
            return {"loss": float("inf"), "status": STATUS_OK}

    def tune(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        search_space: Dict[str, Any],
        verbose: bool = True,
    ) -> tuple:
        """
        하이퍼파라미터 튜닝 실행

        Args:
            model: 튜닝할 모델 인스턴스
            X_train, y_train: 학습 데이터
            X_val, y_val: 검증 데이터
            search_space: HyperOpt 탐색 공간
            verbose: 진행 상황 출력 여부

        Returns:
            (best_params, best_model, trials, exec_time)
        """
        model_class = type(model)
        model_name = model_class.__name__

        if verbose:
            print(f"\n{'='*50}")
            print(f"{model_name} 튜닝 시작")
            print(f"{'='*50}")

        # 튜닝 실행
        start_time = time.time()
        trials = Trials()

        objective_fn = partial(
            self._objective,
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        best_params = fmin(
            fn=objective_fn,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(seed=self.random_state),
            verbose=verbose,
        )

        exec_time = time.time() - start_time

        # 최적 파라미터 변환
        best_params = space_eval(search_space, best_params)

        # 정수형 파라미터 변환
        if model_name in self.INT_PARAMS:
            best_params = self._convert_int_params(
                best_params, self.INT_PARAMS[model_name]
            )

        # 고정 파라미터 추가
        if model_name in self.FIXED_PARAMS:
            best_params.update(self.FIXED_PARAMS[model_name])

        # 최적 모델 생성
        best_model = model_class(**best_params)

        if verbose:
            print(f"\n튜닝 시간: {exec_time:.2f}초")
            print(f"최적 {self.metric}: {-trials.best_trial['result']['loss']:.4f}")

            best_scores = trials.best_trial["result"]["scores"]
            print("\n최적 모델의 전체 평가 점수:")
            for metric_name, score_value in best_scores.items():
                print(f"- {metric_name}: {score_value:.4f}")

            print("\n최적 하이퍼파라미터:")
            for key, value in best_params.items():
                print(f"-{key}: {value}")

        return best_params, best_model, trials, exec_time


# -- eof -----
