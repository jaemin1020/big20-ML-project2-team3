import os
import sys

# 현재 작업 디렉토리 기준으로 상위 1단계 폴더를 루트로 설정
# current_dir  = os.getcwd()
# project_root = os.path.abspath(os.path.join(current_dir, '..'))
project_root = "c:/big20/git/big20-ML-project2-team3/CreditCardFraud"

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
from sklearn.model_selection import cross_val_score  # hyperopt_tune 에서 사용
from sklearn.preprocessing import StandardScaler

from functools import partial
from typing import Dict, Any, Callable, Optional, Union
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK
from hyperopt import space_eval
from hyperopt import early_stop

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import json
import pickle

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
        f"{{'AUC': {roc_auc:.4f}, '정확도': {accuracy:.4f}, "
        f"'정밀도': {precision:.4f}, '재현율': {recall:.4f}, 'F1': {f1:.4f} }}\n"
        f"{{'오차행렬':\n{confusion} }}"
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

    return {
        "AUC": round(roc_auc_score(y_test, pred_proba), 4),
        "정밀도": round(precision_score(y_test, pred), 4),
        "재현율": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
    }


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
        test_size=0.2,
        random_state=23,
        stratify=y_target,  # 불균형 데이터일 때 반드시 처리 필요!!! 중요해~
    )
    return X_train, X_test, y_train, y_test


def get_outlier(df, columns=None, weight=1.5):
    """
    IQR(Interquartile Range) 방법을 사용하여 이상치 탐지

    존 튜키(John Tukey)의 방법론을 기반으로 Q1, Q3를 이용해
    이상치 경계를 계산하고, 각 컬럼별 이상치 개수를 반환합니다.

    Parameters:
    -----------
    df : DataFrame
        분석할 데이터프레임 (df.describe() 결과 또는 원본 데이터)
    columns : str, list, or None
        분석할 컬럼 지정
        - None: 모든 수치형 컬럼
        - str: 단일 컬럼
        - list: 여러 컬럼
    weight : float, default=1.5
        IQR 가중치 (일반적으로 1.5 사용)
        - 1.5: 일반적 이상치 (mild outliers)
        - 3.0: 극단 이상치 (extreme outliers)
        - 값이 작을수록 더 많은 데이터를 이상치로 판단

    Returns:
    --------
    DataFrame
        각 컬럼별 이상치 정보를 담은 데이터프레임
        - Q1: 1사분위수 (25%)
        - Q3: 3사분위수 (75%)
        - IQR: 사분위 범위 (Q3 - Q1)
        - LowerBound: 하한선 (Q1 - weight*IQR)
        - UpperBound: 상한선 (Q3 + weight*IQR)
        - OutlierCount: 이상치 개수

    공식:
    -----
    - IQR = Q3 - Q1
    - LowerBound = Q1 - (weight × IQR)
    - UpperBound = Q3 + (weight × IQR)
    - Outlier: value < LowerBound or value > UpperBound

    사용 예시:
    ----------
    >>> # df.describe() 결과로 분석
    >>> desc = df.describe()
    >>> outliers = get_outlier(desc)

    >>> # 원본 데이터로 분석
    >>> outliers = get_outlier(df)

    >>> # 특정 컬럼만 분석
    >>> outliers = get_outlier(desc, columns='Amount')
    >>> outliers = get_outlier(desc, columns=['Amount', 'Time'])

    >>> # 더 엄격한 기준 적용
    >>> outliers = get_outlier(desc, weight=3.0)
    """
    try:
        # 1. DataFrame validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df는 DataFrame이어야 합니다. 현재: {type(df).__name__}")

        if df.empty:
            raise ValueError("빈 데이터프레임입니다.")

        # 2. weight validation
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(f"weight는 양수여야 합니다. 현재: {weight}")

        # 3. df가 describe() 결과인지 원본 데이터인지 확인
        is_describe_result = "25%" in df.index and "75%" in df.index

        if is_describe_result:
            # describe() 결과를 직접 받은 경우
            stats_df = df
        else:
            # 원본 데이터인 경우 describe() 수행
            stats_df = df.describe()

        # 4. Q1, Q3 추출
        if columns is not None:
            # 컬럼 지정된 경우
            if isinstance(columns, str):
                columns = [columns]

            # 컬럼 존재 확인
            missing_cols = [col for col in columns if col not in stats_df.columns]
            if missing_cols:
                raise KeyError(f"존재하지 않는 컬럼: {missing_cols}")

            Q1 = stats_df.loc["25%", columns]
            Q3 = stats_df.loc["75%", columns]
        else:
            # 모든 수치형 컬럼
            Q1 = stats_df.loc["25%"]
            Q3 = stats_df.loc["75%"]

        # 5. IQR 및 경계 계산
        iqr = Q3 - Q1
        iqr_weight = iqr * weight
        lower_bound = Q1 - iqr_weight
        upper_bound = Q3 + iqr_weight

        # 6. 이상치 개수 계산 (원본 데이터 필요)
        if is_describe_result:
            # describe() 결과만 있는 경우 - 개수 계산 불가
            outlier_count = pd.Series(np.nan, index=Q1.index)
            print("⚠️ 경고: describe() 결과로는 이상치 개수를 계산할 수 없습니다.")
            print("   원본 데이터를 입력하면 정확한 개수를 계산할 수 있습니다.")
        else:
            # 원본 데이터가 있는 경우 - 실제 이상치 개수 계산
            outlier_count = pd.Series(0, index=Q1.index)

            for col in Q1.index:
                if col in df.columns:
                    mask = (df[col] < lower_bound[col]) | (df[col] > upper_bound[col])
                    outlier_count[col] = mask.sum()

        # 7. 결과 데이터프레임 생성
        outlier_bounds = pd.DataFrame(
            {
                "min": stats_df.loc["min"],
                "LowerBound": lower_bound,
                "Q1": Q1,
                "Q3": Q3,
                "IQR": iqr,
                "max": stats_df.loc["max"],
                "UpperBound": upper_bound,
                "OutlierCount": outlier_count,
            }
        )

        # 8. 결과 요약 출력
        print(f"\n{'='*70}")
        print(f"이상치 탐지 결과 (IQR weight: {weight})")
        print(f"{'='*70}")

        if not outlier_count.isna().all():
            total_outliers = int(outlier_count.sum())
            print(f"총 이상치 개수: {total_outliers:,}개")

            if total_outliers > 0:
                print(f"\n이상치가 많은 상위 5개 컬럼:")
                top_outliers = outlier_count.nlargest(5)
                for col, count in top_outliers.items():
                    if count > 0:
                        percentage = (
                            (count / len(df)) * 100 if not is_describe_result else 0
                        )
                        print(f"  - {col}: {int(count):,}개 ({percentage:.2f}%)")

        print(f"{'='*70}\n")

        return outlier_bounds

    except KeyError as e:
        print(f"❌ 컬럼 접근 오류: {e}")
        print(f"   사용 가능한 컬럼: {list(df.columns)}")
        raise
    except Exception as e:
        print(f"❌ get_outlier 함수 실행 중 오류: {e}")
        raise


# eof ----------------------------------------------------------- #


class HyperOptTuner:
    """HyperOpt을 사용한 하이퍼파라미터 튜닝 클래스
    사용법
    # 튜너 초기화
    tuner = HyperOptTuner(max_evals=100, random_state=23, early_stopping_rounds = 20,class_weight='balanced')

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
    get_model_train_eval(best_model, "rf_HyperOpt", X_train, X_test, y_train, y_test, best_params)
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
            "min_data_in_leaf",
            "bagging_freq",
        ],
        "CatBoostClassifier": [
            "iterations",
            "depth",
            "min_data_in_leaf",
            "max_bin",
            "border_count",
        ],
        "LogisticRegression": ["max_iter"],
        "MLPClassifier": ["max_iter"],
        "GradientBoostingClassifier": [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
        ],
        "DecisionTreeClassifier": [
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
        ],
    }

    # 모델별 고정 파라미터
    FIXED_PARAMS = {
        "RandomForestClassifier": {"random_state": 23, "n_jobs": -1},
        "XGBClassifier": {"random_state": 23},
        "LGBMClassifier": {"random_state": 23, "n_jobs": -1},
        "CatBoostClassifier": {
            "random_state": 23,
            "verbose": 0,
            "allow_writing_files": False,
        },
        "LogisticRegression": {"random_state": 23, "n_jobs": -1},
        "MLPClassifier": {"random_state": 23},
        "GradientBoostingClassifier": {"random_state": 23},
        "DecisionTreeClassifier": {"random_state": 23},
    }

    def __init__(
        self,
        max_evals: int = 100,
        metric: str = "roc_auc",
        random_state: int = 23,
        early_stopping_rounds: Optional[int] = None,
        class_weight: Optional[Union[str, dict]] = None,
    ):
        """
        Args:
            max_evals: 최대 평가 횟수
            metric: 평가 지표 ('roc_auc', 'accuracy', 'f1' 등)
            random_state: 랜덤 시드
            early_stopping_rounds: 조기 종료 patience. None이면 사용 안함.
        """
        self.max_evals = max_evals
        self.metric = metric
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.class_weight = class_weight

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
        scores["f1"] = f1_score(y_true, y_pred, zero_division=0)
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

            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_val)
            elif hasattr(model, "decision_function"):
                decision_scores = model.decision_function(X_val)
                # decision_function은 1차원일 수 있으므로 변환 필요
                if decision_scores.ndim == 1:
                    decision_scores = (decision_scores - decision_scores.min()) / (
                        decision_scores.max() - decision_scores.min() + 1e-8
                    )
                    y_pred_proba = np.vstack([1 - decision_scores, decision_scores]).T
                else:
                    y_pred_proba = decision_scores
            else:
                y_pred = model.predict(X_val)
                y_pred_proba = np.vstack([1 - y_pred, y_pred]).T

            scores = self._get_metric_score(y_val, y_pred_proba)

            # hyperopt는 loss를 최소화하는 방향으로 최적화합니다.
            loss = -scores.get(self.metric, 0.0)

            return {"loss": loss, "status": STATUS_OK, "scores": scores}

        except Exception as e:
            print(f"Error in objective function: {str(e)}")
            # 예외 발생 시에도 기본 scores 반환
            return {
                "loss": float("inf"),
                "status": STATUS_OK,
                "scores": {
                    "accuracy": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "roc_auc": 0.0,
                    "precision": 0.0,
                },
            }

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

        early_stop_fn = None
        if self.early_stopping_rounds is not None:

            early_stop_fn = early_stop.no_progress_loss(self.early_stopping_rounds)
            if verbose:
                print(
                    f"\n조기 종료 활성화: {self.early_stopping_rounds} 라운드 동안 개선 없을 시 중단."
                )

        best_params = fmin(
            fn=objective_fn,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(seed=self.random_state),
            verbose=verbose,
            early_stop_fn=early_stop_fn,
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

        if self.class_weight and model_name in [
            "RandomForestClassifier",
            "LogisticRegression",
            "XGBClassifier",
            "LGBMClassifier",
            "DecisionTreeClassifier",
        ]:
            best_params["class_weight"] = self.class_weight

        # 최적 모델 생성
        best_model = model_class(**best_params)

        if verbose:
            print(f"\n튜닝 시간: {exec_time:.2f}초")
            print(f"최적 {self.metric}: {-trials.best_trial['result']['loss']:.4f}")

            result = trials.best_trial["result"]
            if "scores" in result:
                best_scores = result["scores"]
                print("\n최적 모델의 전체 평가 점수:")
                for metric_name, score_value in best_scores.items():
                    print(f"- {metric_name}: {score_value:.4f}")
            else:
                print("\n최적 trial에 scores가 없습니다. (예외 발생 가능성)")

        return best_params, best_model, trials, exec_time


# -- eof -----
def preprocess_and_train_logreg(df, model_name="LogisticRegression"):
    """
    df: 원본 데이터셋
    model_name: 모델 이름 (기본값 LogisticRegression)
    """
    # -----------------------------
    # 1. Time 컬럼 제거
    # -----------------------------
    df_processed = df.drop(["Time"], axis=1)

    # -----------------------------
    # 2. Amount 로그 변환 + StandardScaler 적용
    # -----------------------------
    df_processed["Amount_log"] = np.log1p(df_processed["Amount"])  # log(Amount+1)
    scaler = StandardScaler()
    df_processed["Amount_scaled"] = scaler.fit_transform(
        df_processed["Amount_log"].values.reshape(-1, 1)
    )

    # 원본 Amount, Amount_log 제거
    df_processed = df_processed.drop(["Amount", "Amount_log"], axis=1)

    # -----------------------------
    # 3. 특성과 타겟 분리, 학습/테스트 분할
    # -----------------------------

    X_train, X_test, y_train, y_test = get_train_test_dataset(df_processed)

    # -----------------------------
    # 5. Logistic Regression 학습
    # -----------------------------
    log_reg = LogisticRegression(max_iter=1000, random_state=23)

    # -----------------------------
    # 6. 평가 (사용자 정의 함수 호출)
    # -----------------------------
    get_model_train_eval(log_reg, model_name, X_train, X_test, y_train, y_test)


# -- eof -----
