``` python
# ============================================================================
# MercariLeaderAnalyzer - Kaggle 1위 전략 구현 (수정 버전)
# ============================================================================

"""
주요 수정사항:
1. sparse matrix 길이 처리: len() → .shape[0]
2. early_stopping 콜백 수정
3. 에러 처리 개선
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

XGBOOST_AVAILABLE = True
CATBOOST_AVAILABLE = True
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

# 프로젝트 루트
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class MercariLeaderAnalyzer:
    """
    Mercari Price Prediction - Kaggle 1위 전략 구현

    핵심 전략:
    ---------
    1. **Stage 1-A: Ridge Regressor**
       - Input: TF-IDF 벡터화된 텍스트 (name + description)
       - Output: 텍스트 기반 가격 예측

    2. **Stage 1-B: LightGBM**
       - Input: 카테고리 인코딩 + 수치 피처
       - Output: 메타데이터 기반 가격 예측

    3. **Stage 2: Weighted Ensemble**
       - Ridge + LightGBM 예측값을 가중 평균
       - 최적 가중치는 검증 세트에서 탐색
    """

    def __init__(
        self,
        random_state: int = 23,
        models_dir: str = "../models",
        results_dir: str = "../results",
        images_dir: str = "../images",
    ):
        """초기화"""
        self.random_state = random_state
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.images_dir = images_dir

        # 원본 데이터
        self.train: Optional[pd.DataFrame] = None
        self.test: Optional[pd.DataFrame] = None

        # 전처리된 피처
        self.X_text_train = None
        self.X_text_valid = None
        self.X_text_test = None

        self.X_meta_train = None
        self.X_meta_valid = None
        self.X_meta_test = None

        self.y_train = None
        self.y_valid = None

        # 전체 데이터 (최종 학습용)
        self.X_text_full = None
        self.X_meta_full = None
        self.y_full = None

        # 모델
        self.ridge_model: Optional[Ridge] = None
        self.lgbm_model: Optional[LGBMRegressor] = None

        # 앙상블 가중치
        self.ensemble_weights = {"ridge": 0.5, "lgbm": 0.5}

        # 인코더/벡터라이저
        self.tfidf = None
        self.label_encoders = {}

        # 메트릭
        self.metrics = {}

        # 디렉토리 생성
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def load_data(
        self,
        train_path: str = "../data/train.tsv",
        test_path: str = "../data/test.tsv",
        sep: str = "\t",
    ):
        """데이터 로딩 및 기본 정제"""
        self.train = pd.read_csv(train_path, sep=sep)
        self.test = pd.read_csv(test_path, sep=sep)

        # 필터링
        self.train = self.train[self.train["price"] > 0].dropna(subset=["price"])

        # 결측 처리
        for df in [self.train, self.test]:
            df["brand_name"] = df["brand_name"].fillna("Unknown")
            df["category_name"] = df["category_name"].fillna("Unknown")
            df["item_description"] = df["item_description"].fillna("No description")

        print("✅ Data Loaded.")
        print(f"   train: {self.train.shape}, test: {self.test.shape}")

    def _split_category(self, df: pd.DataFrame):
        """카테고리 3분할"""
        cats = df["category_name"].str.split("/", n=2, expand=True)
        df["cat1"] = cats[0].fillna("NoCat1")
        df["cat2"] = cats[1].fillna("NoCat2") if cats.shape[1] > 1 else "NoCat2"
        df["cat3"] = cats[2].fillna("NoCat3") if cats.shape[1] > 2 else "NoCat3"

    def preprocess(
        self,
        use_cache: bool = True,
        save_cache: bool = True,
        tfidf_max_features: int = 50000,
        debug: bool = True,
    ):
        """
        2-Stage 전략에 맞는 전처리

        ✅ 수정: sparse matrix 길이 처리
        """
        if self.train is None or self.test is None:
            raise RuntimeError("load_data()를 먼저 호출하세요.")

        cache_dir = os.path.join(self.results_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "mercari_leader_preprocessed_tfidf.pkl")

        # 캐시 로드
        if use_cache and os.path.exists(cache_path):
            if debug:
                print(f"📦 캐시 로드: {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
                (
                    self.X_text_train,
                    self.X_text_valid,
                    self.X_text_full,
                    self.X_text_test,
                    self.X_meta_train,
                    self.X_meta_valid,
                    self.X_meta_full,
                    self.X_meta_test,
                    self.y_train,
                    self.y_valid,
                    self.y_full,
                    self.tfidf,
                    self.label_encoders,
                ) = cached
            if debug:
                print("✅ 캐시 로드 완료")
            return

        # 카테고리 분할
        self._split_category(self.train)
        self._split_category(self.test)

        # ===== 1. 텍스트 파이프라인 (Ridge용) =====
        if debug:
            print("\n📝 [Stage 1-A] 텍스트 전처리 (Ridge용)")

        self.train["text_all"] = (
            self.train["name"].astype(str)
            + " "
            + self.train["item_description"].astype(str)
        )
        self.test["text_all"] = (
            self.test["name"].astype(str)
            + " "
            + self.test["item_description"].astype(str)
        )

        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )

        X_text_full = self.tfidf.fit_transform(self.train["text_all"])
        X_text_test = self.tfidf.transform(self.test["text_all"])

        if debug:
            print(f"   TF-IDF shape: {X_text_full.shape}")

        # ===== 2. 메타데이터 파이프라인 (LightGBM용) =====
        if debug:
            print("\n🏷️  [Stage 1-B] 메타데이터 전처리 (LightGBM용)")

        # Label Encoding
        cat_cols = ["brand_name", "cat1", "cat2", "cat3"]

        for col in cat_cols:
            le = LabelEncoder()
            # train + test 합쳐서 fit
            all_values = pd.concat([self.train[col], self.test[col]]).unique()
            le.fit(all_values)

            self.train[f"{col}_encoded"] = le.transform(self.train[col])
            self.test[f"{col}_encoded"] = le.transform(self.test[col])

            self.label_encoders[col] = le

        # 메타 피처 구성
        meta_cols = [
            "item_condition_id",
            "shipping",
            "brand_name_encoded",
            "cat1_encoded",
            "cat2_encoded",
            "cat3_encoded",
        ]

        X_meta_full = self.train[meta_cols].astype("int32").values
        X_meta_test = self.test[meta_cols].astype("int32").values

        if debug:
            print(f"   Meta features shape: {X_meta_full.shape}")

        # ===== 3. 타겟 변환 =====
        y_full = np.log1p(self.train["price"].values)

        # ===== 4. Train/Valid Split =====
        # ✅ 수정: len() → .shape[0] (sparse matrix 호환)
        indices = np.arange(X_text_full.shape[0])
        train_idx, valid_idx = train_test_split(
            indices, test_size=0.2, random_state=self.random_state
        )

        # 텍스트
        self.X_text_train = X_text_full[train_idx]
        self.X_text_valid = X_text_full[valid_idx]
        self.X_text_full = X_text_full
        self.X_text_test = X_text_test

        # 메타
        self.X_meta_train = X_meta_full[train_idx]
        self.X_meta_valid = X_meta_full[valid_idx]
        self.X_meta_full = X_meta_full
        self.X_meta_test = X_meta_test

        # 타겟
        self.y_train = y_full[train_idx]
        self.y_valid = y_full[valid_idx]
        self.y_full = y_full

        # 캐시 저장
        if save_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    (
                        self.X_text_train,
                        self.X_text_valid,
                        self.X_text_full,
                        self.X_text_test,
                        self.X_meta_train,
                        self.X_meta_valid,
                        self.X_meta_full,
                        self.X_meta_test,
                        self.y_train,
                        self.y_valid,
                        self.y_full,
                        self.tfidf,
                        self.label_encoders,
                    ),
                    f,
                )
            if debug:
                print(f"\n💾 캐시 저장: {cache_path}")

        if debug:
            print("\n✅ 전처리 완료")
            print(f"   Train: {len(train_idx):,}, Valid: {len(valid_idx):,}")

    def train_ridge(self, alpha: float = 0.5, verbose: bool = True):
        """Stage 1-A: Ridge Regressor 학습 (텍스트 전용)"""
        if self.X_text_train is None:
            raise RuntimeError("preprocess()를 먼저 호출하세요.")

        if verbose:
            print("\n" + "=" * 80)
            print("  [Stage 1-A] Ridge Regressor 학습 (텍스트)")
            print("=" * 80)

        self.ridge_model = Ridge(alpha=alpha, random_state=self.random_state)
        self.ridge_model.fit(self.X_text_train, self.y_train)

        # 예측
        y_pred_log = self.ridge_model.predict(self.X_text_valid)

        # 메트릭 계산 (원래 스케일)
        y_true = np.expm1(self.y_valid)
        y_pred = np.maximum(np.expm1(y_pred_log), 0)

        metrics = self._calculate_metrics(y_true, y_pred)

        if verbose:
            print(f"✓ Ridge 학습 완료")
            print(f"  RMSLE: {metrics['rmsle']:.6f}")
            print(f"  RMSE:  {metrics['rmse']:.2f}")

        return metrics

    def train_lgbm(self, params: Optional[dict] = None, verbose: bool = True):
        """
        Stage 1-B: LightGBM 학습 (메타데이터 전용)

        ✅ 수정: early_stopping 콜백 수정
        """
        if self.X_meta_train is None:
            raise RuntimeError("preprocess()를 먼저 호출하세요.")

        if verbose:
            print("\n" + "=" * 80)
            print("  [Stage 1-B] LightGBM 학습 (메타데이터)")
            print("=" * 80)

        # 기본 파라미터 (1위 전략 기반)
        if params is None:
            params = {
                "n_estimators": 3000,
                "learning_rate": 0.75,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbose": -1,
            }

        self.lgbm_model = LGBMRegressor(**params)

        # ✅ 수정: early_stopping 콜백 수정
        try:
            from lightgbm import early_stopping, log_evaluation

            callbacks = [early_stopping(50, verbose=False)]
            if verbose:
                callbacks.append(log_evaluation(100))

            self.lgbm_model.fit(
                self.X_meta_train,
                self.y_train,
                eval_set=[(self.X_meta_valid, self.y_valid)],
                callbacks=callbacks,
            )
        except Exception as e:
            # Fallback: 콜백 없이 학습
            print(f"⚠️ early_stopping 오류, 콜백 없이 학습: {e}")
            self.lgbm_model.fit(
                self.X_meta_train,
                self.y_train,
                eval_set=[(self.X_meta_valid, self.y_valid)],
            )

        # 예측
        y_pred_log = self.lgbm_model.predict(self.X_meta_valid)

        # 메트릭
        y_true = np.expm1(self.y_valid)
        y_pred = np.maximum(np.expm1(y_pred_log), 0)

        metrics = self._calculate_metrics(y_true, y_pred)

        if verbose:
            print(f"✓ LightGBM 학습 완료")
            print(f"  RMSLE: {metrics['rmsle']:.6f}")
            print(f"  RMSE:  {metrics['rmse']:.2f}")

        return metrics

    def optimize_ensemble(self, verbose: bool = True):
        """Stage 2: 앙상블 가중치 최적화"""
        if self.ridge_model is None or self.lgbm_model is None:
            raise RuntimeError("train_ridge()와 train_lgbm()를 먼저 호출하세요.")

        if verbose:
            print("\n" + "=" * 80)
            print("  [Stage 2] 앙상블 가중치 최적화")
            print("=" * 80)

        # 각 모델의 예측값 (log scale)
        ridge_pred_log = self.ridge_model.predict(self.X_text_valid)
        lgbm_pred_log = self.lgbm_model.predict(self.X_meta_valid)

        best_rmsle = float("inf")
        best_weights = None

        # Grid search (0.0 ~ 1.0, 0.05 간격)
        for ridge_weight in np.arange(0.0, 1.05, 0.05):
            lgbm_weight = 1.0 - ridge_weight

            # 가중 평균 (log scale에서)
            ensemble_pred_log = (
                ridge_weight * ridge_pred_log + lgbm_weight * lgbm_pred_log
            )

            # 원래 스케일로 변환
            y_true = np.expm1(self.y_valid)
            y_pred = np.maximum(np.expm1(ensemble_pred_log), 0)

            # RMSLE 계산
            rmsle = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

            if rmsle < best_rmsle:
                best_rmsle = rmsle
                best_weights = {"ridge": ridge_weight, "lgbm": lgbm_weight}

        self.ensemble_weights = best_weights

        if verbose:
            print(f"✓ 최적 가중치 발견")
            print(f"  Ridge:    {best_weights['ridge']:.2f}")
            print(f"  LightGBM: {best_weights['lgbm']:.2f}")
            print(f"  RMSLE:    {best_rmsle:.6f}")

        return {**best_weights, "rmsle": best_rmsle}

    def evaluate(self, verbose: bool = True):
        """최종 앙상블 평가"""
        if self.ridge_model is None or self.lgbm_model is None:
            raise RuntimeError("모델을 먼저 학습하세요.")

        ridge_pred_log = self.ridge_model.predict(self.X_text_valid)
        lgbm_pred_log = self.lgbm_model.predict(self.X_meta_valid)

        ensemble_pred_log = (
            self.ensemble_weights["ridge"] * ridge_pred_log
            + self.ensemble_weights["lgbm"] * lgbm_pred_log
        )

        y_true = np.expm1(self.y_valid)
        y_pred = np.maximum(np.expm1(ensemble_pred_log), 0)

        self.metrics = self._calculate_metrics(y_true, y_pred)

        if verbose:
            print("\n" + "=" * 80)
            print("  최종 평가 (Ensemble)")
            print("=" * 80)
            for k, v in self.metrics.items():
                print(
                    f"  {k.upper()}: {v:.6f}"
                    if k == "rmsle"
                    else f"  {k.upper()}: {v:.4f}"
                )

        return self.metrics

    def _calculate_metrics(self, y_true, y_pred):
        """메트릭 계산 헬퍼"""
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "rmsle": float(
                np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
            ),
        }

    def predict_test(self, save_submission: bool = True):
        """테스트 데이터 예측 및 제출 파일 생성"""
        if self.ridge_model is None or self.lgbm_model is None:
            raise RuntimeError("모델을 먼저 학습하세요.")

        # 각 모델 예측
        ridge_pred_log = self.ridge_model.predict(self.X_text_test)
        lgbm_pred_log = self.lgbm_model.predict(self.X_meta_test)

        # 앙상블
        ensemble_pred_log = (
            self.ensemble_weights["ridge"] * ridge_pred_log
            + self.ensemble_weights["lgbm"] * lgbm_pred_log
        )

        y_pred = np.maximum(np.expm1(ensemble_pred_log), 0)

        # Submission 저장
        if save_submission:
            os.makedirs(self.results_dir, exist_ok=True)

            if "test_id" in self.test.columns:
                ids = self.test["test_id"].values
            elif "id" in self.test.columns:
                ids = self.test["id"].values
            else:
                ids = np.arange(len(self.test))

            sub_df = pd.DataFrame({"test_id": ids, "price": y_pred})

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_leader_{timestamp}.csv"
            save_path = os.path.join(self.results_dir, filename)

            sub_df.to_csv(save_path, index=False)
            print(f"📄 Submission 저장: {save_path}")

        return y_pred

    def save_models(self):
        """모델 저장"""
        os.makedirs(self.models_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ridge
        ridge_path = os.path.join(self.models_dir, f"ridge_leader_{timestamp}.pkl")
        with open(ridge_path, "wb") as f:
            pickle.dump(self.ridge_model, f)

        # LGBM
        lgbm_path = os.path.join(self.models_dir, f"lgbm_leader_{timestamp}.pkl")
        with open(lgbm_path, "wb") as f:
            pickle.dump(self.lgbm_model, f)

        # 앙상블 정보
        ensemble_info = {
            "weights": self.ensemble_weights,
            "metrics": self.metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        ensemble_path = os.path.join(
            self.results_dir, f"ensemble_info_{timestamp}.json"
        )
        with open(ensemble_path, "w") as f:
            json.dump(ensemble_info, f, indent=4)

        print(f"💾 모델 저장 완료:")
        print(f"   Ridge: {ridge_path}")
        print(f"   LGBM:  {lgbm_path}")
        print(f"   Info:  {ensemble_path}")

    # ========================================================================
    # 1. 텍스트 기반 피처
    # ========================================================================

    def add_text_features(self):
        """
        텍스트 길이/품질 피처 생성
        - 이름/설명 길이
        - 단어 수
        - 대문자 비율
        - 브랜드명 포함 여부
        """
        print("📝 텍스트 피처 생성 중...")

        for df in [self.train, self.test]:
            # 길이 피처
            df["name_len"] = df["name"].str.len()
            df["desc_len"] = df["item_description"].str.len()

            # 단어 수
            df["name_word_count"] = df["name"].str.split().str.len()
            df["desc_word_count"] = df["item_description"].str.split().str.len()

            # 대문자 비율 (품질 지표)
            df["name_capital_ratio"] = df["name"].apply(
                lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
            )

            # 설명/이름 길이 비율
            df["desc_name_ratio"] = (
                df["desc_len"] / df["name_len"].replace(0, 1)
            ).fillna(0)

            # 브랜드명이 이름에 포함되어 있는지
            df["has_brand_in_name"] = df.apply(
                lambda row: (
                    1
                    if str(row["brand_name"]).lower() in str(row["name"]).lower()
                    else 0
                ),
                axis=1,
            )

            # 브랜드명이 설명에 포함되어 있는지
            df["has_brand_in_desc"] = df.apply(
                lambda row: (
                    1
                    if str(row["brand_name"]).lower()
                    in str(row["item_description"]).lower()
                    else 0
                ),
                axis=1,
            )

        print("  ✓ 텍스트 피처 8개 생성 완료")

    # ========================================================================
    # 2. 가격 통계 피처 (가장 중요!)
    # ========================================================================

    def add_price_statistics(self):
        """
        브랜드/카테고리별 가격 통계 피처
        - 평균, 중앙값, 표준편차, 최소, 최대
        - 이 피처들이 RMSLE를 크게 개선시킴
        """
        print("💰 가격 통계 피처 생성 중...")

        # log1p 변환된 가격 사용
        price_log = np.log1p(self.train["price"])

        # 1) 브랜드별 통계
        brand_stats = (
            self.train.groupby("brand_name")["price"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .add_prefix("brand_price_")
        )

        self.train = self.train.merge(brand_stats, on="brand_name", how="left")
        self.test = self.test.merge(brand_stats, on="brand_name", how="left")

        # 2) 카테고리 1단계 통계
        cat1_stats = (
            self.train.groupby("cat1")["price"]
            .agg(["mean", "median", "std"])
            .add_prefix("cat1_price_")
        )

        self.train = self.train.merge(cat1_stats, on="cat1", how="left")
        self.test = self.test.merge(cat1_stats, on="cat1", how="left")

        # 3) 카테고리 2단계 통계
        cat2_stats = (
            self.train.groupby("cat2")["price"]
            .agg(["mean", "median", "std"])
            .add_prefix("cat2_price_")
        )

        self.train = self.train.merge(cat2_stats, on="cat2", how="left")
        self.test = self.test.merge(cat2_stats, on="cat2", how="left")

        # 4) 카테고리 3단계 통계
        cat3_stats = (
            self.train.groupby("cat3")["price"]
            .agg(["mean", "median"])
            .add_prefix("cat3_price_")
        )

        self.train = self.train.merge(cat3_stats, on="cat3", how="left")
        self.test = self.test.merge(cat3_stats, on="cat3", how="left")

        # 결측값 처리 (test에 없는 브랜드/카테고리)
        stat_cols = [c for c in self.train.columns if "_price_" in c]
        for df in [self.train, self.test]:
            df[stat_cols] = df[stat_cols].fillna(df[stat_cols].median())

        print(f"  ✓ 가격 통계 피처 {len(stat_cols)}개 생성 완료")

    # ========================================================================
    # 3. 상호작용 피처
    # ========================================================================

    def add_interaction_features(self):
        """
        범주형 변수 조합 피처
        - 브랜드 × 카테고리
        - 상태 × 배송
        """
        print("🔗 상호작용 피처 생성 중...")

        for df in [self.train, self.test]:
            # 브랜드 × 카테고리 1단계
            df["brand_cat1"] = (
                df["brand_name"].astype(str) + "_" + df["cat1"].astype(str)
            )

            # 브랜드 × 카테고리 2단계
            df["brand_cat2"] = (
                df["brand_name"].astype(str) + "_" + df["cat2"].astype(str)
            )

            # 상태 × 배송
            df["condition_shipping"] = (
                df["item_condition_id"].astype(str) + "_" + df["shipping"].astype(str)
            )

            # 카테고리 1 × 2 조합
            df["cat1_cat2"] = df["cat1"].astype(str) + "_" + df["cat2"].astype(str)

        print("  ✓ 상호작용 피처 4개 생성 완료")

    # ========================================================================
    # 4. 희귀성 피처
    # ========================================================================

    def add_rarity_features(self):
        """
        브랜드/카테고리의 희귀성 지표
        - 등장 빈도가 낮으면 가격 예측이 어려움
        """
        print("🔍 희귀성 피처 생성 중...")

        # 브랜드 빈도
        brand_counts = self.train["brand_name"].value_counts()
        for df in [self.train, self.test]:
            df["brand_rarity"] = df["brand_name"].map(brand_counts).fillna(0)
            df["is_rare_brand"] = (df["brand_rarity"] < 10).astype(int)

        # 카테고리 빈도
        for cat in ["cat1", "cat2", "cat3"]:
            cat_counts = self.train[cat].value_counts()
            for df in [self.train, self.test]:
                df[f"{cat}_count"] = df[cat].map(cat_counts).fillna(0)

        print("  ✓ 희귀성 피처 7개 생성 완료")

    # ========================================================================
    # 5. 통합 메서드
    # ========================================================================

    def add_all_features(self):
        """
        모든 피처를 한 번에 생성

        사용법:
          analyzer.load_data()
          analyzer.add_all_features()  # ← 이것만 호출
          analyzer.preprocess(...)
        """
        print("\n" + "=" * 80)
        print("  🚀 피처 엔지니어링 시작")
        print("=" * 80 + "\n")

        # 카테고리 분할 (필수)
        self._split_category(self.train)
        self._split_category(self.test)

        # 1단계: 텍스트 피처
        self.add_text_features()

        # 2단계: 가격 통계 (가장 중요!)
        self.add_price_statistics()

        # 3단계: 상호작용
        self.add_interaction_features()

        # 4단계: 희귀성
        self.add_rarity_features()

        total_features = (
            8  # 텍스트
            + len([c for c in self.train.columns if "_price_" in c])  # 가격 통계
            + 4  # 상호작용
            + 7  # 희귀성
        )

        print(f"\n✅ 피처 엔지니어링 완료! 총 {total_features}개 피처 추가")
        print("=" * 80 + "\n")

    # ========================================================================
    # 6. preprocess 메서드 수정 (메타 피처에 새 피처 추가)
    # ========================================================================

    def preprocess_with_new_features(
        self,
        use_cache: bool = False,  # 새 피처 추가 시 캐시 사용 안 함
        save_cache: bool = True,
        tfidf_max_features: int = 100000,  # 50K → 100K
        debug: bool = True,
    ):
        """
        기존 preprocess에 새 피처를 메타데이터에 포함

        ⚠️ 주의: add_all_features()를 먼저 호출해야 함!
        """
        if self.train is None or self.test is None:
            raise RuntimeError("load_data()를 먼저 호출하세요.")

        # 새 피처가 있는지 확인
        if "name_len" not in self.train.columns:
            print("⚠️ add_all_features()를 먼저 호출하세요!")
            return

        cache_dir = os.path.join(self.results_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "mercari_enhanced_preprocessed.pkl")

        # ===== 1. 텍스트 파이프라인 =====
        if debug:
            print("\n📝 [Stage 1-A] 텍스트 전처리 (TF-IDF)")

        self.train["text_all"] = (
            self.train["name"].astype(str)
            + " "
            + self.train["item_description"].astype(str)
        )
        self.test["text_all"] = (
            self.test["name"].astype(str)
            + " "
            + self.test["item_description"].astype(str)
        )

        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.5,
            sublinear_tf=True,
            stop_words="english",
        )

        X_text_full = self.tfidf.fit_transform(self.train["text_all"])
        X_text_test = self.tfidf.transform(self.test["text_all"])

        if debug:
            print(f"   TF-IDF shape: {X_text_full.shape}")

        # ===== 2. 메타데이터 파이프라인 (새 피처 포함!) =====
        if debug:
            print("\n🏷️  [Stage 1-B] 메타데이터 전처리 (확장)")

        # Label Encoding (기존)
        cat_cols = ["brand_name", "cat1", "cat2", "cat3"]

        for col in cat_cols:
            le = LabelEncoder()
            all_values = pd.concat([self.train[col], self.test[col]]).unique()
            le.fit(all_values)

            self.train[f"{col}_encoded"] = le.transform(self.train[col])
            self.test[f"{col}_encoded"] = le.transform(self.test[col])

            self.label_encoders[col] = le

        # 상호작용 피처도 인코딩
        interaction_cols = [
            "brand_cat1",
            "brand_cat2",
            "condition_shipping",
            "cat1_cat2",
        ]

        for col in interaction_cols:
            if col in self.train.columns:
                le = LabelEncoder()
                all_values = pd.concat([self.train[col], self.test[col]]).unique()
                le.fit(all_values)

                self.train[f"{col}_encoded"] = le.transform(self.train[col])
                self.test[f"{col}_encoded"] = le.transform(self.test[col])

                self.label_encoders[col] = le

        # ✅ 메타 피처 구성 (확장!)
        meta_cols = [
            # 기본
            "item_condition_id",
            "shipping",
            "brand_name_encoded",
            "cat1_encoded",
            "cat2_encoded",
            "cat3_encoded",
            # 텍스트 피처
            "name_len",
            "desc_len",
            "name_word_count",
            "desc_word_count",
            "name_capital_ratio",
            "desc_name_ratio",
            "has_brand_in_name",
            "has_brand_in_desc",
            # 가격 통계
            "brand_price_mean",
            "brand_price_median",
            "brand_price_std",
            "brand_price_count",
            "cat1_price_mean",
            "cat1_price_median",
            "cat1_price_std",
            "cat2_price_mean",
            "cat2_price_median",
            "cat2_price_std",
            "cat3_price_mean",
            "cat3_price_median",
            # 상호작용
            "brand_cat1_encoded",
            "brand_cat2_encoded",
            "condition_shipping_encoded",
            "cat1_cat2_encoded",
            # 희귀성
            "brand_rarity",
            "is_rare_brand",
            "cat1_count",
            "cat2_count",
            "cat3_count",
        ]

        # 존재하는 컬럼만 선택
        meta_cols = [c for c in meta_cols if c in self.train.columns]

        X_meta_full = self.train[meta_cols].fillna(0).astype("float32").values
        X_meta_test = self.test[meta_cols].fillna(0).astype("float32").values

        if debug:
            print(
                f"   Meta features: {len(meta_cols)}개 (기존 6개 → {len(meta_cols)}개)"
            )

        # ===== 3. 타겟 변환 =====
        y_full = np.log1p(self.train["price"].values)

        # ===== 4. Train/Valid Split =====
        from sklearn.model_selection import train_test_split

        indices = np.arange(X_text_full.shape[0])
        train_idx, valid_idx = train_test_split(
            indices, test_size=0.2, random_state=self.random_state
        )

        # 텍스트
        self.X_text_train = X_text_full[train_idx]
        self.X_text_valid = X_text_full[valid_idx]
        self.X_text_full = X_text_full
        self.X_text_test = X_text_test

        # 메타
        self.X_meta_train = X_meta_full[train_idx]
        self.X_meta_valid = X_meta_full[valid_idx]
        self.X_meta_full = X_meta_full
        self.X_meta_test = X_meta_test

        # 타겟
        self.y_train = y_full[train_idx]
        self.y_valid = y_full[valid_idx]
        self.y_full = y_full

        if debug:
            print("\n✅ 전처리 완료 (확장 버전)")
            print(f"   Train: {len(train_idx):,}, Valid: {len(valid_idx):,}")
            print(f"   메타 피처: {X_meta_full.shape[1]}개")

    """
    고급 앙상블 메서드 모음
    기존 MercariLeaderAnalyzer에 추가
    """

    # ========================================================================
    # 1. XGBoost 학습
    # ========================================================================

    def train_xgboost(self, params=None, verbose=True):
        """
        XGBoost 학습 (메타데이터 전용)

        특징:
        - LightGBM과 다른 방식으로 트리 구성
        - 정규화가 더 강함
        - 과적합에 강함
        """
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost가 설치되지 않았습니다.")
            return None

        if self.X_meta_train is None:
            raise RuntimeError("preprocess()를 먼저 호출하세요.")

        if verbose:
            print("\n" + "=" * 80)
            print("  [XGBoost] 학습 시작")
            print("=" * 80)

        # 기본 파라미터
        if params is None:
            params = {
                "n_estimators": 3000,
                "learning_rate": 0.05,
                "max_depth": 8,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "tree_method": "hist",  # 빠른 학습
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbosity": 0,
                "early_stopping_rounds": 50,
            }

        self.xgb_model = XGBRegressor(**params)

        self.xgb_model.fit(
            self.X_meta_train,
            self.y_train,
            eval_set=[(self.X_meta_valid, self.y_valid)],
            verbose=False,
        )

        # 예측
        y_pred_log = self.xgb_model.predict(self.X_meta_valid)
        y_true = np.expm1(self.y_valid)
        y_pred = np.maximum(np.expm1(y_pred_log), 0)

        metrics = self._calculate_metrics(y_true, y_pred)

        if verbose:
            print(f"✓ XGBoost 학습 완료")
            print(f"  RMSLE: {metrics['rmsle']:.6f}")
            print(f"  RMSE:  {metrics['rmse']:.2f}")

        return metrics

    # ========================================================================
    # 2. CatBoost 학습
    # ========================================================================

    def train_catboost(self, params=None, verbose=True):
        """
        CatBoost 학습 (메타데이터 전용)

        특징:
        - 범주형 변수 자동 처리 (인코딩 불필요)
        - Ordered Boosting으로 과적합 방지
        - GPU 지원 우수
        """
        if not CATBOOST_AVAILABLE:
            print("❌ CatBoost가 설치되지 않았습니다.")
            return None

        if self.X_meta_train is None:
            raise RuntimeError("preprocess()를 먼저 호출하세요.")

        if verbose:
            print("\n" + "=" * 80)
            print("  [CatBoost] 학습 시작")
            print("=" * 80)

        # 기본 파라미터
        if params is None:
            params = {
                "iterations": 3000,
                "learning_rate": 0.05,
                "depth": 8,
                "l2_leaf_reg": 3,
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": self.random_state,
                "verbose": False,
                "early_stopping_rounds": 50,
            }

        self.catboost_model = CatBoostRegressor(**params)

        self.catboost_model.fit(
            self.X_meta_train,
            self.y_train,
            eval_set=(self.X_meta_valid, self.y_valid),
            verbose=False,
        )

        # 예측
        y_pred_log = self.catboost_model.predict(self.X_meta_valid)
        y_true = np.expm1(self.y_valid)
        y_pred = np.maximum(np.expm1(y_pred_log), 0)

        metrics = self._calculate_metrics(y_true, y_pred)

        if verbose:
            print(f"✓ CatBoost 학습 완료")
            print(f"  RMSLE: {metrics['rmsle']:.6f}")
            print(f"  RMSE:  {metrics['rmse']:.2f}")

        return metrics

    # ========================================================================
    # 3. 4-Model 앙상블 최적화
    # ========================================================================

    def optimize_4model_ensemble(self, verbose=True):
        """
        Ridge + LightGBM + XGBoost + CatBoost 가중치 최적화

        방법: Grid Search (0.0 ~ 1.0, 더 세밀하게)
        """
        if verbose:
            print("\n" + "=" * 80)
            print("  [4-Model Ensemble] 가중치 최적화")
            print("=" * 80)

        # 각 모델 예측
        ridge_pred = (
            self.ridge_model.predict(self.X_text_valid) if self.ridge_model else None
        )
        lgbm_pred = (
            self.lgbm_model.predict(self.X_meta_valid) if self.lgbm_model else None
        )
        xgb_pred = (
            self.xgb_model.predict(self.X_meta_valid)
            if hasattr(self, "xgb_model")
            else None
        )
        cat_pred = (
            self.catboost_model.predict(self.X_meta_valid)
            if hasattr(self, "catboost_model")
            else None
        )

        # 사용 가능한 모델만 선택
        available_models = {}
        if ridge_pred is not None:
            available_models["ridge"] = ridge_pred
        if lgbm_pred is not None:
            available_models["lgbm"] = lgbm_pred
        if xgb_pred is not None:
            available_models["xgb"] = xgb_pred
        if cat_pred is not None:
            available_models["catboost"] = cat_pred

        n_models = len(available_models)

        if n_models < 2:
            print("⚠️ 앙상블을 위해 최소 2개 모델이 필요합니다.")
            return None

        if verbose:
            print(f"  사용 가능한 모델: {list(available_models.keys())}")

        # Grid Search
        best_rmsle = float("inf")
        best_weights = None

        # 4모델 그리드 (간단 버전: 균등 분할)
        from itertools import product

        step = 0.1
        weight_range = np.arange(0, 1.0 + step, step)

        if n_models == 2:
            # 2모델: 단순 그리드
            for w1 in weight_range:
                w2 = 1.0 - w1
                weights = [w1, w2]

                ensemble_pred = sum(
                    w * pred for w, pred in zip(weights, available_models.values())
                )

                y_true = np.expm1(self.y_valid)
                y_pred = np.maximum(np.expm1(ensemble_pred), 0)
                rmsle = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

                if rmsle < best_rmsle:
                    best_rmsle = rmsle
                    best_weights = dict(zip(available_models.keys(), weights))

        elif n_models == 3:
            # 3모델: 2D 그리드
            for w1 in weight_range:
                for w2 in weight_range:
                    w3 = 1.0 - w1 - w2
                    if w3 < 0:
                        continue

                    weights = [w1, w2, w3]
                    ensemble_pred = sum(
                        w * pred for w, pred in zip(weights, available_models.values())
                    )

                    y_true = np.expm1(self.y_valid)
                    y_pred = np.maximum(np.expm1(ensemble_pred), 0)
                    rmsle = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

                    if rmsle < best_rmsle:
                        best_rmsle = rmsle
                        best_weights = dict(zip(available_models.keys(), weights))

        else:  # 4모델
            # 4모델: 3D 그리드 (시간 오래 걸림)
            # 간소화: 균등 가중치 근처만 탐색
            for w1 in np.arange(0.1, 0.5, 0.05):
                for w2 in np.arange(0.1, 0.5, 0.05):
                    for w3 in np.arange(0.1, 0.5, 0.05):
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 < 0.1 or w4 > 0.5:
                            continue

                        weights = [w1, w2, w3, w4]
                        ensemble_pred = sum(
                            w * pred
                            for w, pred in zip(weights, available_models.values())
                        )

                        y_true = np.expm1(self.y_valid)
                        y_pred = np.maximum(np.expm1(ensemble_pred), 0)
                        rmsle = np.sqrt(
                            np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
                        )

                        if rmsle < best_rmsle:
                            best_rmsle = rmsle
                            best_weights = dict(zip(available_models.keys(), weights))

        self.ensemble_weights = best_weights

        if verbose:
            print(f"\n✓ 최적 가중치 발견:")
            for model_name, weight in best_weights.items():
                print(f"  {model_name:10s}: {weight:.3f}")
            print(f"  RMSLE: {best_rmsle:.6f}")

        return {"weights": best_weights, "rmsle": best_rmsle}

    # ========================================================================
    # 4. 최종 평가 (4-Model)
    # ========================================================================

    def evaluate_4model(self, verbose=True):
        """4-Model 앙상블 최종 평가"""

        # 각 모델 예측
        predictions = {}

        if self.ridge_model:
            predictions["ridge"] = self.ridge_model.predict(self.X_text_valid)
        if self.lgbm_model:
            predictions["lgbm"] = self.lgbm_model.predict(self.X_meta_valid)
        if hasattr(self, "xgb_model"):
            predictions["xgb"] = self.xgb_model.predict(self.X_meta_valid)
        if hasattr(self, "catboost_model"):
            predictions["catboost"] = self.catboost_model.predict(self.X_meta_valid)

        # 가중 평균
        ensemble_pred_log = sum(
            self.ensemble_weights.get(name, 0) * pred
            for name, pred in predictions.items()
        )

        y_true = np.expm1(self.y_valid)
        y_pred = np.maximum(np.expm1(ensemble_pred_log), 0)

        self.metrics = self._calculate_metrics(y_true, y_pred)

        if verbose:
            print("\n" + "=" * 80)
            print("  최종 평가 (4-Model Ensemble)")
            print("=" * 80)
            for k, v in self.metrics.items():
                print(
                    f"  {k.upper()}: {v:.6f}"
                    if k == "rmsle"
                    else f"  {k.upper()}: {v:.4f}"
                )

        return self.metrics

    # ========================================================================
    # 5. 테스트 예측 (4-Model)
    # ========================================================================

    def predict_test_4model(self, save_submission=True):
        """4-Model 앙상블로 테스트 예측"""

        predictions = {}

        if self.ridge_model:
            predictions["ridge"] = self.ridge_model.predict(self.X_text_test)
        if self.lgbm_model:
            predictions["lgbm"] = self.lgbm_model.predict(self.X_meta_test)
        if hasattr(self, "xgb_model"):
            predictions["xgb"] = self.xgb_model.predict(self.X_meta_test)
        if hasattr(self, "catboost_model"):
            predictions["catboost"] = self.catboost_model.predict(self.X_meta_test)

        # 가중 평균
        ensemble_pred_log = sum(
            self.ensemble_weights.get(name, 0) * pred
            for name, pred in predictions.items()
        )

        y_pred = np.maximum(np.expm1(ensemble_pred_log), 0)

        # Submission 저장
        if save_submission:
            import os
            from datetime import datetime

            os.makedirs(self.results_dir, exist_ok=True)

            if "test_id" in self.test.columns:
                ids = self.test["test_id"].values
            elif "id" in self.test.columns:
                ids = self.test["id"].values
            else:
                ids = np.arange(len(self.test))

            sub_df = pd.DataFrame({"test_id": ids, "price": y_pred})

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_4model_{timestamp}.csv"
            save_path = os.path.join(self.results_dir, filename)

            sub_df.to_csv(save_path, index=False)
            print(f"📄 Submission 저장: {save_path}")

        return y_pred

    # ============================================================================
    # 빠른 실행 스크립트
    # ============================================================================

    def quick_4model_ensemble(analyzer):
        """
        4-Model 앙상블 빠른 실행

        사용법:
        analyzer = MercariLeaderAnalyzer()
        analyzer.load_data()
        analyzer.add_all_features()
        analyzer.preprocess_with_new_features()

        # 기존 2-Model
        analyzer.train_ridge(alpha=0.5)
        analyzer.train_lgbm()

        # 4-Model 확장
        quick_4model_ensemble(analyzer)
        """
        print("\n" + "=" * 80)
        print("  🚀 4-Model 앙상블 파이프라인")
        print("=" * 80 + "\n")

        # 1. XGBoost
        if XGBOOST_AVAILABLE:
            analyzer.train_xgboost()

        # 2. CatBoost
        if CATBOOST_AVAILABLE:
            analyzer.train_catboost()

        # 3. 앙상블 최적화
        analyzer.optimize_4model_ensemble()

        # 4. 평가
        analyzer.evaluate_4model()

        # 5. 테스트 예측
        analyzer.predict_test_4model()

        print("\n✅ 4-Model 앙상블 완료!")


# ============================================================================
# 사용 예시
# ============================================================================

"""
# Step 1: 기존 메서드들을 MercariLeaderAnalyzer에 추가

# Step 2: 실행
analyzer = MercariLeaderAnalyzer()
analyzer.load_data()
analyzer.add_all_features()
analyzer.preprocess_with_new_features(use_cache=False, tfidf_max_features=100000)

# 기본 2-Model
analyzer.train_ridge(alpha=0.5)
analyzer.train_lgbm(params={'learning_rate': 0.05, 'n_estimators': 5000})

# ⭐ 4-Model 확장
analyzer.train_xgboost()
analyzer.train_catboost()
analyzer.optimize_4model_ensemble()
analyzer.evaluate_4model()

# 예상 결과: RMSLE 0.461 → 0.42-0.43 (5-8% 개선)

# 테스트 예측
analyzer.predict_test_4model()
"""

print("✅ 고급 앙상블 모듈 로드 완료!")
print(
    """
설치 필요:
  pip install xgboost
  pip install catboost

실행 순서:
  1. analyzer.train_xgboost()
  2. analyzer.train_catboost()
  3. analyzer.optimize_4model_ensemble()
  4. analyzer.evaluate_4model()
  
예상: RMSLE 0.461 → 0.42-0.43
"""
)

# ============================================================================
# 사용 예시
# ============================================================================

"""
# 기존 MercariLeaderAnalyzer에 위 메서드들을 추가한 후:

analyzer = MercariLeaderAnalyzer()

# 1. 데이터 로드
analyzer.load_data()

# 2. ⭐ 피처 엔지니어링 (새로 추가!)
analyzer.add_all_features()

# 3. 전처리 (메타 피처 확장 버전)
analyzer.preprocess_with_new_features(
    use_cache=False,  # 새 피처이므로 캐시 사용 안 함
    tfidf_max_features=100000
)

# 4. 모델 학습
analyzer.train_ridge(alpha=0.5)

lgbm_params = {
    'n_estimators': 5000,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 10,
}
analyzer.train_lgbm(params=lgbm_params)

# 5. 앙상블 최적화
analyzer.optimize_ensemble()

# 6. 평가
analyzer.evaluate()

# 예상 결과: RMSLE 0.47 → 0.43-0.44 (7-10% 개선)
"""


print("✅ 피처 엔지니어링 모듈 로드 완료!")
print(
    """
다음 단계:
  1. 위 메서드들을 MercariLeaderAnalyzer 클래스에 추가
  2. analyzer.add_all_features() 호출
  3. analyzer.preprocess_with_new_features() 실행
  4. 모델 재학습 및 평가
  
예상 개선: RMSLE 0.47 → 0.43-0.44
"""
)

print("✅ MercariLeaderAnalyzer 로드 완료!")

print(
    """
사용법:
  analyzer = MercariLeaderAnalyzer()
  analyzer.load_data()
  analyzer.preprocess(tfidf_max_features=50000)
  analyzer.train_ridge(alpha=0.5)
  analyzer.train_lgbm()
  analyzer.optimize_ensemble()
  analyzer.evaluate()
  analyzer.predict_test()
"""
)
```