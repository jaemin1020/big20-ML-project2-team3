# ============================================
# Mercari Price Prediction - Clean Full Code
# (구버전 제거 / 최신 버전만 정리본)
# ============================================

import os
import re
import gc
import json
import pickle
import hashlib
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

# ML / NLP
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA

# optional imports
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from gensim.models import FastText, Word2Vec
except Exception:
    FastText = None
    Word2Vec = None

# PyCaret 회귀
try:
    from pycaret.regression import (
        setup,
        create_model,
        compare_models,
        tune_model,
        save_model,
        load_model,
        predict_model,
        pull,
        get_config,
    )
except Exception:
    setup = create_model = compare_models = tune_model = None
    save_model = load_model = predict_model = pull = get_config = None

import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================================
# ===================== CLASS: MercariPyCaretAnalyzer9 ==================
# ======================================================================

class MercariPyCaretAnalyzer9:
    """
    Mercari 가격 예측용 end-to-end 파이프라인 클래스 (PyCaret 3.x 기준)
    - 데이터 로딩 / 전처리 / 벡터화 / PyCaret 모델링 / 평가 / 제출 파일 생성
    - 단계별 stage 캐시 + vectorized 캐시 지원
    """

    # [__init__] start #########################################################
    def __init__(
        self,
        data_dir="../data",
        images_dir="../images",
        results_dir="../results",
        model_dir="../models",
        use_gpu=True,
    ):
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.results_dir = results_dir
        self.model_dir = model_dir
        self.use_gpu = use_gpu

        self.device = None
        self._detect_device()

        # 원본 train/test
        self.train = None
        self.test = None

        # 벡터화된 데이터
        self.train_vectorized = None
        self.test_vectorized = None

        # PyCaret 관련
        self.setup_result = None
        self.best_model = None
        self.metrics = {}
        self.models = {}

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    # [__init__] end ===========================================================

    # [_detect_device] start ####################################################
    def _detect_device(self):
        device = "cpu"
        try:
            if self.use_gpu and torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"
        self.device = device
        return self.device

    # [_detect_device] end =====================================================

    # [load_data] start #########################################################
    def load_data(
        self,
        train_file="train.tsv",
        test_file="test.tsv",
        sep="\t",
        undersample_frac=None,
    ):
        print("📂 데이터 로딩 시작...")

        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)

        self.train = pd.read_csv(train_path, sep=sep)
        self.test = pd.read_csv(test_path, sep=sep)

        # price 전처리 (log1p)
        self.train = (
            self.train[self.train["price"] > 0]
            .dropna(subset=["price"])
            .reset_index(drop=True)
        )
        self.train["price"] = np.log1p(self.train["price"])

        # category split 및 기본 fill
        for df in [self.train, self.test]:
            df["category_name"] = df.get("category_name", "").fillna("missing")

            def _split_cat(x):
                parts = x.split("/") if isinstance(x, str) else ["missing"] * 3
                while len(parts) < 3:
                    parts.append("missing")
                return parts[:3]

            df["main_cat"], df["sub_cat"], df["sub_sub_cat"] = zip(
                *df["category_name"].apply(_split_cat)
            )

            df["brand_name"] = df["brand_name"].fillna("Unknown").astype(str)
            df["item_description"] = df["item_description"].fillna("No description").astype(str)
            df["name"] = df["name"].fillna("No name").astype(str)

            if "shipping" in df.columns:
                df["shipping"] = df["shipping"].astype("category")
            if "item_condition_id" in df.columns:
                df["item_condition_id"] = df["item_condition_id"].astype("category")

        print("🔄 희귀값 통합 중...")
        self._collapse_rare_values("brand_name", top_k=5000, rare_label="Other_brand")
        self._collapse_rare_values("main_cat", top_k=1000, rare_label="Other_main")
        self._collapse_rare_values("sub_cat", top_k=1000, rare_label="Other_sub")
        self._collapse_rare_values("sub_sub_cat", top_k=1000, rare_label="Other_sub_sub")

        if undersample_frac:
            self._stratified_sample(frac=undersample_frac)

        if "price" not in self.train.columns:
            raise RuntimeError("❌ load_data 이후 price 컬럼이 누락되었습니다.")

        print(f"✅ 데이터 로딩 완료: train {self.train.shape}, test {self.test.shape}")

    # [load_data] end ==========================================================

    # [_collapse_rare_values] start ############################################
    def _collapse_rare_values(self, col, top_k, rare_label="Other"):
        combined = pd.concat([self.train[col], self.test[col]], axis=0)
        value_counts = combined.value_counts()
        top_values = set(value_counts.index[:top_k])

        self.train[col] = self.train[col].apply(lambda x: x if x in top_values else rare_label)
        self.test[col] = self.test[col].apply(lambda x: x if x in top_values else rare_label)

        del combined, value_counts
        gc.collect()

    # [_collapse_rare_values] end ==============================================

    # [_stratified_sample] start ###############################################
    def _stratified_sample(self, frac=0.35, bins=10):
        print("⚠️ Stratified undersampling 수행...")
        df = self.train.copy()
        df["price_bin"] = pd.qcut(df["price"], q=bins, duplicates="drop")
        sampled = (
            df.groupby("price_bin", group_keys=False, observed=True)
            .apply(lambda x: x.sample(frac=frac, random_state=23))
            .reset_index(drop=True)
        )
        sampled.drop(columns=["price_bin"], inplace=True)
        self.train = sampled
        print(f"👉 언더샘플 적용 후 train: {self.train.shape}")

    # [_stratified_sample] end ================================================

    # [normalize_text] start ####################################################
    def normalize_text(
        self,
        cols=("name", "item_description"),
        lower=True,
        strip_punct=True,
        numbers_to_token=True,
    ):
        print("🧼 텍스트 정규화 시작...")

        def _norm(text):
            t = str(text)
            if lower:
                t = t.lower()
            if strip_punct:
                t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
            if numbers_to_token:
                t = re.sub(r"\d+", " num ", t)
            return re.sub(r"\s+", " ", t).strip()

        for df in [self.train, self.test]:
            for c in cols:
                df[f"{c}_norm"] = df[c].astype(str).apply(_norm)

        print("✅ 텍스트 정규화 완료")

    # [normalize_text] end =====================================================

    # [build_text_stats] start #################################################
    def build_text_stats(self, cols=("name", "item_description")):
        print("📈 텍스트 통계 피처 생성 중...")

        def _stats(s):
            s = str(s)
            tokens = s.split()
            token_count = len(tokens)
            uniq = len(set(tokens))
            avg_word_len = np.mean([len(w) for w in tokens]) if tokens else 0.0
            upper_ratio = sum(1 for ch in s if ch.isupper()) / max(1, len(s))
            digit_ratio = sum(1 for ch in s if ch.isdigit()) / max(1, len(s))
            punct_ratio = sum(
                1 for ch in s if re.match(r"[^\w\s]", ch)
            ) / max(1, len(s))
            return token_count, uniq, avg_word_len, upper_ratio, digit_ratio, punct_ratio

        for base in cols:
            for df in [self.train, self.test]:
                arr = df[base].astype(str).apply(_stats).tolist()
                tok, uniq, avgw, upr, dr, pr = zip(*arr)
                df[f"{base}_tok_count"] = tok
                df[f"{base}_uniq_tok"] = uniq
                df[f"{base}_avg_word_len"] = avgw
                df[f"{base}_upper_ratio"] = upr
                df[f"{base}_digit_ratio"] = dr
                df[f"{base}_punct_ratio"] = pr

        for df in [self.train, self.test]:
            df["name_len_char"] = df["name"].astype(str).str.len()
            df["name_len_word"] = df["name"].astype(str).str.split().str.len()
            df["desc_len_char"] = df["item_description"].astype(str).str.len()
            df["desc_len_word"] = df["item_description"].astype(str).str.split().str.len()
            df["desc_name_char_ratio"] = (
                df["desc_len_char"] / df["name_len_char"]
            ).replace([np.inf, -np.inf], 0).fillna(0)
            df["desc_name_word_ratio"] = (
                df["desc_len_word"] / df["name_len_word"]
            ).replace([np.inf, -np.inf], 0).fillna(0)

            df["has_brand_in_name"] = df.apply(
                lambda r: 1
                if str(r["brand_name"]).lower() in str(r["name"]).lower()
                else 0,
                axis=1,
            )
            df["has_brand_in_desc"] = df.apply(
                lambda r: 1
                if str(r["brand_name"]).lower()
                in str(r["item_description"]).lower()
                else 0,
                axis=1,
            )

        print("✅ 텍스트 통계 피처 생성 완료")

    # [build_text_stats] end ===================================================

    # [build_price_brand_cat_features] start ###################################
    def build_price_brand_cat_features(
        self,
        price_col="price",
        brand_col="brand_name",
        cat_cols=("main_cat", "sub_cat", "sub_sub_cat"),
        rare_thresh_brand=20,
        rare_thresh_cat=20,
    ):
        """
        가격-브랜드-카테고리 기반 통계 피처 (SAFE)
        """
        print("\n=== SAFE VERSION: build_price_brand_cat_features ===")

        if price_col not in self.train.columns:
            raise RuntimeError("❌ train 데이터에 price가 없습니다.")

        grp_cols = [brand_col] + list(cat_cols)

        # 1) 그룹 통계
        for col in grp_cols:
            print(f"📊 building stats for '{col}' ...")
            stats = (
                self.train.groupby(col)[price_col]
                .agg(["mean", "median", "std"])
                .rename(
                    columns={
                        "mean": f"{col}_price_mean",
                        "median": f"{col}_price_median",
                        "std": f"{col}_price_std",
                    }
                )
            )

            self.train = self.train.join(stats, on=col)
            self.test = self.test.join(stats, on=col)

            for df in [self.train, self.test]:
                for cc in [
                    f"{col}_price_mean",
                    f"{col}_price_median",
                    f"{col}_price_std",
                ]:
                    if cc in df.columns:
                        df[cc] = df[cc].fillna(0)

        # 2) train brand_price_z
        print("⚖️ computing z-score ...")
        mu_train = self.train[f"{brand_col}_price_mean"]
        sd_train = self.train[f"{brand_col}_price_std"].replace(0, 1)

        self.train["brand_price_z"] = (
            (self.train[price_col] - mu_train) / sd_train
        ).replace([np.inf, -np.inf], 0).fillna(0)

        self.test["brand_price_z"] = 0.0

        # 3) rare flags
        brand_freq = self.train[brand_col].value_counts()
        cat_freqs = {c: self.train[c].value_counts() for c in cat_cols}

        for df_name in ("train", "test"):
            df = getattr(self, df_name)
            df["rare_brand"] = df[brand_col].apply(
                lambda x: int(brand_freq.get(x, 0) < rare_thresh_brand)
            )
            for c in cat_cols:
                df[f"rare_{c}"] = df[c].apply(
                    lambda x: int(cat_freqs[c].get(x, 0) < rare_thresh_cat)
                )
            setattr(self, df_name, df)

        # 4) brand in text
        def _brand_in_field(row, bcol, fcol):
            b = str(row[bcol]).lower()
            f = str(row[fcol]).lower()
            return int(b != "" and b in f)

        for df_name in ("train", "test"):
            df = getattr(self, df_name)
            df["brand_in_name_ratio"] = df.apply(
                lambda r: _brand_in_field(r, brand_col, "name"),
                axis=1,
            )
            df["brand_in_desc_ratio"] = df.apply(
                lambda r: _brand_in_field(r, brand_col, "item_description"),
                axis=1,
            )
            setattr(self, df_name, df)

        print("✅ SAFE price-brand-category feature creation DONE.")

    # [build_price_brand_cat_features] end =====================================

    # [build_interactions] start ###############################################
    def build_interactions(
        self,
        pairs=(("item_condition_id", "shipping"), ("brand_name", "main_cat")),
    ):
        print("🔗 상호작용 피처 생성 중...")
        for a, b in pairs:
            colname = f"{a}__x__{b}"
            for df in (self.train, self.test):
                if a in df.columns and b in df.columns:
                    df[colname] = df[a].astype(str) + "_" + df[b].astype(str)
        print("✅ 상호작용 피처 생성 완료")

    # [build_interactions] end ================================================

    # [vectorize_text_tfidf] start #############################################
    def vectorize_text_tfidf(
        self,
        max_features_name=15000,
        max_features_desc=20000,
        n_components=150,
    ):
        print("🔍 TF-IDF 벡터화 시작 (CPU)...")

        vec_name = TfidfVectorizer(
            max_features=max_features_name,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
            dtype=np.float32,
        )
        Xn_train = vec_name.fit_transform(self.train["name"].astype(str))
        Xn_test = vec_name.transform(self.test["name"].astype(str))

        n_comp_name = max(1, min(n_components, Xn_train.shape[1] - 1))
        svd_name = TruncatedSVD(n_components=n_comp_name, random_state=23)
        name_train_svd = svd_name.fit_transform(Xn_train)
        name_test_svd = svd_name.transform(Xn_test)

        del Xn_train, Xn_test
        gc.collect()

        vec_desc = TfidfVectorizer(
            max_features=max_features_desc,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
            dtype=np.float32,
        )
        Xd_train = vec_desc.fit_transform(self.train["item_description"].astype(str))
        Xd_test = vec_desc.transform(self.test["item_description"].astype(str))

        n_comp_desc = max(1, min(n_components, Xd_train.shape[1] - 1))
        svd_desc = TruncatedSVD(n_components=n_comp_desc, random_state=23)
        desc_train_svd = svd_desc.fit_transform(Xd_train)
        desc_test_svd = svd_desc.transform(Xd_test)

        del Xd_train, Xd_test
        gc.collect()

        train_vec = np.hstack([name_train_svd, desc_train_svd]).astype(np.float32)
        test_vec = np.hstack([name_test_svd, desc_test_svd]).astype(np.float32)

        name_cols = [f"name_{i}" for i in range(n_comp_name)]
        desc_cols = [f"desc_{i}" for i in range(n_comp_desc)]
        cols = name_cols + desc_cols

        self.train_vectorized = pd.DataFrame(train_vec, columns=cols)
        self.test_vectorized = pd.DataFrame(test_vec, columns=cols)

        self._add_categorical_numeric_features()
        print(
            f"✅ TF-IDF 벡터화 완료: train {self.train_vectorized.shape}, "
            f"test {self.test_vectorized.shape}"
        )

    # [vectorize_text_tfidf] end ==============================================

    # [vectorize_text_fasttext] start ##########################################
    def vectorize_text_fasttext(
        self,
        text_columns=("name", "item_description"),
        fasttext_size=100,
        fasttext_window=5,
        fasttext_min_count=2,
        n_components=None,
    ):
        if FastText is None:
            raise RuntimeError("gensim FastText가 설치되어 있지 않습니다.")
        print("🔍 FastText 학습 시작...")

        sentences = []
        for col in text_columns:
            sentences += [
                s.split()
                for s in pd.concat([self.train[col], self.test[col]]).astype(str)
            ]

        ft = FastText(
            sentences,
            vector_size=fasttext_size,
            window=fasttext_window,
            min_count=fasttext_min_count,
            sg=1,
            workers=4,
        )

        def _vec(text):
            words = str(text).split()
            vecs = [ft.wv[w] for w in words if w in ft.wv]
            return np.mean(vecs, axis=0) if vecs else np.zeros(fasttext_size)

        train_feats = []
        test_feats = []

        for col in tqdm(text_columns, desc="FastText Vectorizing"):
            train_feats.append(
                np.vstack(self.train[col].astype(str).apply(_vec))
            )
            test_feats.append(
                np.vstack(self.test[col].astype(str).apply(_vec))
            )

        train_vec = np.hstack(train_feats).astype(np.float32)
        test_vec = np.hstack(test_feats).astype(np.float32)

        if n_components:
            pca = PCA(n_components=n_components, random_state=42)
            train_vec = pca.fit_transform(train_vec)
            test_vec = pca.transform(test_vec)

        dim = train_vec.shape[1] // len(text_columns)
        cols = [f"{col}_ft_{i}" for col in text_columns for i in range(dim)]

        self.train_vectorized = pd.DataFrame(train_vec, columns=cols)
        self.test_vectorized = pd.DataFrame(test_vec, columns=cols)

        self._add_categorical_numeric_features()
        print("✅ FastText 벡터화 완료")

    # [vectorize_text_fasttext] end ===========================================

    # [_add_categorical_numeric_features] start ################################
    def _add_categorical_numeric_features(self):
        print("📌 구조적/수치형 피처를 벡터화 프레임에 결합 중...")

        categorical_features = [
            "main_cat",
            "sub_cat",
            "sub_sub_cat",
            "brand_name",
            "item_condition_id",
            "shipping",
            "item_condition_id__x__shipping",
            "brand_name__x__main_cat",
        ]

        numeric_features = [
            # text stats
            "name_tok_count",
            "name_uniq_tok",
            "name_avg_word_len",
            "name_upper_ratio",
            "name_digit_ratio",
            "name_punct_ratio",
            "item_description_tok_count",
            "item_description_uniq_tok",
            "item_description_avg_word_len",
            "item_description_upper_ratio",
            "item_description_digit_ratio",
            "item_description_punct_ratio",
            "name_len_char",
            "name_len_word",
            "desc_len_char",
            "desc_len_word",
            "desc_name_char_ratio",
            "desc_name_word_ratio",
            "has_brand_in_name",
            "has_brand_in_desc",
            # price-brand-cat
            "brand_name_price_mean",
            "brand_name_price_median",
            "brand_name_price_std",
            "brand_price_z",
            "rare_brand",
            "main_cat_price_mean",
            "main_cat_price_median",
            "main_cat_price_std",
            "rare_main_cat",
            "sub_cat_price_mean",
            "sub_cat_price_median",
            "sub_cat_price_std",
            "rare_sub_cat",
            "sub_sub_cat_price_mean",
            "sub_sub_cat_price_median",
            "sub_sub_cat_price_std",
            "rare_sub_sub_cat",
            "brand_in_name_ratio",
            "brand_in_desc_ratio",
        ]

        for col in categorical_features + numeric_features:
            if col in self.train.columns:
                self.train_vectorized[col] = self.train[col].reset_index(drop=True)
                self.test_vectorized[col] = self.test[col].reset_index(drop=True)

        print(
            f"👍 결합 완료: train {self.train_vectorized.shape}, "
            f"test {self.test_vectorized.shape}"
        )

    # [_add_categorical_numeric_features] end ================================

    # [vectorize_text] start ###################################################
    def vectorize_text(self, method="tfidf", **kwargs):
        m = method.lower()

        # 1) 캐시 먼저 확인
        if self.load_vectorized(m):
            return

        if m == "tfidf":
            self.vectorize_text_tfidf(**kwargs)
        elif m == "fasttext":
            self.vectorize_text_fasttext(**kwargs)
        else:
            raise ValueError("지원하는 method: 'tfidf', 'fasttext'")

        self.save_vectorized(m)

    # [vectorize_text] end =====================================================

    # [save_vectorized] start ##################################################
    def save_vectorized(self, method="tfidf"):
        os.makedirs(self.model_dir, exist_ok=True)
        train_path = os.path.join(self.model_dir, f"vectorized_{method}_train.pkl")
        test_path = os.path.join(self.model_dir, f"vectorized_{method}_test.pkl")
        self.train_vectorized.to_pickle(train_path)
        self.test_vectorized.to_pickle(test_path)
        print(f"💾 벡터화 결과 저장: {train_path}, {test_path}")

    # [save_vectorized] end ====================================================

    # [load_vectorized] start ##################################################
    def load_vectorized(self, method="tfidf"):
        train_path = os.path.join(self.model_dir, f"vectorized_{method}_train.pkl")
        test_path = os.path.join(self.model_dir, f"vectorized_{method}_test.pkl")
        if os.path.exists(train_path) and os.path.exists(test_path):
            self.train_vectorized = pd.read_pickle(train_path)
            self.test_vectorized = pd.read_pickle(test_path)
            print(f"📂 벡터화 데이터 로드: {train_path}, {test_path}")
            return True
        return False

    # [load_vectorized] end ====================================================

    # [setup_pycaret] start ####################################################
    def setup_pycaret(self, session_id=23, fold=3, use_gpu=True, n_jobs=4):
        if setup is None:
            raise RuntimeError("PyCaret이 설치되어 있지 않습니다.")
        if self.train_vectorized is None:
            raise RuntimeError("train_vectorized 가 없습니다. vectorize_text() 먼저 호출하세요.")

        print("🔧 PyCaret setup 시작...")

        categorical_cols = [
            "main_cat",
            "sub_cat",
            "sub_sub_cat",
            "brand_name",
            "item_condition_id",
            "shipping",
            "item_condition_id__x__shipping",
            "brand_name__x__main_cat",
        ]
        existing_categorical = [
            c for c in categorical_cols if c in self.train_vectorized.columns
        ]

        df_for_setup = self.train_vectorized.copy()
        df_for_setup["price"] = self.train["price"].reset_index(drop=True)

        self.setup_result = setup(
            data=df_for_setup,
            target="price",
            session_id=session_id,
            categorical_features=existing_categorical or None,
            normalize=True,
            transformation=False,
            fold_strategy="kfold",
            fold=fold,
            use_gpu=use_gpu,
            n_jobs=n_jobs,
            verbose=True,
            html=False,
        )

        gc.collect()
        print("✅ PyCaret setup 완료 (RMSLE는 별도 계산)")

    # [setup_pycaret] end ======================================================

    # [compute_rmsle_kaggle] start #############################################
    def compute_rmsle_kaggle(self, y_true_log, y_pred_log):
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
        rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
        return float(rmsle)

    # [compute_rmsle_kaggle] end ==============================================

    # [find_best_model] start ###################################################
    def find_best_model(
        self,
        use_kaggle_winners=True,
        candidate_names=None,
    ):
        if create_model is None:
            raise RuntimeError("PyCaret이 설치되어 있지 않습니다.")

        print("\n🤖 모델 탐색 시작... (최종 기준: RMSLE(Kaggle))")

        results = []

        if use_kaggle_winners:
            print("📌 Kaggle winners 기반 후보 모델만 사용합니다.")
            if candidate_names is None:
                candidate_names = ["lightgbm", "xgboost", "catboost", "et", "rf", "ridge"]

            for name in candidate_names:
                try:
                    model_path = os.path.join(self.model_dir, f"candidate_{name}")

                    if os.path.exists(model_path + ".pkl"):
                        print(f"\n📦 candidate 존재 → 로드: {model_path}.pkl")
                        model = load_model(model_path)
                    else:
                        print(f"\n   - 모델 생성: {name}")
                        model = create_model(name)
                        save_model(model, model_path)
                        print(f"📦 후보 모델 저장: {model_path}.pkl")

                    print("📊 RMSLE(Kaggle) 계산 중 (train 전체 예측 기반)...")
                    pred_df = predict_model(model, data=self.train_vectorized.copy())
                    y_true_log = self.train["price"].values
                    y_pred_log = pred_df["prediction_label"].values

                    rmsle_k = self.compute_rmsle_kaggle(y_true_log, y_pred_log)
                    print(f"   → RMSLE_kaggle = {rmsle_k:.6f}")

                    results.append((name, model, rmsle_k))

                except Exception as e:
                    print(f"⚠ 모델 {name} 생성/평가 실패: {e}")

        if not results:
            raise RuntimeError("비교 가능한 모델이 없습니다 (results 비어 있음).")

        results.sort(key=lambda x: x[2])
        best_name, best_model, best_rmsle = results[0]

        print(f"\n🏆 BEST MODEL (RMSLE 기준): {best_name} → {best_rmsle:.6f}")

        self.best_model = best_model
        self.best_model_name = best_name
        self.metrics = {"CV_RMSLE": best_rmsle}
        self.models["candidates"] = results
        self.models["best"] = (best_name, best_model, best_rmsle)

        return self.best_model

    # [find_best_model] end =====================================================

    # [save_best_model] start ##################################################
    def save_best_model(self, model_name=None):
        if self.best_model is None:
            raise ValueError("저장할 모델이 없습니다.")
        if save_model is None:
            raise RuntimeError("PyCaret save_model 을 사용할 수 없습니다.")

        os.makedirs(self.model_dir, exist_ok=True)

        invalid_chars_pattern = r'[<>:"/\\|?*]'
        raw_name = model_name or str(self.best_model).split("(")[0]
        base_name = re.sub(invalid_chars_pattern, "_", raw_name).strip()
        if not base_name:
            base_name = "trained_model"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = os.path.join(self.model_dir, f"{base_name}_{timestamp}")
        latest_path = os.path.join(self.model_dir, f"{base_name}_latest")

        save_model(self.best_model, versioned_path)
        save_model(self.best_model, latest_path)

        print(f"💾 모델 저장 완료: {versioned_path}.pkl, {latest_path}.pkl")

    # [save_best_model] end ====================================================

    # [save_metrics] start #####################################################
    def save_metrics(self, model_name=None, vector_method="tfidf"):
        if self.best_model is None:
            raise ValueError("평가할 모델이 없습니다. best_model이 비어 있습니다.")

        # 1) train/price 확보
        if self.train is None or "price" not in getattr(self.train, "columns", []):
            print("ℹ️ train/price가 메모리에 없습니다.")
            raise RuntimeError("train/price가 없습니다. 전처리를 먼저 수행하세요.")

        # 2) train_vectorized 확보
        if self.train_vectorized is None:
            print(
                f"ℹ️ train_vectorized가 없습니다. "
                f"vectorized_{vector_method}_*.pkl 로드 시도..."
            )
            if not self.load_vectorized(vector_method):
                raise RuntimeError(
                    f"vectorized_{vector_method}_train.pkl 을 찾지 못했습니다."
                )

        if predict_model is None:
            raise RuntimeError("PyCaret predict_model 을 사용할 수 없습니다.")

        print("📊 메트릭 계산 중...")

        pred_df = predict_model(self.best_model, data=self.train_vectorized.copy())

        y_true_log = self.train["price"].values
        y_pred_log = pred_df["prediction_label"].values

        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)

        from sklearn.metrics import (
            r2_score,
            mean_squared_error,
            mean_absolute_error,
        )

        R2 = float(round(r2_score(y_true, y_pred), 6))
        RMSE = float(round(mean_squared_error(y_true, y_pred, squared=False), 6))
        MAE = float(round(mean_absolute_error(y_true, y_pred), 6))
        RMSLE_K = self.compute_rmsle_kaggle(y_true_log, y_pred_log)

        self.metrics = {
            "R2": R2,
            "RMSE": RMSE,
            "MAE": MAE,
            "RMSLE_Kaggle": RMSLE_K,
        }

        invalid_chars_pattern = r'[<>:"/\\|?*]'
        raw_name = model_name or str(self.best_model).split("(")[0]
        base_name = re.sub(invalid_chars_pattern, "_", raw_name).strip()
        if not base_name:
            base_name = "model_metrics"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.results_dir, exist_ok=True)
        file_path = os.path.join(
            self.results_dir, f"{base_name}_metrics_{timestamp}.json"
        )

        with open(file_path, "w", encoding="utf8") as f:
            json.dump(self.metrics, f, indent=4, ensure_ascii=False)

        print(f"💾 Metrics 저장: {file_path}")
        print(
            f"   - R2={R2:.6f}, RMSE={RMSE:.6f}, MAE={MAE:.6f}, "
            f"RMSLE_Kaggle={RMSLE_K:.6f}"
        )

    # [save_metrics] end =======================================================

    # [predict_test] start #####################################################
    def predict_test(self, submission_file="submission.csv"):
        if self.best_model is None:
            raise ValueError("예측할 모델이 없습니다. best_model이 비어 있습니다.")
        if self.test_vectorized is None:
            raise RuntimeError(
                "test_vectorized가 없습니다. vectorize_text() 또는 load_vectorized() 후 실행하세요."
            )
        if predict_model is None:
            raise RuntimeError("PyCaret predict_model 을 사용할 수 없습니다.")

        print("📦 테스트 예측 시작...")
        pred_df = predict_model(self.best_model, data=self.test_vectorized.copy())
        price_pred = np.expm1(pred_df["prediction_label"].values)

        submission = pd.DataFrame(
            {
                "test_id": self.test["test_id"],
                "price": price_pred,
            }
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.results_dir, f"{timestamp}_{submission_file}")
        submission.to_csv(path, index=False)

        print(f"💾 제출 파일 저장: {path}")
        print(
            f"   - 예측 가격 범위: ${price_pred.min():.2f} ~ "
            f"${price_pred.max():.2f}, 평균: ${price_pred.mean():.2f}"
        )
        return submission

    # [predict_test] end =======================================================

    # ===== stage 기반 캐시 관련 메서드들 ======================================

    def _stage_cache_path(self, stage, use_date=False, tag=None):
        parts = [f"stage_{stage}"]
        if tag:
            parts.append(str(tag))
        if use_date:
            parts.append(datetime.now().strftime("%Y%m%d"))
        fname = "_".join(parts) + ".pkl"
        return os.path.join(self.results_dir, fname)

    def _save_stage(self, stage, path=None):
        if path is None:
            path = self._stage_cache_path(stage, use_date=False)
        data = {
            "stage": stage,
            "train": self.train,
            "test": self.test,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"💾 Stage saved: {stage} -> {path}")
            return True
        except Exception as e:
            print(f"⚠️ Stage 저장 실패 ({stage}): {e}")
            return False

    def _load_stage(self, stage, path=None):
        if path is None:
            path = self._stage_cache_path(stage, use_date=False)
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict) or "train" not in data or "test" not in data:
                print(f"⚠️ Stage 파일 형식 이상: {path}")
                return False
            self.train = data["train"]
            self.test = data["test"]
            print(
                f"⚡ Stage loaded: {stage} <- {path} "
                f"(timestamp: {data.get('timestamp')})"
            )
            return True
        except Exception as e:
            print(f"⚠️ Stage 로드 실패 ({stage}): {e}")
            return False

    def _compute_param_tag(self, param_dict):
        if not param_dict:
            return None
        try:
            s = json.dumps(param_dict, sort_keys=True, ensure_ascii=False)
            return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
        except Exception:
            return None

    def preprocess_all_staged(
        self,
        use_cache=True,
        save_cache=True,
        cols=("name", "item_description"),
        undersample_frac=None,
        param_dict=None,
        debug=False,
    ):
        tag = self._compute_param_tag(param_dict)

        stages = [
            ("loaded", "load_data"),
            ("normalized", "normalize_text"),
            ("text_stats", "build_text_stats"),
            ("price_brand_cat", "build_price_brand_cat_features"),
            ("interactions", "build_interactions"),
        ]

        # STAGE 0: loaded
        if use_cache and self._load_stage("loaded"):
            if debug:
                print("Stage 'loaded' 를 캐시에서 로드했습니다.")
        else:
            if debug:
                print("Stage 'loaded' 캐시 없음 → load_data() 실행")
            try:
                u_frac = (
                    param_dict.get("undersample_frac")
                    if isinstance(param_dict, dict)
                    and "undersample_frac" in param_dict
                    else undersample_frac
                )
                self.load_data(undersample_frac=u_frac)
                if save_cache:
                    self._save_stage("loaded")
            except Exception as e:
                print(f"❌ load_data() 실패: {e}")
                raise

        # 나머지 스테이지
        for stage, method_name in stages[1:]:
            if use_cache and self._load_stage(stage):
                if debug:
                    print(f"Stage '{stage}' 를 캐시에서 로드했습니다.")
                continue

            try:
                print(f"\n➡ Running stage: {stage} -> {method_name}()")
                method = getattr(self, method_name)
                if method_name in ("normalize_text", "build_text_stats"):
                    method(cols=cols)
                else:
                    method()
                if save_cache:
                    self._save_stage(stage)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"❌ 스테이지 '{stage}' ({method_name}) 에서 오류 발생: {e}")
                raise

        print("\n✅ preprocess_all_staged 완료")


# ======================================================================
# ========== 외부 유틸 함수들 (full train / ensemble / 리포트) ==========
# ======================================================================

def compare_saved_models_metrics(analyzer, models_dir="../models"):
    """
    ../models 폴더의 candidate_*.pkl 모델들을
    analyzer.train_vectorized 기준으로 다시 평가하고,
    R2/RMSE/MAE/RMSLE 비교 + 그래프 저장
    """
    import glob
    import joblib
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    from datetime import datetime

    candidate_files = glob.glob(os.path.join(models_dir, "candidate_*.pkl"))
    model_files = []
    model_files.extend(candidate_files)

    if not model_files:
        print("❌ candidate_* 모델이 없습니다.")
        return None

    print("📂 사용 대상 모델 파일들:")
    for f in model_files:
        print("   -", f)

    X = analyzer.train_vectorized
    y_true = analyzer.train["price"]  # log1p(price)

    rows = []

    def compute_rmsle(y_true_log, y_pred_log):
        y_true_lin = np.expm1(y_true_log)
        y_pred_lin = np.expm1(y_pred_log)
        return np.sqrt(np.mean((np.log1p(y_pred_lin) - np.log1p(y_true_lin)) ** 2))

    for mf in model_files:
        name = os.path.basename(mf).replace(".pkl", "")
        print(f"\n▶ loading {name} ...")
        model = joblib.load(mf)
        if not hasattr(model, "predict"):
            print(f"⚠ {name}: predict 메서드가 없습니다. → skip")
            continue

        preds = model.predict(X)

        r2 = r2_score(y_true, preds)
        rmse = mean_squared_error(y_true, preds, squared=False)
        mae = mean_absolute_error(y_true, preds)

        try:
            rmsle_val = compute_rmsle(y_true, preds)
        except Exception as e:
            print(f"⚠ {name}: RMSLE 계산 중 오류 → {e}")
            rmsle_val = np.nan

        rows.append({
            "model": name,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "RMSLE": rmsle_val,
        })

    if not rows:
        print("❌ 유효한 모델이 하나도 없습니다.")
        return None

    df = pd.DataFrame(rows)
    print("\n=== Model Metrics ===")
    print(df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("../results", exist_ok=True)
    df_path = f"../results/{ts}_model_metrics.csv"
    df.to_csv(df_path, index=False, encoding="utf-8-sig")
    print(f"💾 metrics CSV 저장 완료: {df_path}")

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="R2")
    plt.title("pycaret 학습모델별 metrics 비교표")
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("../images", exist_ok=True)
    file_path = f"../images/{ts}_model_metrics.png"
    plt.savefig(file_path, dpi=200)
    plt.show()
    print(f"📁 그래프 저장 완료: {file_path}")

    return df


def compare_models_subplots(analyzer, df_metrics, save=True):
    metrics = ["R2", "RMSE", "MAE", "RMSLE"]
    titles = [
        "R2 비교",
        "RMSE 비교",
        "MAE 비교",
        "RMSLE 비교",
    ]

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i, m in enumerate(metrics):
        ax = axes[i]
        sns.barplot(data=df_metrics, x="model", y=m, ax=ax)
        ax.set_title(titles[i])
        ax.tick_params(axis="x", rotation=45)

        for idx, row in df_metrics.iterrows():
            val = row[m]
            if pd.isna(val):
                continue
            ax.text(idx, val + val * 0.01, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if save:
        os.makedirs("../images", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"../images/{ts}_metrics_subplots.png"
        plt.savefig(file_path, dpi=200)
        print(f"📁 subplot 그래프 저장 완료: {file_path}")

    plt.show()


# ------------------------ Full Data 재학습 ------------------------ #

def train_full_best(
    analyzer,
    create_model_kwargs=None,
    save_meta_json=True,
):
    """
    analyzer.best_model_name 에 해당하는 모델을 full data로 재학습
    """
    from pycaret.regression import create_model, finalize_model, save_model
    import time

    if not hasattr(analyzer, "best_model_name"):
        raise AttributeError("best_model_name 이 없습니다. find_best_model() 먼저!")

    best_name = analyzer.best_model_name
    print(f"\n🚀 FULL DATA 재학습 시작 (best_model={best_name})")

    start_ts = datetime.now()
    start_time = time.time()
    print(f"⏱ Start: {start_ts}")

    if create_model_kwargs is None:
        create_model_kwargs = {}
    print(f"⚙ create_model params: {create_model_kwargs}")

    # 진행 표시용
    stop_progress = False

    def progress_timer():
        import time, sys
        while not stop_progress:
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r⏳ FULL 학습 중...  {elapsed:.1f} sec")
            sys.stdout.flush()
            time.sleep(2)

    import threading
    progress_thread = threading.Thread(target=progress_timer, daemon=True)
    progress_thread.start()

    grid = create_model(best_name, **create_model_kwargs)
    full_model = finalize_model(grid)

    end_ts = datetime.now()
    elapsed_sec = (end_ts - start_ts).total_seconds()
    elapsed_min = elapsed_sec / 60

    stop_progress = True

    print(f"\n⏰ End: {end_ts}")
    print(f"🕒 Elapsed: {elapsed_sec:.1f} sec ({elapsed_min:.1f} min)")

    try:
        metric_df = grid.score_grid
        metrics_dict = metric_df.loc["Mean"].to_dict()
    except Exception:
        metrics_dict = {}

    meta = {
        "model_name": best_name,
        "start_time": start_ts.isoformat(),
        "end_time": end_ts.isoformat(),
        "elapsed_sec": elapsed_sec,
        "elapsed_min": elapsed_min,
        "create_model_kwargs": create_model_kwargs,
        "cv_metrics": metrics_dict,
    }

    full_model.meta = meta

    os.makedirs("../models", exist_ok=True)
    fname = f"full_{best_name}_{start_ts.strftime('%Y%m%d_%H%M%S')}"
    save_model(full_model, os.path.join("../models", fname))
    print(f"💾 saved model: ../models/{fname}.pkl")

    if save_meta_json:
        os.makedirs("../results", exist_ok=True)
        meta_path = os.path.join("../results", f"{fname}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        print(f"📝 meta json saved: {meta_path}")

    analyzer.best_full_model = full_model
    analyzer.full_model_path = f"../models/{fname}.pkl"

    return full_model


# ------------------------ full_* 모델 로딩 ------------------------ #

def load_latest_full_models(
    models_dir="../models",
    target_model_names=None,
):
    if target_model_names is None:
        target_model_names = ["lightgbm", "xgboost", "et", "rf"]

    import glob

    pattern = os.path.join(models_dir, "full_*.pkl")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"❌ '{pattern}' 경로에 full 모델이 없습니다.")

    latest = {}  # {model_name: (ts, base_name)}

    for f in files:
        base = os.path.basename(f)              # full_lightgbm_yyyymmdd_hhMMss.pkl
        name_no_ext = os.path.splitext(base)[0] # full_lightgbm_yyyymmdd_hhMMss
        parts = name_no_ext.split("_", 2)
        if len(parts) < 3:
            print(f"⚠ 스킵 (파일명 패턴 불일치): {base}")
            continue

        _, model_name, ts = parts
        if model_name not in target_model_names:
            continue

        prev = latest.get(model_name)
        if (prev is None) or (ts > prev[0]):
            latest[model_name] = (ts, name_no_ext)

    if not latest:
        raise RuntimeError(f"❌ target_model_names={target_model_names} 중 최신 full 모델을 찾지 못했습니다.")

    print("\n🔎 최신 full 모델 선택 결과:")
    results = {}
    for model_name, (ts, base_name) in latest.items():
        print(f"  - {model_name}: {base_name}.pkl (ts={ts})")
        print(f"▶ loading {base_name} ...")
        try:
            m = load_model(os.path.join(models_dir, base_name))
            results[model_name] = m
        except Exception as e:
            print(f"❌ 로드 실패: {base_name}: {e}")

    if len(results) < 2:
        print("⚠ 경고: 2개 미만 로드됨 (앙상블 의미 거의 없음)")

    return results


# ------------------------ Voting Ensemble ------------------------ #

def predict_only_voting_ensemble(
    analyzer,
    target_model_names=("lightgbm", "xgboost", "et", "rf"),
    models_dir="../models",
    results_dir="../results",
    submission_prefix="blend",
):
    from sklearn.ensemble import VotingRegressor

    os.makedirs(results_dir, exist_ok=True)

    # 최신 full 모델 로드
    models = load_latest_full_models(
        models_dir=models_dir,
        target_model_names=list(target_model_names),
    )

    estimators = [(k, v) for k, v in models.items()]
    voting = VotingRegressor(estimators=estimators)

    print("🔄 voting.fit on full train ...")
    voting.fit(analyzer.train_vectorized, analyzer.train["price"])

    # train RMSLE
    y_pred_train = voting.predict(analyzer.train_vectorized)
    rmsle = np.sqrt(
        np.mean(
            (np.log1p(y_pred_train) - np.log1p(analyzer.train["price"])) ** 2
        )
    )
    print(f"✨ blend RMSLE(train): {rmsle:.6f}")

    # test 예측
    y_pred = voting.predict(analyzer.test_vectorized)
    pred_price = np.expm1(y_pred)
    submission = analyzer.test[["test_id"]].copy()
    submission["price"] = pred_price

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(results_dir, f"{ts}_{submission_prefix}.csv")
    submission.to_csv(save_path, index=False)

    print(f"💾 saved: {save_path}")
    print(
        f"  - range ${pred_price.min():.2f} ~ ${pred_price.max():.2f}, "
        f"mean ${pred_price.mean():.2f}"
    )

    return models, voting, rmsle, submission


def evaluate_all_models(
    analyzer,
    models,
    blend_rmsle,
    results_dir="../results",
):
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics = {}
    for name, model in models.items():
        p = model.predict(analyzer.train_vectorized)
        rmsle = np.sqrt(
            np.mean((np.log1p(p) - np.log1p(analyzer.train["price"])) ** 2)
        )
        metrics[name] = rmsle

    metrics["blend"] = blend_rmsle
    df = pd.DataFrame.from_dict(metrics, orient="index", columns=["train_RMSLE"])
    df = df.sort_values("train_RMSLE")

    csv_path = os.path.join(results_dir, f"{ts}_model_train_metrics.csv")
    df.to_csv(csv_path)
    print(f"💾 saved metrics: {csv_path}")

    plt.figure(figsize=(8, 4))
    df["train_RMSLE"].plot(kind="bar")
    plt.ylabel("train RMSLE")
    plt.title("Full Models vs Blend")
    plt.tight_layout()
    png_path = os.path.join(results_dir, f"{ts}_model_train_metrics.png")
    plt.savefig(png_path, dpi=150)
    print(f"📊 saved plot: {png_path}")

    print("\n===== RMSLE ranking =====")
    print(df)

    return df


def weighted_voting_ensemble(
    analyzer,
    models: dict,
    weights=None,
    name_prefix="blend",
):
    from sklearn.ensemble import VotingRegressor

    X = analyzer.train_vectorized
    y = analyzer.train["price"]

    est = [(name, m) for name, m in models.items()]
    voting = VotingRegressor(estimators=est, weights=weights)

    print("\n🔄 voting.fit on full train ...")
    voting.fit(X, y)

    pred_train = voting.predict(X)
    rmsle = np.sqrt(np.mean((np.log1p(y) - np.log1p(pred_train)) ** 2))

    df_compare = pd.DataFrame(
        {"train_RMSLE": [rmsle]}, index=[name_prefix]
    )

    print("\n===== RMSLE of blended =====")
    print(df_compare)

    preds_test = voting.predict(analyzer.test_vectorized)
    submission = analyzer.test[["test_id"]].copy()
    submission["price"] = np.expm1(preds_test)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"../results/{ts}_{name_prefix}.csv"
    submission.to_csv(csv_path, index=False)

    print(f"💾 saved: {csv_path}")
    print(
        f"  - range ${submission.price.min():.2f} ~ "
        f"${submission.price.max():.2f}, mean ${submission.price.mean():.2f}"
    )

    return voting, rmsle, submission


def get_rmsle_comparison(models, analyzer, blend_rmsle):
    results = {}
    y_true = analyzer.train["price"]
    for name, model in models.items():
        pred = model.predict(analyzer.train_vectorized)
        rmsle_val = np.sqrt(
            np.mean((np.log1p(pred) - np.log1p(y_true)) ** 2)
        )
        results[name] = rmsle_val

    results["blend"] = blend_rmsle
    return pd.DataFrame.from_dict(results, orient="index", columns=["RMSLE"])


# ------------------------ LightGBM 중요도/SHAP ------------------------ #

def plot_lightgbm_importance(analyzer, models, top_k=40, save_dir="../results"):
    lgbm = models["lightgbm"]
    est = lgbm.named_steps.get("actual_estimator", None)

    if est is None or not hasattr(est, "feature_importances_"):
        print("⚠ LightGBM 중요도 정보를 찾을 수 없습니다.")
        return None

    features = getattr(est, "feature_name_", None)
    if features is None or len(features) != len(est.feature_importances_):
        features = getattr(est, "feature_names_in_", None)

    if features is None or len(features) != len(est.feature_importances_):
        n = min(len(analyzer.train_vectorized.columns), len(est.feature_importances_))
        features = list(analyzer.train_vectorized.columns)[:n]
        importances = est.feature_importances_[:n]
    else:
        importances = est.feature_importances_

    fi = pd.DataFrame(
        {"feature": features, "importance": importances}
    ).sort_values("importance", ascending=False)

    print("\n=== Top features (LightGBM) ===")
    print(fi.head(20))

    plt.figure(figsize=(6, 12))
    fi.head(top_k).plot(
        kind="barh",
        x="feature",
        y="importance",
        title="LightGBM Feature Importance",
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"{ts}_lightgbm_feature_importance.png")
    plt.savefig(fname, dpi=200)
    plt.show()

    print(f"💾 saved: {fname}")
    return fi


def run_shap_on_pycaret_lightgbm():
    """
    PyCaret setup 이후,
    get_config('X_train') / get_config('y_train') 기반으로
    별도 LGBMRegressor 학습 후 SHAP summary plot 생성
    """
    if get_config is None:
        print("PyCaret get_config 사용 불가.")
        return

    import shap
    from lightgbm import LGBMRegressor

    X_train_processed = get_config("X_train").copy()
    y_train_log = get_config("y_train").copy()

    print("X_train_processed shape:", X_train_processed.shape)
    print("y_train_log shape      :", y_train_log.shape)

    lgbm_shap = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=23,
        n_jobs=-1,
    )

    lgbm_shap.fit(X_train_processed, y_train_log)
    print("✅ SHAP용 LightGBM 학습 완료")

    n_sample = 3000
    if len(X_train_processed) > n_sample:
        X_shap_sample = X_train_processed.sample(n_sample, random_state=23)
    else:
        X_shap_sample = X_train_processed

    feature_names = list(X_train_processed.columns)
    print("X_shap_sample shape:", X_shap_sample.shape)

    explainer = shap.TreeExplainer(lgbm_shap)
    shap_values = explainer.shap_values(X_shap_sample)

    shap.initjs()
    shap.summary_plot(
        shap_values,
        X_shap_sample,
        feature_names=feature_names,
        max_display=30,
    )


# ------------------------ Submission 분포 요약 ------------------------ #

def summarize_submissions(results_dir="../results"):
    """
    Best submission을 자동 선택하려는 이상한 점수 대신,
    단순히 submission 파일들의 price 분포를 요약해서 sanity check만 수행.
    """
    import glob

    pattern = os.path.join(results_dir, "*submission*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print("❌ submission 파일을 찾지 못했습니다.")
        return None

    rows = []
    for f in files:
        df = pd.read_csv(f)
        if "price" not in df.columns:
            continue
        s = df["price"]
        rows.append(
            {
                "file": os.path.basename(f),
                "mean_price": s.mean(),
                "std_price": s.std(),
                "min_price": s.min(),
                "max_price": s.max(),
            }
        )

    summary = pd.DataFrame(rows).sort_values("mean_price")
    print("\n=== Submission price summary ===")
    print(summary)
    return summary


# ======================================================================
# 끝. 이 파일을 기준으로 import 해서 사용하면 됨.
# ======================================================================
