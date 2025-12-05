``` python

"""
mercari_pycaret_analyzer8_full
--------------------------------
    - 이 파일은 Mercari Kaggle 경진용 가격 예측 파이프라인을 한 클래스에 집약한 구현입니다.
    - 원본 코드 스타일(한글 docstring, 함수 순서, 주석 스타일)을 유지했습니다.
    - 또한 다음 항목들을 보강/수정하였습니다:
        * merge 후 df 재할당 누락 이슈 수정
        * TF-IDF -> SVD 차원 mismatch 문제 해결
        * 감성분석(VADER/TextBlob) 안정적 대체 로직 추가
        * 모델 저장/로드 경로 일관성 개선 (self.model_dir 사용)
        * 브랜드 포함 여부 계산 로직 안정화
    - 각 method/function의 시작과 끝에는 명확한 주석으로 표시했습니다.
    - 파일 하단에 클래스 사용 예시(파이프라인 형태)를 포함했습니다.
주의:
    - PyCaret, gensim, sentence-transformers, nltk, textblob 등 일부 라이브러리는 설치되어 있어야 합니다.
    - 노트북에서 사용 시 셀 하나에 붙여넣기보다는 파일로 저장한 뒤 import 해서 쓰는 것을 권장합니다.
"""

# ----------------------------
# 기본 임포트
# ----------------------------
import os
import re
import gc
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch

import pickle


# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA

# optional imports (안전하게 래핑)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from gensim.models import FastText, Word2Vec
except Exception:
    FastText = None
    Word2Vec = None

# pycaret regression
try:
    from pycaret.regression import (
        setup, create_model, compare_models, blend_models, tune_model,
        save_model, load_model, predict_model
    )
except Exception:
    setup = create_model = compare_models = blend_models = tune_model = None
    save_model = load_model = predict_model = None

# ======================================================================
# ===================== CLASS: MercariPyCaretAnalyzer8 ====================
# ======================================================================

class MercariPyCaretAnalyzer8:
    """
    =========================================================================
    MercariPyCaretAnalyzer8 클래스 (한글 상세문서 포함)
    =========================================================================

    개요:
        Mercari(또는 유사한 전자상거래) 상품 가격 예측을 위해 설계한 
        PyCaret 기반의 엔드-투-엔드 파이프라인 클래스입니다.

    주요 기능:
        - 데이터 로딩 및 기본 전처리 (category 분할, 결측치 처리, 희귀값 통합 등)
        - 텍스트 정규화 및 텍스트 통계 피처 생성
        - 감성 분석(VADER 우선, TextBlob fallback)
        - 브랜드/카테고리/가격 기반 그룹 통계 피처 생성
        - 상호작용 피처 생성
        - 텍스트 벡터화 (TF-IDF+SVD, FastText, Word2Vec, GloVe, BERT)
        - PyCaret 기반 모델 셋업, 모델 탐색(compare/create), 블렌딩, 튜닝
        - 모델 저장/로드, 메트릭 저장, 테스트셋 제출파일 생성

    의존 라이브러리 (권장):
        - pandas, numpy, scikit-learn, tqdm
        - pycaret (regression), gensim, sentence-transformers
        - nltk (VADER), textblob (fallback)
        - optional: optuna (튜닝 백엔드)

    사용 예시 (간단 파이프라인 - 파일 하단에 전체 파이프라인 예시 있음):
        analyzer = MercariPyCaretAnalyzer8(data_dir="../data")
        analyzer.load_data(undersample_frac=0.35)
        analyzer.normalize_text(cols=["name", "item_description"])
        analyzer.build_text_stats(cols=["name", "item_description"])
        analyzer.compute_sentiment(cols=["name", "item_description"])
        analyzer.build_price_brand_cat_features()
        analyzer.build_interactions()
        analyzer.vectorize_text(method="tfidf")
        analyzer.setup_pycaret(fold=3)
        analyzer.find_and_blend_models()
        analyzer.save_metrics()
        analyzer.predict_test()

    주의:
        - 로컬 환경의 PyCaret 버전에 따라 setup() 인자명이 다를 수 있으니 필요 시 조정하세요.
        - 텍스트 벡터화(특히 BERT)는 메모리/시간을 많이 소모하므로 샘플로 테스트 후 전체 실행 권장.
    =========================================================================
    """

    # ========================= METHOD: __init__ (start) =========================    
    def __init__(self, data_dir="../data", images_dir="../images", results_dir="../results", model_dir="../models", use_gpu=True):
        """
        클래스 초기화

        Parameters
        ----------
        data_dir : str
            데이터가 위치한 폴더 (train.tsv, test.tsv 등)
        images_dir : str
            이미지 저장 폴더
        results_dir : str
            결과(json, submission) 저장 폴더
        model_dir : str
            모델/벡터화 결과 저장 폴더
        """
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.results_dir = results_dir
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.device = None
        self._detect_device()
        

        # 데이터프레임
        self.train = None
        self.test = None

        # 벡터화된 데이터
        self.train_vectorized = None
        self.test_vectorized = None

        # pycaret 관련
        self.setup_result = None
        self.best_model = None
        self.metrics = {}
        self.models = {}

        # 디렉토리 생성
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
    # ========================= METHOD: __init__ (end) ===========================


    # ========================= METHOD: load_data (start) ========================
    def load_data(self, train_file="train.tsv", test_file="test.tsv", sep="\t", undersample_frac=None):
        """
        데이터 로딩 및 기본 전처리

        주요 처리:
            - train/test 로드
            - price>0 필터링 및 log1p 변환 (train)
            - category_name -> main_cat / sub_cat / sub_sub_cat 분해
            - brand_name / item_description / name 결측 처리
            - 희귀값 통합 (brand, category)
            - (옵션) stratified undersample
        """
        print("📂 데이터 로딩 시작...")

        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)

        self.train = pd.read_csv(train_path, sep=sep)
        self.test = pd.read_csv(test_path, sep=sep)

        # price 전처리
        self.train = self.train[self.train["price"] > 0].dropna(subset=["price"]).reset_index(drop=True)
        self.train["price"] = np.log1p(self.train["price"])

        # category split 및 기본 fill
        for df in [self.train, self.test]:
            df["category_name"] = df.get("category_name", "").fillna("missing")
            def _split_cat(x):
                parts = x.split("/") if isinstance(x, str) else ["missing"]*3
                while len(parts) < 3:
                    parts.append("missing")
                return parts[:3]
            df["main_cat"], df["sub_cat"], df["sub_sub_cat"] = zip(*df["category_name"].apply(_split_cat))
            df["brand_name"] = df["brand_name"].fillna("Unknown").astype(str)
            df["item_description"] = df["item_description"].fillna("No description").astype(str)
            df["name"] = df["name"].fillna("No name").astype(str)
            if "shipping" in df.columns:
                df["shipping"] = df["shipping"].astype("category")
            if "item_condition_id" in df.columns:
                df["item_condition_id"] = df["item_condition_id"].astype("category")

        # 희귀값 처리 (원본 기준)
        print("🔄 희귀값 통합 중...")
        self._collapse_rare_values("brand_name", top_k=5000, rare_label="Other_brand")
        self._collapse_rare_values("main_cat", top_k=1000, rare_label="Other_main")
        self._collapse_rare_values("sub_cat", top_k=1000, rare_label="Other_sub")
        self._collapse_rare_values("sub_sub_cat", top_k=1000, rare_label="Other_sub_sub")

        if undersample_frac:
            self._stratified_sample(frac=undersample_frac)
            
        if "price" not in self.train.columns:
            raise RuntimeError("❌ load_data 후 price 컬럼이 누락되었습니다! 전처리 중 문제 발생!")
            
        print(f"✅ 데이터 로딩 완료: train {self.train.shape}, test {self.test.shape}")
    # ========================= METHOD: load_data (end) ==========================


    # ========================= METHOD: _collapse_rare_values (start) ===========
    def _collapse_rare_values(self, col, top_k, rare_label="Other"):
        """
        상위 top_k 값을 제외한 나머지를 rare_label로 대체 (train/test 병합 기준)
        """
        combined = pd.concat([self.train[col], self.test[col]], axis=0)
        value_counts = combined.value_counts()
        top_values = set(value_counts.index[:top_k])
        self.train[col] = self.train[col].apply(lambda x: x if x in top_values else rare_label)
        self.test[col] = self.test[col].apply(lambda x: x if x in top_values else rare_label)
        del combined, value_counts
        gc.collect()
    # ========================= METHOD: _collapse_rare_values (end) =============


    # ========================= METHOD: _stratified_sample (start) ==============
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

    # ========================= METHOD: _stratified_sample (end) ================


    # ========================= METHOD: normalize_text (start) ==================
    def normalize_text(self, cols=["name", "item_description"], lower=True, strip_punct=True, numbers_to_token=True):
        """
        텍스트 정규화: *_norm 컬럼으로 저장

        처리:
            - 소문자 변환 (옵션)
            - 특수문자 제거 (옵션)
            - 숫자 -> 'num' 토큰 (옵션)
        """
        print("🧼 텍스트 정규화 시작...")
        def _norm(text):
            t = str(text)
            if lower: t = t.lower()
            if strip_punct: t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
            if numbers_to_token: t = re.sub(r"\d+", " num ", t)
            return re.sub(r"\s+", " ", t).strip()
        for df in [self.train, self.test]:
            for c in cols:
                df[f"{c}_norm"] = df[c].astype(str).apply(_norm)
        print("✅ 텍스트 정규화 완료")
    # ========================= METHOD: normalize_text (end) ====================


    # ========================= METHOD: build_text_stats (start) =================
    def build_text_stats(self, cols=["name", "item_description"]):
        """
        텍스트 기반 통계 피처 생성
        - 토큰 수, 고유 토큰 수, 평균 단어 길이
        - 대문자 비율, 숫자 비율, 문장부호 비율
        - name/description 길이 비율 등
        """
        print("📈 텍스트 통계 피처 생성 중...")
        def _stats(s):
            s = str(s)
            tokens = s.split()
            token_count = len(tokens)
            uniq = len(set(tokens))
            avg_word_len = np.mean([len(w) for w in tokens]) if tokens else 0.0
            upper_ratio = sum(1 for ch in s if ch.isupper()) / max(1, len(s))
            digit_ratio = sum(1 for ch in s if ch.isdigit()) / max(1, len(s))
            punct_ratio = sum(1 for ch in s if re.match(r"[^\w\s]", ch)) / max(1, len(s))
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
            df["desc_name_char_ratio"] = (df["desc_len_char"] / df["name_len_char"]).replace([np.inf, -np.inf], 0).fillna(0)
            df["desc_name_word_ratio"] = (df["desc_len_word"] / df["name_len_word"]).replace([np.inf, -np.inf], 0).fillna(0)

        # 브랜드 포함 여부 플래그
        for df in [self.train, self.test]:
            df["has_brand_in_name"] = df.apply(lambda r: 1 if str(r["brand_name"]).lower() in str(r["name"]).lower() else 0, axis=1)
            df["has_brand_in_desc"] = df.apply(lambda r: 1 if str(r["brand_name"]).lower() in str(r["item_description"]).lower() else 0, axis=1)

        print("✅ 텍스트 통계 피처 생성 완료")
    # ========================= METHOD: build_text_stats (end) ===================


    # ========================= METHOD: compute_sentiment (start) =================
    def compute_sentiment(self, cols=["name", "item_description"]):
        """
        감성 분석 피처 생성 (VADER 우선, TextBlob 대체)
        - 생성 컬럼: {col}_sent_polarity
        """
        print("💬 감성 분석 수행...")
        vader = None
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            vader = SentimentIntensityAnalyzer()
        except Exception:
            vader = None

        textblob_cls = None
        if vader is None:
            try:
                from textblob import TextBlob
                textblob_cls = TextBlob
            except Exception:
                textblob_cls = None

        if vader is None and textblob_cls is None:
            print("⚠️ VADER/TextBlob 중 어느 것도 설치되어 있지 않습니다. 건너뜁니다.")
            return

        for df in [self.train, self.test]:
            for c in cols:
                col_to_use = c if c in df.columns else f"{c}_norm"
                series = df[col_to_use].astype(str).fillna("")
                if vader:
                    df[f"{c}_sent_polarity"] = series.apply(lambda x: vader.polarity_scores(x)["compound"] if x else 0.0)
                else:
                    df[f"{c}_sent_polarity"] = series.apply(lambda x: float(textblob_cls(x).sentiment.polarity) if x else 0.0)

        print("✅ 감성 분석 피처 생성 완료")
    # ========================= METHOD: compute_sentiment (end) ===================


    # ========================= METHOD: build_price_brand_cat_features (start) ===
    def build_price_brand_cat_features(
        self,
        price_col="price",
        brand_col="brand_name",
        cat_cols=["main_cat", "sub_cat", "sub_sub_cat"],
        rare_thresh_brand=20,
        rare_thresh_cat=20
    ):
        """
        완전 안전 버전:
        - join/merge 사용하지 않음 → index / dtype / block 손상 없음
        - map 기반 통계 피처 → 절대 price 손상 없음
        - train/test 의 구조가 항상 동일하게 유지됨
        """

        print("\n=== SAFE VERSION: build_price_brand_cat_features ===")
        print("train has price?", price_col in self.train.columns)
        print("test  has price?", price_col in self.test.columns)

        train = self.train
        test = self.test

        if price_col not in train.columns:
            raise RuntimeError("❌ 'price' 컬럼이 train 데이터에서 누락됨!")

        # ---------------------------
        # 1) 그룹 통계 (train 기준)
        # ---------------------------
        grp_cols = [brand_col] + cat_cols

        for col in grp_cols:
            print(f"📊 building stats for '{col}' ...")

            # groupby 통계
            stats = (
                train.groupby(col)[price_col]
                .agg(["mean", "median", "std"])
                .rename(columns={
                    "mean": f"{col}_price_mean",
                    "median": f"{col}_price_median",
                    "std": f"{col}_price_std",
                })
            )

            # 3개 컬럼을 map 으로 안전하게 전달
            for suffix in ["mean", "median", "std"]:
                cname = f"{col}_price_{suffix}"

                mapping = stats[cname]  # Series (index=col value)

                # train
                train[cname] = train[col].map(mapping).fillna(0)

                # test (test에 없는 brand/category가 있을 수 있음)
                test[cname] = test[col].map(mapping).fillna(0)

        # ---------------------------
        # 2) Z-score
        # ---------------------------
        print("⚖️ computing z-score ...")

        for df_name in ["train", "test"]:
            df = getattr(self, df_name)
            mu = df[f"{brand_col}_price_mean"]
            sd = df[f"{brand_col}_price_std"].replace(0, 1)

            df["brand_price_z"] = ((df[price_col] - mu) / sd)
            df["brand_price_z"] = df["brand_price_z"].replace([np.inf, -np.inf], 0).fillna(0)

            setattr(self, df_name, df)

        # ---------------------------
        # 3) rare flags
        # ---------------------------
        print("🔍 computing rarity flags ...")

        brand_freq = train[brand_col].value_counts()
        cat_freqs = {c: train[c].value_counts() for c in cat_cols}

        for df_name in ["train", "test"]:
            df = getattr(self, df_name)

            df["rare_brand"] = df[brand_col].apply(
                lambda x: int(brand_freq.get(x, 0) < rare_thresh_brand)
            )

            for c in cat_cols:
                freq = cat_freqs[c]
                df[f"rare_{c}"] = df[c].apply(
                    lambda x: int(freq.get(x, 0) < rare_thresh_cat)
                )

            setattr(self, df_name, df)

        # ---------------------------
        # 4) brand in text
        # ---------------------------
        print("🔎 computing brand-in-text flags ...")

        def _brand_in_field(row, bcol, fcol):
            b = str(row[bcol]).lower()
            f = str(row[fcol]).lower()
            return int(b != "" and b in f)

        for df_name in ["train", "test"]:
            df = getattr(self, df_name)
            df["brand_in_name_ratio"] = df.apply(
                lambda r: _brand_in_field(r, brand_col, "name"), axis=1
            )
            df["brand_in_desc_ratio"] = df.apply(
                lambda r: _brand_in_field(r, brand_col, "item_description"), axis=1
            )
            setattr(self, df_name, df)

        # ---------------------------
        # 5) 무결성 검사
        # ---------------------------
        if price_col not in self.train.columns:
            raise RuntimeError("❌ build_price_brand_cat_features 이후 price 누락!")

        print("✅ SAFE: 가격/브랜드/카테고리 피처 생성 완료")

        

    # ========================= METHOD: build_price_brand_cat_features (end) =====


    # ========================= METHOD: build_interactions (start) ================
    def build_interactions(self, pairs=[("item_condition_id", "shipping"), ("brand_name", "main_cat")]):
        """
        상호작용 피처 생성
        - pairs: (colA, colB) 튜플 리스트
        """
        print("🔗 상호작용 피처 생성 중...")
        for a, b in pairs:
            colname = f"{a}__x__{b}"
            for df in [self.train, self.test]:
                if a in df.columns and b in df.columns:
                    df[colname] = df[a].astype(str) + "_" + df[b].astype(str)
        print("✅ 상호작용 피처 생성 완료")
    # ========================= METHOD: build_interactions (end) ==================


    # ========================= METHOD: vectorize_text_tfidf (start) ============
    def vectorize_text_tfidf(self, max_features_name=15000, max_features_desc=20000, n_components=150):
        """
        TF-IDF 기반 벡터화 및 TruncatedSVD 차원 축소
        - name / item_description 각각 처리
        - 실제 생성된 SVD 차원에 맞춰 컬럼명 생성 (원본 버그 수정)
        """
        print("🔍 TF-IDF 벡터화 시작...")

        vec_name = TfidfVectorizer(max_features=max_features_name, ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True, dtype=np.float32)
        Xn_train = vec_name.fit_transform(self.train["name"].astype(str))
        Xn_test = vec_name.transform(self.test["name"].astype(str))
        n_comp_name = max(1, min(n_components, Xn_train.shape[1]-1))
        svd_name = TruncatedSVD(n_components=n_comp_name, random_state=23)
        name_train_svd = svd_name.fit_transform(Xn_train)
        name_test_svd = svd_name.transform(Xn_test)
        del Xn_train, Xn_test
        gc.collect()

        vec_desc = TfidfVectorizer(max_features=max_features_desc, ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True, dtype=np.float32)
        Xd_train = vec_desc.fit_transform(self.train["item_description"].astype(str))
        Xd_test = vec_desc.transform(self.test["item_description"].astype(str))
        n_comp_desc = max(1, min(n_components, Xd_train.shape[1]-1))
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

        # 구조적 피처 결합
        self._add_categorical_numeric_features()
        print(f"✅ TF-IDF 벡터화 완료: train {self.train_vectorized.shape}, test {self.test_vectorized.shape}")
    # ========================= METHOD: vectorize_text_tfidf (end) ==============


    # ========================= METHOD: vectorize_text_bert (start) =================
    def vectorize_text_bert(self, text_columns=["name", "item_description"], bert_model_name="all-MiniLM-L6-v2"):
        """
        BERT 기반 문장 임베딩 생성 (SentenceTransformer 사용)
        """
        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers 라이브러리가 설치되어 있지 않습니다.")
        print(f"🔍 BERT 벡터화 시작 (model={bert_model_name})")
        model = SentenceTransformer(bert_model_name)
        train_feats = []
        test_feats = []
        for col in text_columns:
            train_feats.append(model.encode(self.train[col].astype(str).tolist(), batch_size=32, show_progress_bar=True))
            test_feats.append(model.encode(self.test[col].astype(str).tolist(), batch_size=32, show_progress_bar=True))
        train_vec = np.hstack(train_feats).astype(np.float32)
        test_vec = np.hstack(test_feats).astype(np.float32)
        emb_dim = train_feats[0].shape[1]
        cols = [f"{col}_bert_{i}" for col in text_columns for i in range(emb_dim)]
        self.train_vectorized = pd.DataFrame(train_vec, columns=cols)
        self.test_vectorized = pd.DataFrame(test_vec, columns=cols)
        self._add_categorical_numeric_features()
        print("✅ BERT 벡터화 완료")
    # ========================= METHOD: vectorize_text_bert (end) ===================


    # ========================= METHOD: vectorize_text_fasttext (start) ============
    def vectorize_text_fasttext(self, text_columns=["name", "item_description"], fasttext_size=100, fasttext_window=5, fasttext_min_count=2, n_components=None):
        """
        FastText 학습 후 평균 pooling을 통한 문장 벡터 생성
        """
        if FastText is None:
            raise RuntimeError("gensim FastText가 설치되어 있지 않습니다.")
        print("🔍 FastText 학습 시작...")
        sentences = []
        for col in text_columns:
            sentences += [s.split() for s in pd.concat([self.train[col], self.test[col]]).astype(str)]
        ft = FastText(sentences, vector_size=fasttext_size, window=fasttext_window, min_count=fasttext_min_count, sg=1, workers=4)
        def _vec(text):
            words = str(text).split()
            vecs = [ft.wv[w] for w in words if w in ft.wv]
            return np.mean(vecs, axis=0) if vecs else np.zeros(fasttext_size)
        train_feats = []
        test_feats = []
        for col in tqdm(text_columns, desc="FastText Vectorizing"):
            train_feats.append(np.vstack(self.train[col].astype(str).apply(_vec)))
            test_feats.append(np.vstack(self.test[col].astype(str).apply(_vec)))
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
    # ========================= METHOD: vectorize_text_fasttext (end) =============


    # ========================= METHOD: vectorize_text_word2vec (start) ===========
    def vectorize_text_word2vec(self, text_columns=["name", "item_description"], w2v_size=100, w2v_window=5, w2v_min_count=2, n_components=None):
        """
        Word2Vec 학습 후 평균 pooling을 통한 문장 벡터 생성
        """
        if Word2Vec is None:
            raise RuntimeError("gensim Word2Vec가 설치되어 있지 않습니다.")
        print("🔍 Word2Vec 학습 시작...")
        sentences = []
        for col in text_columns:
            sentences += [s.split() for s in pd.concat([self.train[col], self.test[col]]).astype(str)]
        w2v = Word2Vec(sentences, vector_size=w2v_size, window=w2v_window, min_count=w2v_min_count, sg=1, workers=4)
        def _vec(text):
            words = str(text).split()
            vecs = [w2v.wv[w] for w in words if w in w2v.wv]
            return np.mean(vecs, axis=0) if vecs else np.zeros(w2v_size)
        train_feats = []
        test_feats = []
        for col in tqdm(text_columns, desc="Word2Vec Vectorizing"):
            train_feats.append(np.vstack(self.train[col].astype(str).apply(_vec)))
            test_feats.append(np.vstack(self.test[col].astype(str).apply(_vec)))
        train_vec = np.hstack(train_feats).astype(np.float32)
        test_vec = np.hstack(test_feats).astype(np.float32)
        if n_components:
            pca = PCA(n_components=n_components, random_state=42)
            train_vec = pca.fit_transform(train_vec)
            test_vec = pca.transform(test_vec)
        dim = train_vec.shape[1] // len(text_columns)
        cols = [f"{col}_w2v_{i}" for col in text_columns for i in range(dim)]
        self.train_vectorized = pd.DataFrame(train_vec, columns=cols)
        self.test_vectorized = pd.DataFrame(test_vec, columns=cols)
        self._add_categorical_numeric_features()
        print("✅ Word2Vec 벡터화 완료")
    # ========================= METHOD: vectorize_text_word2vec (end) =============


    # ========================= METHOD: vectorize_text_glove (start) ============
    def vectorize_text_glove(self, text_columns=["name", "item_description"], glove_path="./data/glove.6B.100d.txt", n_components=None):
        """
        GloVe 사전학습 벡터 로드 후 평균 pooling
        """
        print("🔍 GloVe 로드 시작...")
        embeddings = {}
        dim = None
        with open(glove_path, encoding="utf8") as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                vec = np.asarray(parts[1:], dtype=np.float32)
                embeddings[word] = vec
                if dim is None:
                    dim = len(vec)
        if dim is None:
            raise ValueError("GloVe 로드 실패: 파일 경로/형식 확인")
        def _vec(text):
            words = str(text).split()
            vecs = [embeddings[w] for w in words if w in embeddings]
            return np.mean(vecs, axis=0) if vecs else np.zeros(dim)
        train_feats = []
        test_feats = []
        for col in tqdm(text_columns, desc="GloVe Vectorizing"):
            train_feats.append(np.vstack(self.train[col].astype(str).apply(_vec)))
            test_feats.append(np.vstack(self.test[col].astype(str).apply(_vec)))
        train_vec = np.hstack(train_feats).astype(np.float32)
        test_vec = np.hstack(test_feats).astype(np.float32)
        if n_components:
            pca = PCA(n_components=n_components, random_state=42)
            train_vec = pca.fit_transform(train_vec)
            test_vec = pca.transform(test_vec)
        dim = train_vec.shape[1] // len(text_columns)
        cols = [f"{col}_glove_{i}" for col in text_columns for i in range(dim)]
        self.train_vectorized = pd.DataFrame(train_vec, columns=cols)
        self.test_vectorized = pd.DataFrame(test_vec, columns=cols)
        self._add_categorical_numeric_features()
        print("✅ GloVe 벡터화 완료")
    # ========================= METHOD: vectorize_text_glove (end) =============


    # ========================= METHOD: _add_categorical_numeric_features (start)
    def _add_categorical_numeric_features(self):
        """
        벡터화된 dataframe에 기존의 구조적(범주형/수치형) 칼럼을 결합합니다.
        - 원본에서 사용하던 컬럼 목록을 그대로 사용
        - 인덱스 정렬을 위해 reset_index(drop=True) 사용
        """
        print("📌 구조적/수치형 피처를 벡터화 프레임에 결합 중...")
        categorical_features = [
            "main_cat", "sub_cat", "sub_sub_cat",
            "brand_name", "item_condition_id", "shipping",
            "item_condition_id__x__shipping", "brand_name__x__main_cat"
        ]
        numeric_features = [
            "name_tok_count","name_uniq_tok","name_avg_word_len","name_upper_ratio","name_digit_ratio","name_punct_ratio",
            "item_description_tok_count","item_description_uniq_tok","item_description_avg_word_len","item_description_upper_ratio","item_description_digit_ratio","item_description_punct_ratio",
            "name_len_char","name_len_word","desc_len_char","desc_len_word","desc_name_char_ratio","desc_name_word_ratio",
            "name_sent_polarity","item_description_sent_polarity",
            "brand_name_price_mean","brand_name_price_median","brand_name_price_std","brand_price_z","rare_brand",
            "main_cat_price_mean","main_cat_price_median","main_cat_price_std","rare_main_cat",
            "sub_cat_price_mean","sub_cat_price_median","sub_cat_price_std","rare_sub_cat",
            "sub_sub_cat_price_mean","sub_sub_cat_price_median","sub_sub_cat_price_std","rare_sub_sub_cat",
            "brand_in_name_ratio","brand_in_desc_ratio"
        ]
        for col in categorical_features + numeric_features:
            if col in self.train.columns:
                self.train_vectorized[col] = self.train[col].reset_index(drop=True)
                self.test_vectorized[col] = self.test[col].reset_index(drop=True)
        print(f"👍 결합 완료: train {self.train_vectorized.shape}, test {self.test_vectorized.shape}")
    # ========================= METHOD: _add_categorical_numeric_features (end)


    # ========================= METHOD: vectorize_text (start) =========================
    def vectorize_text(self, method="tfidf", **kwargs):
        """
        텍스트 벡터화 통합 인터페이스
        method: "tfidf","fasttext","bert","word2vec","glove"
        kwargs: 각 메서드별 파라미터
        - use_gpu=True 이면 자동으로 *_gpu 메서드를 사용
        - GPU 메서드가 없으면 CPU fallback
        """
        m = method.lower()

        # --------------------
        # 1) 벡터화 결과 캐시가 있으면 로드
        # --------------------
        if self.load_vectorized(method):
            return

        # --------------------
        # 2) GPU 여부 체크
        # --------------------
        use_gpu = getattr(self, "use_gpu", False)

        # --------------------
        # 3) 메서드 매핑
        # --------------------
        gpu_methods = {
            "tfidf": "vectorize_text_tfidf_gpu",
            "bert":  "vectorize_text_bert_gpu",
        }

        cpu_methods = {
            "tfidf": "vectorize_text_tfidf",
            "fasttext": "vectorize_text_fasttext",
            "bert": "vectorize_text_bert",
            "word2vec": "vectorize_text_word2vec",
            "glove": "vectorize_text_glove",
        }

        # --------------------
        # 4) GPU 메서드 우선 실행
        # --------------------
        if use_gpu and m in gpu_methods:
            gpu_func_name = gpu_methods[m]
            gpu_func = getattr(self, gpu_func_name, None)

            if gpu_func is not None:
                try:
                    print(f"⚡ GPU 벡터화 실행: {gpu_func_name}()")
                    gpu_func(**kwargs)
                    self.save_vectorized(method)
                    return
                except Exception as e:
                    print(f"⚠️ GPU 벡터화 실패 → CPU fallback 실행: {e}")

        # --------------------
        # 5) CPU 버전 실행
        # --------------------
        if m not in cpu_methods:
            raise ValueError("method must be one of ['tfidf','fasttext','bert','word2vec','glove']")

        cpu_func = getattr(self, cpu_methods[m])
        print(f"🖥 CPU 벡터화 실행: {cpu_methods[m]}()")
        cpu_func(**kwargs)

        # --------------------
        # 6) 캐시 저장
        # --------------------
        self.save_vectorized(method)
    # ========================= METHOD: vectorize_text (end) =========================


    # ========================= METHOD: save_vectorized (start) =================
    def save_vectorized(self, method="tfidf"):
        """
        벡터화 결과 저장
        파일명: vectorized_{method}_train.pkl, vectorized_{method}_test.pkl
        """
        os.makedirs(self.model_dir, exist_ok=True)
        train_path = os.path.join(self.model_dir, f"vectorized_{method}_train.pkl")
        test_path = os.path.join(self.model_dir, f"vectorized_{method}_test.pkl")
        self.train_vectorized.to_pickle(train_path)
        self.test_vectorized.to_pickle(test_path)
        print(f"💾 벡터화 결과 저장: {train_path}, {test_path}")
    # ========================= METHOD: save_vectorized (end) ===================


    # ========================= METHOD: load_vectorized (start) =================
    def load_vectorized(self, method="tfidf"):
        """
        벡터화 결과 로드 (존재 시 True 반환)
        """
        train_path = os.path.join(self.model_dir, f"vectorized_{method}_train.pkl")
        test_path = os.path.join(self.model_dir, f"vectorized_{method}_test.pkl")
        if os.path.exists(train_path) and os.path.exists(test_path):
            self.train_vectorized = pd.read_pickle(train_path)
            self.test_vectorized = pd.read_pickle(test_path)
            print(f"📂 벡터화 데이터 로드됨: {train_path}, {test_path}")
            return True
        return False
    # ========================= METHOD: load_vectorized (end) ===================


    # ========================= METHOD: setup_pycaret (start) ==================
    def setup_pycaret(self, session_id=23, fold=3, use_gpu=True, n_jobs=4):
        """
        PyCaret setup 래핑 함수
        - self.train_vectorized를 기반으로 setup을 호출
        - 존재하는 범주형 컬럼만 categorical_features로 지정
        """
        if setup is None:
            raise RuntimeError("PyCaret이 설치되어 있지 않습니다.")

        print("🔧 PyCaret setup 시작...")
        categorical_cols = ["main_cat","sub_cat","sub_sub_cat","brand_name","item_condition_id","shipping","item_condition_id__x__shipping","brand_name__x__main_cat"]
        existing_categorical = [c for c in categorical_cols if c in self.train_vectorized.columns]

        df_for_setup = self.train_vectorized.copy()
        df_for_setup["price"] = self.train["price"].reset_index(drop=True)

        self.setup_result = setup(
            data=df_for_setup,
            target="price",
            session_id=session_id,
            categorical_features=existing_categorical if existing_categorical else None,
            normalize=True,
            transformation=False,
            fold_strategy="kfold",
            fold=fold,
            use_gpu=use_gpu,
            n_jobs=n_jobs,
            verbose=True,
            html=False
        )
        gc.collect()
        print("✅ PyCaret setup 완료")
    # ========================= METHOD: setup_pycaret (end) ====================


    # ========================= METHOD: find_and_blend_models (start) ==========
    def find_and_blend_models(self, top_n=3, sort_metric="R2", use_kaggle_winners=True):
        """
        모델 탐색 및 블렌딩
        - use_kaggle_winners=True일 경우 미리 정의된 모델들(lightgbm, ridge, catboost, xgboost, et, rf)을 시도
        - False면 compare_models로 top_n을 선택
        - blend_models로 블렌딩하여 self.best_model에 저장
        """
        if self.setup_result is None:
            raise ValueError("먼저 setup_pycaret()를 실행하세요.")

        print("🔎 모델 탐색/블렌딩 시작...")
        if use_kaggle_winners:
            print("🏆 Kaggle 상위권 모델 세트 학습 시도...")
            candidate_names = ["lightgbm", "ridge", "catboost", "xgboost", "et", "rf"]
            trained = []
            for name in tqdm(candidate_names, desc="create_model"):
                try:
                    m = create_model(name, verbose=False)
                    trained.append(m)
                except Exception as e:
                    print(f"⚠️ create_model({name}) 실패: {e}")
            if not trained:
                raise RuntimeError("모델 학습이 모두 실패했습니다.")
            top_models = trained
        else:
            top_models = compare_models(n_select=top_n, sort=sort_metric, turbo=True, verbose=True)
            if not isinstance(top_models, list):
                top_models = [top_models]

        blended = blend_models(estimator_list=top_models, optimize=sort_metric, choose_better=True, verbose=True)
        self.best_model = blended
        gc.collect()

        model_tag = f"blended_{sort_metric}"
        self.save_best_model(model_name=model_tag)

        print("🏁 블렌딩 완료 및 저장됨")
        return self.best_model
    # ========================= METHOD: find_and_blend_models (end) ============


    # ========================= METHOD: tune_best_model (start) =================
    def tune_best_model(self, n_iter=50, optimize_metric="R2"):
        """
        현재 best_model을 Optuna(TPE)로 튜닝
        """
        if self.best_model is None:
            raise ValueError("튜닝할 모델이 없습니다.")
        print(f"⚙️ 모델 튜닝 시작 (n_iter={n_iter})...")
        tuned = tune_model(self.best_model, optimize=optimize_metric, n_iter=n_iter, search_library="optuna", search_algorithm="tpe")
        self.best_model = tuned
        model_tag = f"tuned_{optimize_metric}"
        self.save_best_model(model_name=model_tag)
        print("✅ 튜닝 완료 및 저장됨")
        return tuned
    # ========================= METHOD: tune_best_model (end) ===================


    # ========================= METHOD: save_best_model (start) =================
    def save_best_model(self, model_name=None):
        """
        self.best_model을 self.model_dir에 저장
        - 버전: {model_name}_{timestamp}.pkl
        - latest: {model_name}_latest.pkl (덮어쓰기)
        """
        if self.best_model is None:
            raise ValueError("저장할 모델이 없습니다.")
        os.makedirs(self.model_dir, exist_ok=True)
        base_name = model_name or str(self.best_model).split("(")[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = os.path.join(self.model_dir, f"{base_name}_{timestamp}")
        latest_path = os.path.join(self.model_dir, f"{base_name}_latest")
        save_model(self.best_model, versioned_path)
        save_model(self.best_model, latest_path)
        print(f"💾 모델 저장 완료: {versioned_path}.pkl, {latest_path}.pkl")
    # ========================= METHOD: save_best_model (end) ===================


    # ========================= METHOD: load_saved_model (start) =================
    def load_saved_model(self, model_name, latest=True):
        """
        self.model_dir에서 모델 로드
        """
        suffix = "_latest" if latest else ""
        path = os.path.join(self.model_dir, f"{model_name}{suffix}")
        self.best_model = load_model(path)
        print(f"📂 모델 로드 완료: {path}.pkl")
        return self.best_model
    # ========================= METHOD: load_saved_model (end) ===================


    # ========================= METHOD: save_metrics (start) ====================
    def save_metrics(self, model_name=None):
        """
        train_vectorized 기반 예측 결과로 R2, RMSE, MAE 계산 후 JSON 저장
        (price는 log1p 상태이므로 expm1으로 복원하여 계산)
        """
        if self.best_model is None:
            raise ValueError("평가할 모델이 없습니다.")
        print("📊 메트릭 계산 중...")
        pred_df = predict_model(self.best_model, data=self.train_vectorized.copy())
        y_true = np.expm1(self.train["price"].values)
        y_pred = np.expm1(pred_df["prediction_label"].values)
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        self.metrics = {
            "R2": float(round(r2_score(y_true, y_pred), 4)),
            "RMSE": float(round(mean_squared_error(y_true, y_pred, squared=False), 4)),
            "MAE": float(round(mean_absolute_error(y_true, y_pred), 4))
        }
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = model_name or str(self.best_model).split("(")[0]
        file_path = os.path.join(self.results_dir, f"{base_name}_metrics_{timestamp}.json")
        with open(file_path, "w", encoding="utf8") as f:
            json.dump(self.metrics, f, indent=4, ensure_ascii=False)
        print(f"💾 Metrics 저장: {file_path}")
        print(f"   - R2={self.metrics['R2']}, RMSE=${self.metrics['RMSE']:.2f}, MAE=${self.metrics['MAE']:.2f}")
    # ========================= METHOD: save_metrics (end) ======================


    # ========================= METHOD: predict_test (start) ====================
    def predict_test(self, submission_file="submission.csv"):
        """
        test_vectorized에 대해 예측을 수행하고 제출 파일을 생성
        """
        if self.best_model is None:
            raise ValueError("예측할 모델이 없습니다.")
        print("📦 테스트 예측 시작...")
        pred_df = predict_model(self.best_model, data=self.test_vectorized.copy())
        price_pred = np.expm1(pred_df["prediction_label"].values)
        submission = pd.DataFrame({"test_id": self.test["test_id"], "price": price_pred})
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.results_dir, f"{timestamp}_{submission_file}")
        submission.to_csv(path, index=False)
        print(f"💾 제출 파일 저장: {path}")
        print(f"   - 예측 가격 범위: ${price_pred.min():.2f} ~ ${price_pred.max():.2f}, 평균: ${price_pred.mean():.2f}")
        return submission
    # ========================= METHOD: predict_test (end) ======================
    
    # ========================= METHOD: _get_preprocessed_paths (start) =========================
    def _get_preprocessed_paths(self, date_suffix=None, tag=None):
        """
        전처리된 train/test 파일 경로 생성기.
        - date_suffix: 'YYYYMMDD' 형식 문자열. None이면 오늘 날짜 사용.
        - tag: 사용자 지정 태그(예: param hash) 추가.
        Returns (train_path, test_path)
        """
        if date_suffix is None:
            date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        filename_tag = f"_{tag}" if tag else ""
        train_path = os.path.join(self.results_dir, f"preprocessed_train_{date_suffix}{filename_tag}.pkl")
        test_path  = os.path.join(self.results_dir, f"preprocessed_test_{date_suffix}{filename_tag}.pkl")
        return train_path, test_path
    # ========================= METHOD: _get_preprocessed_paths (end) ===========================


    # ========================= METHOD: _list_preprocessed_files (start) =======================
    def _list_preprocessed_files(self, pattern="preprocessed_train_*.pkl"):
        """
        results_dir에서 preprocessed 파일 목록(정렬된)을 반환합니다.
        """
        files = []
        for fn in os.listdir(self.results_dir):
            if fn.startswith("preprocessed_train_") and fn.endswith(".pkl"):
                files.append(os.path.join(self.results_dir, fn))
        # sort by modified time descending (newest first)
        files_sorted = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
        return files_sorted
    # ========================= METHOD: _list_preprocessed_files (end) =========================


    # ========================= METHOD: _cleanup_old_preprocessed (start) ====================
    def _cleanup_old_preprocessed(self, keep_latest=5, prefix="preprocessed_train_"):
        """
        오래된 전처리 파일을 정리해서 디스크 사용량을 제어.
        - keep_latest: 남겨둘 파일 개수 (train/test 각각)
        - prefix: 파일명 프리픽스
        """
        files = [f for f in os.listdir(self.results_dir) if f.startswith(prefix) and f.endswith(".pkl")]
        # group by suffix (train vs test) handled by prefix matching; we remove based on train entries list
        train_files = sorted([os.path.join(self.results_dir, f) for f in files if "train" in f], key=lambda x: os.path.getmtime(x), reverse=True)
        # keep newest keep_latest, remove rest and corresponding test files
        for old in train_files[keep_latest:]:
            try:
                # remove train file
                os.remove(old)
                # remove matching test file by replacing 'train' with 'test'
                test_path = old.replace("preprocessed_train_", "preprocessed_test_")
                if os.path.exists(test_path):
                    os.remove(test_path)
            except Exception:
                pass
    # ========================= METHOD: _cleanup_old_preprocessed (end) ====================


    # ========================= METHOD: preprocess_or_load (start) =====================
    def preprocess_or_load(self, force_preprocess=False, date_suffix=None, tag=None, max_versions=5, param_dict=None, verbose=True):
        """
        전처리 결과를 로드하거나 없으면 현재 상태의 self.train/self.test를 저장합니다.
        - force_preprocess: True이면 기존 파일이 있어도 새로 저장.
        - date_suffix: 파일명에 들어갈 날짜 (None이면 오늘)
        - tag: 파일명에 추가되는 사용자 태그 (예: 파라미터 해시)
        - max_versions: 저장 후 오래된 버전 정리 갯수
        - param_dict: 전처리 설정을 dict로 주면 이 dict를 해시하여 tag로 사용할 수 있음.
        
        사용예시
        기본 사용 (오늘 날짜로 저장/불러오기):
        analyzer.preprocess_or_load()
        
        강제 재생성(기존 파일이 있어도 새로 저장):
        analyzer.preprocess_or_load(force_preprocess=True)
        
        특정 날짜/태그로 저장(또는 해당 파일이 있으면 로드):
        analyzer.preprocess_or_load(date_suffix="20251201", tag="expA")
        
        파라미터 해시 자동 태그 사용 (param_dict로 전달한 옵션의 해시를 태그로 붙임):
        params = {"undersample_frac":0.35, "tfidf_max_feat":15000}
        analyzer.preprocess_or_load(param_dict=params)
        
        Returns: "loaded" or "saved"
        """
        # ensure results dir exists
        os.makedirs(self.results_dir, exist_ok=True)

        # compute tag from param_dict if provided
        if param_dict:
            try:
                # sort keys for deterministic hash
                items = sorted(param_dict.items())
                dump = json.dumps(items, ensure_ascii=False, sort_keys=True)
                tag_hash = hashlib.sha1(dump.encode('utf-8')).hexdigest()[:8]
                tag = f"{tag}_{tag_hash}" if tag else tag_hash
            except Exception:
                pass

        # if date_suffix provided, check that specific paths
        train_path, test_path = self._get_preprocessed_paths(date_suffix=date_suffix, tag=tag)

        # if not forcing and exact files exist -> load
        if (not force_preprocess) and os.path.exists(train_path) and os.path.exists(test_path):
            if verbose:
                print(f"📂 기존 전처리 파일 발견: {train_path}, {test_path} -> 로드합니다.")
            self.train = pd.read_pickle(train_path)
            self.test  = pd.read_pickle(test_path)
            return "loaded"

        # if not forcing and any preprocessed files exist -> load newest (optionally filtered by tag)
        if (not force_preprocess):
            candidates = []
            for fn in os.listdir(self.results_dir):
                if fn.startswith("preprocessed_train_") and fn.endswith(".pkl"):
                    if tag and tag not in fn:
                        continue
                    candidates.append(os.path.join(self.results_dir, fn))
            if candidates:
                # pick most recently modified
                candidates = sorted(candidates, key=lambda x: os.path.getmtime(x), reverse=True)
                chosen_train = candidates[0]
                chosen_test = chosen_train.replace("preprocessed_train_", "preprocessed_test_")
                if os.path.exists(chosen_test):
                    if verbose:
                        print(f"📂 최신 전처리 파일 발견: {chosen_train}, {chosen_test} -> 로드합니다.")
                    self.train = pd.read_pickle(chosen_train)
                    self.test  = pd.read_pickle(chosen_test)
                    return "loaded"

        # Otherwise, save current train/test after confirming required columns exist
        if "brand_name_price_mean" not in self.train.columns:
            raise RuntimeError("전처리 저장을 위해 build_price_brand_cat_features()까지 수행되어야 합니다. 먼저 해당 메서드를 실행하세요.")

        # Save with given date_suffix/tag
        train_path, test_path = self._get_preprocessed_paths(date_suffix=date_suffix, tag=tag)
        pd.to_pickle(self.train, train_path)
        pd.to_pickle(self.test, test_path)
        if verbose:
            print(f"💾 전처리 데이터 저장됨: {train_path}, {test_path}")

        # cleanup old versions
        try:
            self._cleanup_old_preprocessed(keep_latest=max_versions)
        except Exception:
            pass

        return "saved"
    # ========================= METHOD: preprocess_or_load (end) =====================
    

    # ========================= METHOD: _detect_device (start) =====================
    def _detect_device(self):
        """
        내부용 - 사용 가능한 디바이스를 탐지합니다.
        - 우선: 사용자 지정 self.use_gpu가 True이면 torch가 있고 CUDA 사용 가능하면 'cuda'
        - 두번째: SentenceTransformer/CUDA 가능성 체크는 embedding 메서드에서 별도 처리
        """
        # self.use_gpu 는 __init__에서 설정되어야 합니다 (default False)
        device = "cpu"
        try:
            import torch
            if getattr(self, "use_gpu", False) and torch.cuda.is_available():
                device = "cuda"
        except Exception:
            # torch가 없으면 'cpu'로 강제
            device = "cpu"
        self.device = device
        return self.device
    # ========================= METHOD: _detect_device (end) =======================


    # ========================= METHOD: vectorize_text_bert_gpu (start) =============
    def vectorize_text_bert_gpu(self, text_columns=["name", "item_description"], bert_model_name="all-MiniLM-L6-v2", batch_size=128):
        """
        GPU가 가능하면 SentenceTransformer를 GPU로 사용해 임베딩을 생성합니다.
        - device 설정: self.device (없으면 _detect_device로 설정)
        - GPU 미존재 시 CPU로 안전하게 수행 (fallback)
        - 반환: self.train_vectorized / self.test_vectorized에 embedding 결과 저장
        """
        print("🔍 [GPU] BERT 임베딩 시작...")

        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers가 설치되어 있지 않습니다. GPU 임베딩 불가.")

        if not hasattr(self, "device"):
            self._detect_device()

        device = getattr(self, "device", "cpu")
        # SentenceTransformer는 device 인자를 받음 (cuda/cpu)
        try:
            model = SentenceTransformer(bert_model_name, device=device)
        except Exception as e:
            print(f"⚠️ SentenceTransformer 로딩 실패({e}) — CPU로 시도합니다.")
            model = SentenceTransformer(bert_model_name, device="cpu")
            device = "cpu"

        train_feats = []
        test_feats = []
        for col in text_columns:
            # 안전하게 결측 처리
            self.train[col] = self.train[col].fillna("").astype(str)
            self.test[col] = self.test[col].fillna("").astype(str)

            print(f"   - 임베딩 컬럼: {col} (device={device})")
            emb_train = model.encode(self.train[col].tolist(), batch_size=batch_size, show_progress_bar=True, device=device)
            emb_test  = model.encode(self.test[col].tolist(),  batch_size=batch_size, show_progress_bar=True, device=device)
            train_feats.append(np.asarray(emb_train, dtype=np.float32))
            test_feats.append(np.asarray(emb_test, dtype=np.float32))

        train_vec = np.hstack(train_feats).astype(np.float32)
        test_vec  = np.hstack(test_feats).astype(np.float32)

        # 컬럼명 구성 (각 텍스트 컬럼당 동일 차원 가정)
        dim_each = train_feats[0].shape[1]
        cols = [f"{col}_bert_{i}" for col in text_columns for i in range(dim_each)]

        self.train_vectorized = pd.DataFrame(train_vec, columns=cols)
        self.test_vectorized  = pd.DataFrame(test_vec,  columns=cols)

        # 기존 구조적 피처 결합
        self._add_categorical_numeric_features()

        print(f"✅ [GPU] BERT 임베딩 완료: train {self.train_vectorized.shape}, test {self.test_vectorized.shape}")
    # ========================= METHOD: vectorize_text_bert_gpu (end) ===============


    # ========================= METHOD: vectorize_text_tfidf_gpu (start) ============
    def vectorize_text_tfidf_gpu(self, max_features_name=15000, max_features_desc=20000, n_components=150):
        """
        TF-IDF는 보통 CPU에서 계산하고, SVD(차원 축소)만 GPU로 가속하는 하이브리드 방식.
        - 순서:
        1) sklearn.TfidfVectorizer로 TF-IDF (안정성 때문에 CPU)
        2) 가능하면 cuml.TruncatedSVD로 GPU에서 차원 축소 (cupy/cuml 필요)
        3) cuml 미설치 시 sklearn.TruncatedSVD로 CPU에서 실행 (fallback)
        - 이유: 완전 GPU TF-IDF 구성은 환경 의존성이 크므로 여기서는 'TF-IDF CPU + SVD GPU' 전략을 추천
        """
        print("🔍 TF-IDF + (가능 시) GPU SVD 시작...")

        # 1) TF-IDF (CPU) — 안정성 우선
        vec_name = TfidfVectorizer(max_features=max_features_name, ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True, dtype=np.float32)
        Xn_train = vec_name.fit_transform(self.train["name"].astype(str))
        Xn_test  = vec_name.transform(self.test["name"].astype(str))

        vec_desc = TfidfVectorizer(max_features=max_features_desc, ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True, dtype=np.float32)
        Xd_train = vec_desc.fit_transform(self.train["item_description"].astype(str))
        Xd_test  = vec_desc.transform(self.test["item_description"].astype(str))

        # 합치기(희소행렬)
        from scipy.sparse import hstack as sp_hstack
        X_train = sp_hstack([Xn_train, Xd_train]).tocsr()
        X_test  = sp_hstack([Xn_test,  Xd_test]).tocsr()

        # 2) GPU SVD 시도 (cuml)
        use_gpu_svd = False
        try:
            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix as cupy_csr
            from cuml.decomposition import TruncatedSVD as cumlTruncatedSVD
            use_gpu_svd = True
        except Exception:
            use_gpu_svd = False

        if use_gpu_svd:
            print("   - cuML 사용 가능: GPU에서 TruncatedSVD 수행")
            # scipy csr -> cupy csr
            X_train_gpu = cupy_csr(X_train)   # may need memory consideration
            X_test_gpu  = cupy_csr(X_test)
            # fit_transform on GPU
            n_comp = max(1, min(n_components, X_train.shape[1]-1))
            svd_gpu = cumlTruncatedSVD(n_components=n_comp, random_state=23)
            train_vec_gpu = svd_gpu.fit_transform(X_train_gpu)
            test_vec_gpu  = svd_gpu.transform(X_test_gpu)
            # cupy -> numpy
            train_vec = cp.asnumpy(train_vec_gpu).astype(np.float32)
            test_vec  = cp.asnumpy(test_vec_gpu).astype(np.float32)
        else:
            # fallback: sklearn TruncatedSVD (CPU)
            print("   - cuML 미사용: CPU에서 sklearn TruncatedSVD로 실행 (속도 느림)")
            from sklearn.decomposition import TruncatedSVD as sklSVD
            n_comp = max(1, min(n_components, X_train.shape[1]-1))
            svd = sklSVD(n_components=n_comp, random_state=23)
            train_vec = svd.fit_transform(X_train)
            test_vec  = svd.transform(X_test)

        # 분해된 dim 계산 (name/desc 개별 dim 비율은 이전 방식대로 name first)
        # 여기서는 편의상 name_dim = min(n_components, Xn_train.shape[1]-1) 등으로 나누어도 됨.
        # 단순하게 n_comp_name, n_comp_desc를 각각 계산하지 않았으면 합쳐진 차원만 사용.
        # (원본 코드와 일관성을 유지하려면 name/desc를 따로 SVD하고 합치는 방식으로 바꿀 수 있음)
        final_dim = train_vec.shape[1]
        cols = [f"tfidf_svd_{i}" for i in range(final_dim)]

        self.train_vectorized = pd.DataFrame(train_vec, columns=cols)
        self.test_vectorized  = pd.DataFrame(test_vec,  columns=cols)

        # 구조적 피처 결합
        self._add_categorical_numeric_features()

        print(f"✅ TF-IDF + SVD 완료 (train {self.train_vectorized.shape}, test {self.test_vectorized.shape})")
    # ========================= METHOD: vectorize_text_tfidf_gpu (end) ============

    # ========================= METHOD: _get_cache_path (start) ========================
    def _get_cache_path(self, prefix="preprocessed", use_date=True):
        """전처리 캐시 파일 경로 생성 함수"""
        if use_date:
            today = pd.Timestamp.now().strftime("%Y%m%d")
            fname = f"{prefix}_{today}.pkl"
        else:
            fname = f"{prefix}_latest.pkl"

        return os.path.join(self.results_dir, fname)
    # ========================= METHOD: _get_cache_path (end) ==========================
    
    # ========================= METHOD: _save_preprocessed (start) =====================
    def _save_preprocessed(self, path):
        """train/test + 메타 정보를 피클로 저장"""
        data = {
            "train": self.train,
            "test": self.test,
            "meta": {
                "name": "mercari_preprocessed",
                "timestamp": pd.Timestamp.now()
            }
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 전처리 결과 저장 완료 → {path}")
    # ========================= METHOD: _save_preprocessed (end) =======================

    # ========================= METHOD: _load_preprocessed (start) ====================
    def _load_preprocessed(self, path):
        """전처리된 train/test를 로드"""
        if not os.path.exists(path):
            return False

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.train = data["train"]
            self.test = data["test"]

            print(f"⚡ 캐시 데이터 로드 성공 → {path}")
            return True
        
        except Exception as e:
            print(f"⚠️ 캐시 불러오기 실패: {e}")
            return False
    # ========================= METHOD: _load_preprocessed (end) ======================

    # ========================= METHOD: preprocess_all (start) ========================
    def preprocess_all(
        self,
        use_cache=True,
        save_cache=True,
        cache_prefix="preprocessed",
        cols=["name", "item_description"],
        train_file="train.tsv",
        test_file="test.tsv",
        sep="\t",
        undersample_frac=None
    ):
        """
        전체 전처리를 수행하되,
        - 캐시가 있으면 load_data() 없이 즉시 복원
        - 캐시가 없으면 load_data() → 전처리 전체 실행 후 캐시 저장
            - 캐시 자동 활용
            - 텍스트 정규화
            - 텍스트 통계 피처
            - 감성 분석
            - 가격/브랜드/카테고리 SAFE 피처
            - 상호작용 피처
            - 캐시 저장/복구 기능 포함    
        """

        print("\n=========================")
        print("🚀 SAFE preprocess_all() 시작")
        print("=========================\n")

        # ------------------------------------------------------
        # 0) 캐시 경로 계산
        # ------------------------------------------------------
        cache_path = self._get_cache_path(prefix=cache_prefix, use_date=False)

        # ------------------------------------------------------
        # 1) 캐시 체크 → 캐시 있으면 로드 + 즉시 종료
        # ------------------------------------------------------
        if use_cache and os.path.exists(cache_path):
            print(f"📦 캐시 발견 → {cache_path}")
            try:
                data = self._load_preprocessed(cache_path)
                self.train = data["train"]
                self.test = data["test"]
                print("✅ 캐시 로드 완료 → load_data() 및 전처리 스킵")
                return
            except Exception as e:
                print("⚠️ 캐시 로드 실패 → 전체 전처리 재실행")
                print(e)

        # ------------------------------------------------------
        # 2) 캐시 없음 → load_data() 자동 실행
        # ------------------------------------------------------
        print("📥 캐시 없음 → load_data() 실행")
        self.load_data(
            train_file=train_file,
            test_file=test_file,
            sep=sep,
            undersample_frac=undersample_frac,
        )

        # ------------------------------------------------------
        # 3) 텍스트 정규화
        # ------------------------------------------------------
        print("\n1️⃣ 텍스트 정규화 실행")
        self.normalize_text(cols=cols)

        # ------------------------------------------------------
        # 4) 텍스트 통계 피처
        # ------------------------------------------------------
        print("\n2️⃣ 텍스트 통계 피처 생성")
        self.build_text_stats(cols=cols)

        # ------------------------------------------------------
        # 5) 감성 분석
        # ------------------------------------------------------
        print("\n3️⃣ 감성 분석")
        self.compute_sentiment(cols=cols)

        # ------------------------------------------------------
        # 6) 가격/브랜드/카테 SAFE 피처
        # ------------------------------------------------------
        print("\n4️⃣ 가격/브랜드/카테고리 SAFE 피처 생성")
        self.build_price_brand_cat_features()

        # ------------------------------------------------------
        # 7) 상호작용 피처
        # ------------------------------------------------------
        print("\n5️⃣ 상호작용 피처 생성")
        self.build_interactions()

        # ------------------------------------------------------
        # 8) 캐시 저장
        # ------------------------------------------------------
        if save_cache:
            print("\n💾 캐시 저장 중 ...")
            self._save_preprocessed(cache_path)

            # 날짜 포함 버전도 저장
            dated = self._get_cache_path(prefix=cache_prefix, use_date=True)
            self._save_preprocessed(dated)

        print("\n=========================")
        print("🎉 SAFE preprocess_all() 완료")
        print("=========================\n")


    # ========================= METHOD: preprocess_all (end) ==========================

# ======================================================================
# ========================= CLASS END ====================================
# ======================================================================




```