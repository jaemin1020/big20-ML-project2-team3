```python
import pandas as pd
import numpy as np
import os
import json
import datetime
import gc
import re
import warnings
from tqdm import tqdm
from gensim.models import FastText
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

from pycaret.regression import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class MercariPyCaretAnalyzer:
    """
    Mercari Price Suggestion Challenge를 위한 PyCaret 기반 머신러닝 파이프라인
  
    주요 기능: 데이터 전처리, 텍스트 벡터화(TF-IDF/FastText/BERT), 
              피처 엔지니어링, 모델 학습/블렌딩, 성능 평가
  
    Parameters
    ----------
    data_dir : str, default="../data"
    images_dir : str, default="../images"  
    results_dir : str, default="../results"
  
    Examples
    --------
    >>> analyzer = MercariPyCaretAnalyzer()
    >>> analyzer.load_data(undersample_frac=0.35)
    >>> analyzer.vectorize_text(method="tfidf")
    >>> analyzer.setup_pycaret(fold=3)
    >>> analyzer.find_and_blend_models(use_kaggle_winners=True)
    >>> analyzer.save_metrics()
    >>> analyzer.predict_test()
    """

    # __init__ ##############################
    def __init__(self, data_dir="../data", images_dir="../images", results_dir="../results"):
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.results_dir = results_dir
        self.train = None
        self.test = None
        self.best_model = None
        self.setup_result = None
        self.metrics = {}
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    # eof -----------------------------------

    # _collapse_rare_values #################
    def _collapse_rare_values(self, col, top_k, rare_label="Other"):
        """희귀값 통합"""
        combined = pd.concat([self.train[col], self.test[col]], axis=0)
        value_counts = combined.value_counts()
        top_values = set(value_counts.index[:top_k])
        self.train[col] = self.train[col].apply(lambda x: x if x in top_values else rare_label)
        self.test[col] = self.test[col].apply(lambda x: x if x in top_values else rare_label)
        del combined, value_counts
        gc.collect()
    # eof -----------------------------------

    # _simple_normalize #####################
    def _simple_normalize(self, text: str) -> str:
        """텍스트 정규화"""
        text = str(text).lower()
        text = re.sub(r"[_\-\./]", " ", text)
        text = re.sub(r"\d+", " num ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    # eof -----------------------------------

    # _stratified_sample ####################
    def _stratified_sample(self, frac=0.35, bins=10):
        """층화 샘플링"""
        self.train["price_bin"] = pd.qcut(self.train["price"], q=bins, duplicates="drop")
        sampled = self.train.groupby("price_bin", group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=23)
        )
        self.train = sampled.drop(columns=["price_bin"]).reset_index(drop=True)
        gc.collect()
        print(f"⚠️ Stratified undersampling 적용: train {self.train.shape}")
    # eof -----------------------------------

    # load_data #############################
    def load_data(self, train_file="train.tsv", test_file="test.tsv", sep="\t", undersample_frac=0.35):
        """데이터 로딩 및 전처리"""
        print("📂 데이터 로딩 시작...")
        self.train = pd.read_csv(os.path.join(self.data_dir, train_file), sep=sep)
        self.test = pd.read_csv(os.path.join(self.data_dir, test_file), sep=sep)
        self.train = self.train[self.train["price"] > 0].dropna(subset=["price"])
        self.train["price"] = np.log1p(self.train["price"])
        if undersample_frac:
            self._stratified_sample(frac=undersample_frac)
        for df in [self.train, self.test]:
            df["main_cat"], df["sub_cat"], df["sub_sub_cat"] = zip(
                *df["category_name"].apply(lambda x: (x.split("/") if isinstance(x, str) and "/" in x else ["missing"]*3))
            )
            df["brand_name"] = df["brand_name"].fillna("Unknown").astype(str)
            df["item_description"] = df["item_description"].fillna("No description").astype(str)
            df["name"] = df["name"].fillna("No name").astype(str)
            df.drop(columns=["category_name"], inplace=True)
        print("🔄 희귀값 통합 중...")
        self._collapse_rare_values("brand_name", 5000, "Other_brand")
        self._collapse_rare_values("main_cat", 1000, "Other_main")
        self._collapse_rare_values("sub_cat", 1000, "Other_sub")
        self._collapse_rare_values("sub_sub_cat", 1000, "Other_sub_sub")
        print("📏 피처 생성 중...")
        for df in [self.train, self.test]:
            df["name_len_char"] = df["name"].str.len()
            df["name_len_word"] = df["name"].str.split().str.len()
            df["desc_len_char"] = df["item_description"].str.len()
            df["desc_len_word"] = df["item_description"].str.split().str.len()
            df["has_brand_in_name"] = df.apply(lambda r: 1 if r["brand_name"].lower() in r["name"].lower() else 0, axis=1)
            df["has_brand_in_desc"] = df.apply(lambda r: 1 if r["brand_name"].lower() in r["item_description"].lower() else 0, axis=1)
            df["shipping"] = df["shipping"].astype("category")
            df["item_condition_id"] = df["item_condition_id"].astype("category")
        gc.collect()
        print(f"✅ 데이터 로드 완료: train {self.train.shape}, test {self.test.shape}")
    # eof -----------------------------------

    # vectorize_text_tfidf ##################
    def vectorize_text_tfidf(self, max_features_name=15000, max_features_desc=20000, n_components=150):
        """TF-IDF 벡터화"""
        print("🔍 TF-IDF 벡터화 시작...")
        vec_name = TfidfVectorizer(max_features=max_features_name, ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True, dtype=np.float32)
        name_train = vec_name.fit_transform(self.train["name"])
        name_test = vec_name.transform(self.test["name"])
        svd = TruncatedSVD(n_components=min(n_components, name_train.shape[1]-1), random_state=23)
        name_train_svd = svd.fit_transform(name_train)
        name_test_svd = svd.transform(name_test)
        print(f"   - name SVD: {name_train_svd.shape}, 설명력={svd.explained_variance_ratio_.sum():.2%}")
        del name_train, name_test, vec_name
        gc.collect()
        vec_desc = TfidfVectorizer(max_features=max_features_desc, ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True, dtype=np.float32)
        desc_train = vec_desc.fit_transform(self.train["item_description"])
        desc_test = vec_desc.transform(self.test["item_description"])
        svd = TruncatedSVD(n_components=min(n_components, desc_train.shape[1]-1), random_state=23)
        desc_train_svd = svd.fit_transform(desc_train)
        desc_test_svd = svd.transform(desc_test)
        print(f"   - desc SVD: {desc_train_svd.shape}, 설명력={svd.explained_variance_ratio_.sum():.2%}")
        del desc_train, desc_test, vec_desc, svd
        gc.collect()
        train_vec = np.hstack([name_train_svd, desc_train_svd]).astype(np.float32)
        test_vec = np.hstack([name_test_svd, desc_test_svd]).astype(np.float32)
        del name_train_svd, name_test_svd, desc_train_svd, desc_test_svd
        gc.collect()
        self.train_vectorized = pd.DataFrame(train_vec, columns=[f"name_{i}" for i in range(n_components)] + [f"desc_{i}" for i in range(n_components)])
        self.test_vectorized = pd.DataFrame(test_vec, columns=[f"name_{i}" for i in range(n_components)] + [f"desc_{i}" for i in range(n_components)])
        self._add_categorical_numeric_features()
        print(f"✅ TF-IDF 완료: train {self.train_vectorized.shape}, test {self.test_vectorized.shape}")
    # eof -----------------------------------

    # vectorize_text_fasttext ###############
    def vectorize_text_fasttext(self, text_columns=["name", "item_description"], fasttext_size=100, fasttext_window=5, fasttext_min_count=2):
        """FastText 벡터화"""
        print("🔍 FastText 벡터화 시작...")
        sentences = []
        for col in text_columns:
            self.train[col] = self.train[col].fillna("").astype(str)
            self.test[col] = self.test[col].fillna("").astype(str)
            sentences += [str(x).split() for x in pd.concat([self.train[col], self.test[col]])]
        print("   - FastText 학습 중...")
        ft_model = FastText(sentences, vector_size=fasttext_size, window=fasttext_window, min_count=fasttext_min_count, sg=1, workers=4)
        def get_vector(text):
            words = str(text).split()
            vectors = [ft_model.wv[w] for w in words if w in ft_model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(fasttext_size)
        train_features, test_features = [], []
        for col in tqdm(text_columns, desc="벡터 생성"):
            train_features.append(np.vstack(self.train[col].apply(get_vector)))
            test_features.append(np.vstack(self.test[col].apply(get_vector)))
        train_vec = np.hstack(train_features).astype(np.float32)
        test_vec = np.hstack(test_features).astype(np.float32)
        self.train_vectorized = pd.DataFrame(train_vec, columns=[f"{col}_ft_{i}" for col in text_columns for i in range(fasttext_size)])
        self.test_vectorized = pd.DataFrame(test_vec, columns=[f"{col}_ft_{i}" for col in text_columns for i in range(fasttext_size)])
        self._add_categorical_numeric_features()
        print(f"✅ FastText 완료: train {self.train_vectorized.shape}, test {self.test_vectorized.shape}")
    # eof -----------------------------------

    # vectorize_text_bert ###################
    def vectorize_text_bert(self, text_columns=["name", "item_description"], bert_model_name="all-MiniLM-L6-v2"):
        """BERT 벡터화"""
        print(f"🔍 BERT 시작 (모델={bert_model_name})...")
        bert_model = SentenceTransformer(bert_model_name)
        train_features, test_features = [], []
        for col in text_columns:
            self.train[col] = self.train[col].fillna("").astype(str)
            self.test[col] = self.test[col].fillna("").astype(str)
            print(f"   - {col} 인코딩 중...")
            train_features.append(bert_model.encode(self.train[col].tolist(), show_progress_bar=True, batch_size=32))
            test_features.append(bert_model.encode(self.test[col].tolist(), show_progress_bar=True, batch_size=32))
        train_vec = np.hstack(train_features).astype(np.float32)
        test_vec = np.hstack(test_features).astype(np.float32)
        emb_dim = bert_model.get_sentence_embedding_dimension()
        self.train_vectorized = pd.DataFrame(train_vec, columns=[f"{col}_bert_{i}" for col in text_columns for i in range(emb_dim)])
        self.test_vectorized = pd.DataFrame(test_vec, columns=[f"{col}_bert_{i}" for col in text_columns for i in range(emb_dim)])
        self._add_categorical_numeric_features()
        print(f"✅ BERT 완료: train {self.train_vectorized.shape}, test {self.test_vectorized.shape}")
    # eof -----------------------------------

    # _add_categorical_numeric_features #####
    def _add_categorical_numeric_features(self):
        """범주형/수치형 피처 추가"""
        categorical_features = ["main_cat", "sub_cat", "sub_sub_cat", "brand_name", "item_condition_id", "shipping"]
        numeric_features = ["name_len_char", "name_len_word", "desc_len_char", "desc_len_word", "has_brand_in_name", "has_brand_in_desc"]
        print("📌 범주형/수치형 피처 추가 중...")
        for col in categorical_features + numeric_features:
            if col in self.train.columns:
                self.train_vectorized[col] = self.train[col].reset_index(drop=True)
                self.test_vectorized[col] = self.test[col].reset_index(drop=True)
        print(f"   - 범주형:{len(categorical_features)}개, 수치형:{len(numeric_features)}개, 총:{self.train_vectorized.shape[1]}개")
    # eof -----------------------------------

    # vectorize_text ########################
    def vectorize_text(self, method="tfidf", **kwargs):
        """텍스트 벡터화 통합 인터페이스"""
        if self.load_vectorized(method):
            return
        if method == "tfidf":
            self.vectorize_text_tfidf(**kwargs)
        elif method == "fasttext":
            self.vectorize_text_fasttext(**kwargs)
        elif method == "bert":
            self.vectorize_text_bert(**kwargs)
        else:
            raise ValueError("method must be one of ['tfidf','fasttext','bert']")
        self.save_vectorized(method)
    # eof -----------------------------------

    # setup_pycaret #########################
    def setup_pycaret(self, session_id=23, fold=3, use_gpu=False):
        """PyCaret 환경 설정"""
        print("🔧 PyCaret setup 시작...")
        categorical_cols = ["main_cat","sub_cat","sub_sub_cat","brand_name","item_condition_id","shipping"]
        existing_categorical = [col for col in categorical_cols if col in self.train_vectorized.columns]
        print(f"   - 범주형:{len(existing_categorical)}개, 전체:{self.train_vectorized.shape[1]}개")
        self.setup_result = setup(
            data=self.train_vectorized.assign(price=self.train["price"].reset_index(drop=True)),
            target="price", session_id=session_id, categorical_features=existing_categorical if existing_categorical else None,
            normalize=True, transformation=False, fold_strategy="kfold", fold=fold, use_gpu=use_gpu, n_jobs=4, verbose=True, html=False
        )
        gc.collect()
        print("✅ PyCaret setup 완료")
    # eof -----------------------------------

    # find_and_blend_models #################
    def find_and_blend_models(self, top_n=3, sort_metric="R2", use_kaggle_winners=True):
        """모델 탐색 및 블렌딩"""
        if not self.setup_result:
            raise ValueError("먼저 setup_pycaret()를 실행하세요.")
        if use_kaggle_winners:
            print("🏆 Kaggle 상위권 5개 모델 학습...")
            top_models = [create_model(name, verbose=False) for name in tqdm(["lightgbm", "ridge", "catboost", "xgboost", "et"], desc="모델 학습")]
            print("✅ 5개 모델 학습 완료")
        else:
            print(f"🔍 전체 모델 탐색 (상위 {top_n}개)...")
            top_models = compare_models(n_select=top_n, sort=sort_metric, turbo=True, verbose=True)
            if not isinstance(top_models, list):
                top_models = [top_models]
        print("🎯 선정 모델:")
        for i, m in enumerate(top_models, 1):
            print(f"   {i}. {str(m).split('(')[0]}")
        print(f"\n🔀 {len(top_models)}개 모델 블렌딩...")
        blended = blend_models(estimator_list=top_models, optimize=sort_metric, choose_better=True, verbose=True)
        self.best_model = blended
        gc.collect()
        print(f"🏆 Blended model 완료 (기준={sort_metric})")
        return self.best_model
    # eof -----------------------------------

    # tune_best_model #######################
    def tune_best_model(self, n_iter=50, optimize_metric="R2"):
        """모델 튜닝"""
        if not self.best_model:
            raise ValueError("먼저 find_and_blend_models()를 실행하세요.")
        print(f"⚙️ 튜닝 시작 (n_iter={n_iter})...")
        tuned = tune_model(self.best_model, optimize=optimize_metric, n_iter=n_iter, search_library='optuna', search_algorithm='tpe')
        self.best_model = tuned
        print("✅ 튜닝 완료!")
        return tuned
    # eof -----------------------------------

    # save_metrics ##########################
    def save_metrics(self, model_name=None):
        """성능 지표 저장"""
        if not self.best_model:
            raise ValueError("모델이 없습니다.")
        print("📊 성능 평가 중...")
        pred_df = predict_model(self.best_model, data=self.train_vectorized.copy())
        y_true = np.expm1(self.train["price"].values)
        y_pred = np.expm1(pred_df["prediction_label"].values)
        self.metrics = {"R2": round(r2_score(y_true, y_pred), 4), "RMSE": round(mean_squared_error(y_true, y_pred, squared=False), 4), "MAE": round(mean_absolute_error(y_true, y_pred), 4)}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_name or str(self.best_model).split("(")[0]
        file_path = os.path.join(self.results_dir, f"{model_name}_metrics_{timestamp}.json")
        with open(file_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"💾 Metrics 저장: {file_path}")
        print(f"   - R²={self.metrics['R2']}, RMSE=${self.metrics['RMSE']:.2f}, MAE=${self.metrics['MAE']:.2f}")
    # eof -----------------------------------

    # predict_test ##########################
    def predict_test(self, submission_file="submission.csv"):
        """테스트 예측"""
        if not self.best_model:
            raise ValueError("먼저 find_and_blend_models()를 실행하세요.")
        print("📦 Test 예측 시작...")
        predictions = predict_model(self.best_model, data=self.test_vectorized.copy())
        price_pred = np.expm1(predictions["prediction_label"].values)
        submission = pd.DataFrame({"test_id": self.test["test_id"], "price": price_pred})
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.results_dir, f"{timestamp}_{submission_file}")
        submission.to_csv(path, index=False)
        print(f"💾 Submission 저장: {path}")
        print(f"   - 가격 범위: ${price_pred.min():.2f} ~ ${price_pred.max():.2f}, 평균: ${price_pred.mean():.2f}")
        return submission
    # eof -----------------------------------

    # save_vectorized #######################
    def save_vectorized(self, method="tfidf"):
        """벡터화 결과 저장"""
        train_path = os.path.join(self.results_dir, f"vectorized_{method}_train.pkl")
        test_path = os.path.join(self.results_dir, f"vectorized_{method}_test.pkl")
        self.train_vectorized.to_pickle(train_path)
        self.test_vectorized.to_pickle(test_path)
        print(f"💾 {method} 벡터화 결과 저장 완료")
    # eof -----------------------------------

    # load_vectorized #######################
    def load_vectorized(self, method="tfidf"):
        """벡터화 결과 로드"""
        train_path = os.path.join(self.results_dir, f"vectorized_{method}_train.pkl")
        test_path = os.path.join(self.results_dir, f"vectorized_{method}_test.pkl")
        if os.path.exists(train_path) and os.path.exists(test_path):
            self.train_vectorized = pd.read_pickle(train_path)
            self.test_vectorized = pd.read_pickle(test_path)
            print(f"📂 {method} 벡터화 결과 로드: train {self.train_vectorized.shape}, test {self.test_vectorized.shape}")
            return True
        print(f"⚠️ {method} 벡터화 결과 없음")
        return False
    # eof -----------------------------------

# End of class ###########################


# 사용 예시
if __name__ == "__main__":
    print("=" * 60)
    print("Mercari Price Suggestion - v7 최종")
    print("=" * 60)
  
    analyzer = MercariPyCaretAnalyzer()
    analyzer.load_data(undersample_frac=0.35)
    analyzer.vectorize_text(method="tfidf", max_features_name=15000, max_features_desc=20000, n_components=150)
    analyzer.setup_pycaret(fold=3)
    analyzer.find_and_blend_models(use_kaggle_winners=True)
    analyzer.save_metrics()
    analyzer.predict_test()
  
    print("\n✅ 완료!")
```
