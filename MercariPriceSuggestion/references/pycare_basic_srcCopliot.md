``` python
import pandas as pd
import numpy as np
import os
import json
import datetime
import gc
from tqdm import tqdm
import warnings
import re

warnings.filterwarnings("ignore")

from pycaret.regression import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class MercariPyCaretAnalyzer:
    def __init__(self,
                 data_dir="../data",
                 images_dir="../images",
                 results_dir="../results"):
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

    # 희귀값 통합
    def _collapse_rare_values(self, col, top_k, rare_label="Other"):
        combined = pd.concat([self.train[col], self.test[col]], axis=0)
        value_counts = combined.value_counts()
        top_values = value_counts.index[:top_k]

        self.train[col] = self.train[col].where(self.train[col].isin(top_values), rare_label)
        self.test[col] = self.test[col].where(self.test[col].isin(top_values), rare_label)

    # 텍스트 정규화
    def _simple_normalize(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[_\-\./]", " ", text)
        text = re.sub(r"\d+", " num ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # 층화 언더샘플링
    def _stratified_sample(self, frac=0.3, bins=10):
        """
        price 로그 변환된 데이터를 구간별로 나눠 층화 샘플링
        frac: 전체 데이터 중 몇 %를 샘플링할지
        bins: 가격 구간 개수
        """
        # 로그 변환된 price 기준으로 구간 나누기
        self.train["price_bin"] = pd.qcut(self.train["price"], q=bins, duplicates="drop")
        sampled = self.train.groupby("price_bin", group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=23)
        )
        self.train = sampled.drop(columns=["price_bin"])
        print(f"⚠️ Stratified undersampling 적용: train {self.train.shape}")

    # 데이터 로딩
    def load_data(self, train_file="train.tsv", test_file="test.tsv", sep="\t", undersample_frac=0.3):
        print("📂 데이터 로딩 시작...")
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)

        self.train = pd.read_csv(train_path, sep=sep)
        self.test = pd.read_csv(test_path, sep=sep)

        # price 로그 변환
        self.train = self.train[self.train["price"] > 0].dropna(subset=["price"])
        self.train["price"] = np.log1p(self.train["price"])

        # 층화 언더샘플링 적용
        if undersample_frac is not None:
            self._stratified_sample(frac=undersample_frac)

        # category split + 결측치 처리
        for df_name, df in [("train", self.train), ("test", self.test)]:
            df["main_cat"], df["sub_cat"], df["sub_sub_cat"] = zip(
                *df["category_name"].apply(
                    lambda x: (x.split("/") if isinstance(x, str) and "/" in x else ["missing"]*3)
                )
            )
            df["brand_name"] = df["brand_name"].fillna("Unknown").astype(str)
            df["category_name"] = df["category_name"].fillna("Unknown").astype(str)
            df["item_description"] = df["item_description"].fillna("No description").astype(str)
            df["name"] = df["name"].fillna("No name").astype(str)

        # 희귀값 통합
        self._collapse_rare_values("brand_name", top_k=4500, rare_label="Other_brand")
        self._collapse_rare_values("main_cat", top_k=1000, rare_label="Other_main")
        self._collapse_rare_values("sub_cat", top_k=1000, rare_label="Other_sub")
        self._collapse_rare_values("sub_sub_cat", top_k=1000, rare_label="Other_sub_sub")

        # 길이 피처 추가
        for df in [self.train, self.test]:
            df["name_len_char"] = df["name"].str.len()
            df["name_len_word"] = df["name"].str.split().str.len()
            df["desc_len_char"] = df["item_description"].str.len()
            df["desc_len_word"] = df["item_description"].str.split().str.len()

        for df in [self.train, self.test]:
            df["shipping"] = df["shipping"].astype("category")
            df["item_condition_id"] = df["item_condition_id"].astype("category")

        print(f"✅ 데이터 로드 완료: train {self.train.shape}, test {self.test.shape}")

    # 텍스트 벡터화
    def vectorize_text(self, text_columns=["name", "item_description"],
                       method="tfidf", max_features=30000, n_components=100):
        print("📝 텍스트 벡터화 및 차원 축소 시작...")
        for col in text_columns:
            clean_col = f"{col}_clean"
            self.train[clean_col] = self.train[col].apply(self._simple_normalize)
            self.test[clean_col] = self.test[col].apply(self._simple_normalize)

        vectors, feature_names = [], []
        for col in tqdm(text_columns, desc="Text columns"):
            clean_col = f"{col}_clean"
            vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2)) if method=="tfidf" else CountVectorizer(max_features=max_features, ngram_range=(1,2))
            combined_text = pd.concat([self.train[clean_col], self.test[clean_col]], axis=0)
            vec.fit(combined_text)

            train_vec = vec.transform(self.train[clean_col])
            test_vec = vec.transform(self.test[clean_col])

            if n_components < train_vec.shape[1]:
                svd = TruncatedSVD(n_components=n_components, random_state=23)
                train_vec = svd.fit_transform(train_vec)
                test_vec = svd.transform(test_vec)
            else:
                train_vec = train_vec.toarray()
                test_vec = test_vec.toarray()

            vectors.append((train_vec, test_vec))
            feature_names.append([f"{col}_{i}" for i in range(train_vec.shape[1])])
            gc.collect()

        train_features = np.hstack([v[0] for v in vectors])
        test_features = np.hstack([v[1] for v in vectors])

        self.train_vectorized = pd.DataFrame(train_features, columns=[f for sub in feature_names for f in sub])
        self.test_vectorized = pd.DataFrame(test_features, columns=[f for sub in feature_names for f in sub])

        for col in ["main_cat","sub_cat","sub_sub_cat","brand_name","item_condition_id","shipping",
                    "name_len_char","name_len_word","desc_len_char","desc_len_word"]:
            self.train_vectorized[col] = self.train[col].reset_index(drop=True)
            self.test_vectorized[col] = self.test[col].reset_index(drop=True)

        print(f"✅ 벡터화 완료: train {self.train_vectorized.shape}, test {self.test_vectorized.shape}")

    # PyCaret setup
    def setup_pycaret(self, session_id=23):
        print("🔧 PyCaret setup 시작...")
        categorical_cols = ["main_cat","sub_cat","sub_sub_cat","brand_name","item_condition_id","shipping"]
        existing_categorical = [col for col in categorical_cols if col in self.train_vectorized.columns]

        self.setup_result = setup(
            data=self.train_vectorized.assign(price=self.train["price"].reset_index(drop=True)),
            target="price",
            session_id=session_id,
            categorical_features=existing_categorical if existing_categorical else None,
            normalize=True,
            transformation=False,
            verbose=True,
        )
        print("✅ PyCaret setup 완료")

    # 상위권 모델 블렌딩
    def find_and_blend_models(self, sort_metric="R2"):
        if self.setup_result is None:
            raise ValueError("먼저 setup_pycaret()를 실행하세요.")

        print("🔍 상위권 모델 후보 학습 및 블렌딩 시작...")

        # 후보 모델 생성
        lgbm  = create_model("lightgbm")
        xgb   = create_model("xgboost")
        cat   = create_model("catboost")
        ridge = create_model("ridge")
        lr    = create_model("lr")

        # 블렌딩 (단순 평균 앙상블)
        blended = blend_models([lgbm, xgb, cat, ridge, lr], optimize=sort_metric)

        self.best_model = blended
        print(f"🏆 Blended model 선택 완료 (기준={sort_metric})")
        return self.best_model

    # 성능 저장
    def save_metrics(self, model_name=None):
        if self.best_model is None:
            raise ValueError("모델이 없습니다.")

        pred_df = predict_model(self.best_model, data=self.train_vectorized.copy())
        y_log_true = self.train["price"].values
        y_log_pred = pred_df["Label"].values

        # 로그 스케일 → 실제 가격 스케일 복원
        y_true = np.expm1(y_log_true)
        y_pred = np.expm1(y_log_pred)

        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)

        self.metrics = {"R2": round(r2,4), "RMSE": round(rmse,4), "MAE": round(mae,4)}

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name is None:
            model_name = str(self.best_model).split("(")[0]

        file_path = os.path.join(self.results_dir, f"{model_name}_metrics_{timestamp}.json")
        with open(file_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

        print(f"💾 Metrics 저장 완료: {file_path}")

    # 시각화
    def visualize_model(self, plots=["residuals", "feature"]):
        if self.best_model is None:
            raise ValueError("먼저 find_and_blend_models()로 모델을 선택하세요.")

        print("🎨 시각화 시작...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = str(self.best_model).split("(")[0]

        for p in plots:
            try:
                plot_name = "feature" if p == "feature_importance" else p
                save_path = os.path.join(self.images_dir, f"{model_name}_{plot_name}_{timestamp}.png")
                plot_model(self.best_model, plot=plot_name, save=True)
                print(f"✅ {plot_name} plot 저장 완료: {save_path}")
            except Exception as e:
                print(f"⚠️ Plot {p} 실패: {e}")

    # 테스트 예측 & 제출 파일 생성
    def predict_test(self, submission_file="submission.csv"):
        if self.best_model is None:
            raise ValueError("먼저 find_and_blend_models()로 모델을 선택하세요.")

        print("📦 Test 데이터 예측 시작...")
        predictions = predict_model(self.best_model, data=self.test_vectorized.copy())

        # 로그 스케일 → 실제 가격 복원
        price_log_pred = predictions["Label"].values
        price_pred = np.expm1(price_log_pred)

        submission = pd.DataFrame({"test_id": self.test["test_id"], "price": price_pred})

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = os.path.join(self.results_dir, f"{timestamp}_{submission_file}")
        submission.to_csv(submission_path, index=False)
        print(f"💾 Submission 저장 완료: {submission_path}")
        return submission    
    
# End of class ###############    
    
```        