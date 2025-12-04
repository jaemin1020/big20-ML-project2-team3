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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class MercariPyCaretAnalyzer5:
    """
    Mercari Price Suggestion - TF-IDF 개선 버전
    
    개선사항:
    - HashingVectorizer → TF-IDF (정보 손실 최소화)
    - SVD 차원 증가 (80 → 150)
    - 샘플링 비율 증가 (25% → 35%)
    - min_df/max_df로 노이즈 제거
    """
    
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

    def _collapse_rare_values(self, col, top_k, rare_label="Other"):
        """희귀값 통합"""
        combined = pd.concat([self.train[col], self.test[col]], axis=0)
        value_counts = combined.value_counts()
        top_values = set(value_counts.index[:top_k])
        
        del value_counts, combined
        gc.collect()

        self.train[col] = self.train[col].apply(lambda x: x if x in top_values else rare_label)
        self.test[col] = self.test[col].apply(lambda x: x if x in top_values else rare_label)

    def _simple_normalize(self, text: str) -> str:
        """텍스트 정규화"""
        text = str(text).lower()
        text = re.sub(r"[_\-\./]", " ", text)
        text = re.sub(r"\d+", " num ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _stratified_sample(self, frac=0.35, bins=10):
        """층화 샘플링 - 35%로 증가"""
        self.train["price_bin"] = pd.qcut(self.train["price"], q=bins, duplicates="drop")
        sampled = self.train.groupby("price_bin", group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=23)
        )
        self.train = sampled.drop(columns=["price_bin"]).reset_index(drop=True)
        gc.collect()
        print(f"⚠️ Stratified undersampling 적용: train {self.train.shape}")

    def load_data(self, train_file="train.tsv", test_file="test.tsv", sep="\t", undersample_frac=0.35):
        """데이터 로딩 - 샘플링 35%로 증가"""
        print("📂 데이터 로딩 시작...")
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)

        self.train = pd.read_csv(train_path, sep=sep)
        self.test = pd.read_csv(test_path, sep=sep)

        # price 로그 변환
        self.train = self.train[self.train["price"] > 0].dropna(subset=["price"])
        self.train["price"] = np.log1p(self.train["price"])

        # 층화 언더샘플링
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
            df["item_description"] = df["item_description"].fillna("No description").astype(str)
            df["name"] = df["name"].fillna("No name").astype(str)
            
            # category_name 삭제
            df.drop(columns=["category_name"], inplace=True)

        # 희귀값 통합
        print("🔄 희귀값 통합 중...")
        self._collapse_rare_values("brand_name", top_k=5000, rare_label="Other_brand")
        self._collapse_rare_values("main_cat", top_k=1000, rare_label="Other_main")
        self._collapse_rare_values("sub_cat", top_k=1000, rare_label="Other_sub")
        self._collapse_rare_values("sub_sub_cat", top_k=1000, rare_label="Other_sub_sub")

        # 길이 피처
        for df in [self.train, self.test]:
            df["name_len_char"] = df["name"].str.len()
            df["name_len_word"] = df["name"].str.split().str.len()
            df["desc_len_char"] = df["item_description"].str.len()
            df["desc_len_word"] = df["item_description"].str.split().str.len()
            
            # 추가 피처: 브랜드 언급 여부
            df["has_brand_in_name"] = df.apply(
                lambda row: 1 if row["brand_name"].lower() in row["name"].lower() else 0, 
                axis=1
            )
            df["has_brand_in_desc"] = df.apply(
                lambda row: 1 if row["brand_name"].lower() in row["item_description"].lower() else 0,
                axis=1
            )

        # 범주형 변환
        for df in [self.train, self.test]:
            df["shipping"] = df["shipping"].astype("category")
            df["item_condition_id"] = df["item_condition_id"].astype("category")

        gc.collect()
        print(f"✅ 데이터 로드 완료: train {self.train.shape}, test {self.test.shape}")

    def vectorize_text_tfidf(self, 
                            text_columns=["name", "item_description"],
                            max_features_name=15000,
                            max_features_desc=20000,
                            n_components=150):
        """
        TF-IDF 벡터화 (개선 버전)
        
        개선사항:
        - name과 description에 다른 max_features 적용
        - min_df/max_df로 노이즈 제거
        - SVD 차원 증가 (80 → 150)
        """
        print(f"🔍 TF-IDF 벡터화 시작...")
        print(f"   - name: max_features={max_features_name:,}")
        print(f"   - description: max_features={max_features_desc:,}")
        print(f"   - SVD 압축: {n_components}차원")
        
        all_train_features = []
        all_test_features = []
        feature_names = []
        
        for col in tqdm(text_columns, desc="텍스트 컬럼 처리"):
            clean_col = f"{col}_clean"
            
            # 텍스트 정규화
            self.train[clean_col] = self.train[col].apply(self._simple_normalize)
            self.test[clean_col] = self.test[col].apply(self._simple_normalize)
            
            # 컬럼별 max_features 설정
            if col == "name":
                max_feat = max_features_name
            elif col == "item_description":
                max_feat = max_features_desc
            else:
                max_feat = 10000
            
            # TF-IDF 벡터화
            tfidf = TfidfVectorizer(
                max_features=max_feat,
                ngram_range=(1, 2),
                min_df=3,      # 최소 3번 이상 등장
                max_df=0.95,   # 95% 이상 문서에 등장하는 단어 제거
                sublinear_tf=True,  # log scaling
                dtype=np.float32
            )
            
            # fit on combined
            combined_text = pd.concat([self.train[clean_col], self.test[clean_col]], axis=0)
            tfidf.fit(combined_text)
            del combined_text
            gc.collect()
            
            train_vec = tfidf.transform(self.train[clean_col])
            test_vec = tfidf.transform(self.test[clean_col])
            
            print(f"   - {col}: sparse matrix shape = {train_vec.shape}")
            
            # SVD 차원 축소
            actual_components = min(n_components, train_vec.shape[1] - 1)
            svd = TruncatedSVD(n_components=actual_components, random_state=23)
            train_vec_dense = svd.fit_transform(train_vec)
            test_vec_dense = svd.transform(test_vec)
            
            explained_var = svd.explained_variance_ratio_.sum()
            print(f"   - {col}: SVD {actual_components}차원, 설명력 = {explained_var:.2%}")
            
            # 메모리 정리
            del train_vec, test_vec, tfidf, svd
            gc.collect()
            
            all_train_features.append(train_vec_dense)
            all_test_features.append(test_vec_dense)
            feature_names.extend([f"{col}_tfidf_{i}" for i in range(actual_components)])
            
            # clean 컬럼 삭제
            self.train.drop(columns=[clean_col], inplace=True)
            self.test.drop(columns=[clean_col], inplace=True)
            gc.collect()
        
        # 피처 결합
        print("🔗 피처 결합 중...")
        train_features = np.hstack(all_train_features).astype(np.float32)
        test_features = np.hstack(all_test_features).astype(np.float32)
        
        del all_train_features, all_test_features
        gc.collect()
        
        self.train_vectorized = pd.DataFrame(train_features, columns=feature_names)
        self.test_vectorized = pd.DataFrame(test_features, columns=feature_names)
        
        del train_features, test_features
        gc.collect()
        
        # 추가 피처 병합
        categorical_features = [
            "main_cat", "sub_cat", "sub_sub_cat", "brand_name",
            "item_condition_id", "shipping"
        ]
        numeric_features = [
            "name_len_char", "name_len_word", "desc_len_char", "desc_len_word",
            "has_brand_in_name", "has_brand_in_desc"
        ]
        
        for col in categorical_features + numeric_features:
            if col in self.train.columns:
                self.train_vectorized[col] = self.train[col].reset_index(drop=True)
                self.test_vectorized[col] = self.test[col].reset_index(drop=True)
        
        gc.collect()
        
        print(f"✅ 벡터화 완료!")
        print(f"   - train: {self.train_vectorized.shape}")
        print(f"   - test: {self.test_vectorized.shape}")
        print(f"   - 최종 피처 수: {self.train_vectorized.shape[1]}")

    def setup_pycaret(self, session_id=23, fold=3, use_gpu=False):
        """PyCaret 환경 설정"""
        print("🔧 PyCaret setup 시작...")
        
        categorical_cols = [
            "main_cat", "sub_cat", "sub_sub_cat", 
            "brand_name", "item_condition_id", "shipping"
        ]
        existing_categorical = [col for col in categorical_cols 
                               if col in self.train_vectorized.columns]

        self.setup_result = setup(
            data=self.train_vectorized.assign(
                price=self.train["price"].reset_index(drop=True)
            ),
            target="price",
            session_id=session_id,
            categorical_features=existing_categorical if existing_categorical else None,
            normalize=True,
            transformation=False,
            fold_strategy="kfold",
            fold=fold,
            use_gpu=use_gpu,
            n_jobs=4,
            verbose=True,
            html=False
        )
        
        gc.collect()
        print("✅ PyCaret setup 완료")

    def find_and_blend_models(self, top_n=3, sort_metric="R2", use_kaggle_winners=True):
        """
        상위권 모델 탐색 및 블렌딩
        
        use_kaggle_winners=True: Mercari 대회 상위권 모델만 사용 (빠름)
        use_kaggle_winners=False: 모든 모델 비교 (느림)
        """
        if self.setup_result is None:
            raise ValueError("먼저 setup_pycaret()를 실행하세요.")

        if use_kaggle_winners:
            print("🏆 Mercari Kaggle 상위권 모델만 학습 (5개)")
            print("   - LightGBM: 빠르고 효율적")
            print("   - Ridge: 우승자들이 선호한 단순 모델")
            print("   - CatBoost: 범주형 데이터에 강함")
            print("   - XGBoost: 안정적인 부스팅")
            print("   - Extra Trees: 빠른 앙상블")
            
            # Mercari 상위권 모델만 학습
            lgbm = create_model("lightgbm", verbose=False)
            ridge = create_model("ridge", verbose=False)
            cat = create_model("catboost", verbose=False)
            xgb = create_model("xgboost", verbose=False)
            et = create_model("et", verbose=False)
            
            top_models = [lgbm, ridge, cat, xgb, et]
            
            print("✅ 5개 모델 학습 완료")
            
        else:
            print(f"🔍 전체 모델 탐색 시작 (시간 오래 걸림)...")
            top_models = compare_models(
                n_select=top_n,
                sort=sort_metric,
                turbo=True,
                verbose=True
            )
            
            if not isinstance(top_models, list):
                top_models = [top_models]
        
        print("🎯 선정된 모델:")
        for i, model in enumerate(top_models, 1):
            print(f"   {i}. {str(model).split('(')[0]}")
        
        print(f"\n🔀 {len(top_models)}개 모델 블렌딩 시작...")
        blended = blend_models(
            estimator_list=top_models,
            optimize=sort_metric,
            choose_better=True,
            verbose=True
        )
        
        self.best_model = blended
        gc.collect()
        
        print(f"🏆 Blended model 생성 완료 (기준={sort_metric})")
        return self.best_model

    def tune_best_model(self, n_iter=50, optimize_metric="R2"):
        """
        베스트 모델 추가 튜닝
        R² < 0.70 일 때 권장
        """
        if self.best_model is None:
            raise ValueError("먼저 find_and_blend_models()를 실행하세요.")
        
        print(f"⚙️ 모델 튜닝 시작 (n_iter={n_iter})...")
        
        tuned_model = tune_model(
            self.best_model,
            optimize=optimize_metric,
            n_iter=n_iter,
            search_library='optuna',
            search_algorithm='tpe'
        )
        
        self.best_model = tuned_model
        print("✅ 튜닝 완료!")
        return tuned_model

    def save_metrics(self, model_name=None):
        """성능 지표 저장"""
        if self.best_model is None:
            raise ValueError("모델이 없습니다.")

        print("📊 성능 평가 중...")
        pred_df = predict_model(self.best_model, data=self.train_vectorized.copy())
        
        y_log_true = self.train["price"].values
        y_log_pred = pred_df["prediction_label"].values

        # 실제 가격 스케일 복원
        y_true = np.expm1(y_log_true)
        y_pred = np.expm1(y_log_pred)

        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)

        self.metrics = {
            "R2": round(r2, 4), 
            "RMSE": round(rmse, 4), 
            "MAE": round(mae, 4)
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name is None:
            model_name = str(self.best_model).split("(")[0]

        file_path = os.path.join(self.results_dir, f"{model_name}_metrics_{timestamp}.json")
        with open(file_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

        print(f"💾 Metrics 저장 완료: {file_path}")
        print(f"   - R² = {self.metrics['R2']}")
        print(f"   - RMSE = ${self.metrics['RMSE']:.2f}")
        print(f"   - MAE = ${self.metrics['MAE']:.2f}")

    def visualize_model(self, plots=["residuals", "feature"]):
        """모델 시각화"""
        if self.best_model is None:
            raise ValueError("먼저 find_and_blend_models()로 모델을 선택하세요.")

        print("🎨 시각화 시작...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = str(self.best_model).split("(")[0]

        for p in plots:
            try:
                plot_name = "feature" if p == "feature_importance" else p
                save_path = os.path.join(
                    self.images_dir, 
                    f"{model_name}_{plot_name}_{timestamp}.png"
                )
                plot_model(self.best_model, plot=plot_name, save=True)
                print(f"✅ {plot_name} plot 저장 완료")
            except Exception as e:
                print(f"⚠️ Plot {p} 실패: {e}")

    def predict_test(self, submission_file="submission.csv"):
        """테스트 예측 및 제출 파일 생성"""
        if self.best_model is None:
            raise ValueError("먼저 find_and_blend_models()로 모델을 선택하세요.")

        print("📦 Test 데이터 예측 시작...")
        predictions = predict_model(self.best_model, data=self.test_vectorized.copy())

        # 실제 가격 복원
        price_log_pred = predictions["prediction_label"].values
        price_pred = np.expm1(price_log_pred)

        submission = pd.DataFrame({
            "test_id": self.test["test_id"], 
            "price": price_pred
        })

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = os.path.join(
            self.results_dir, 
            f"{timestamp}_{submission_file}"
        )
        submission.to_csv(submission_path, index=False)
        
        print(f"💾 Submission 저장 완료: {submission_path}")
        print(f"   - 예측 가격 범위: ${price_pred.min():.2f} ~ ${price_pred.max():.2f}")
        print(f"   - 평균 가격: ${price_pred.mean():.2f}")
        
        return submission

# End of Class ####################################################################################################
```