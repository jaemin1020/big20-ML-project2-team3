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
from gensim.models import FastText
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

from pycaret.regression import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class MercariPyCaretAnalyzer7:
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

    # vectorize_text start ##################################################################    
    def vectorize_text(self, 
                    method="tfidf", 
                    text_columns=["name", "item_description"],
                    max_features_name=15000,
                    max_features_desc=20000,
                    n_components=150,
                    fasttext_size=100,
                    fasttext_window=5,
                    fasttext_min_count=2,
                    bert_model_name="all-MiniLM-L6-v2"):
        """
        텍스트 데이터를 다양한 방식으로 벡터화합니다.
        매개변수
        ----------
        method : str, 기본값="tfidf"
            사용할 벡터화 방법을 지정합니다.
            선택 가능 옵션:
            - "tfidf"    : TF-IDF 기반 벡터화 후 SVD 차원 축소
            - "fasttext" : FastText 단어 임베딩을 문서 단위 평균으로 변환
            - "bert"     : 사전학습된 BERT 문장 임베딩 (sentence-transformers 활용)

        text_columns : list of str, 기본값=["name", "item_description"]
            벡터화할 텍스트 컬럼 이름 리스트.

        max_features_name : int, 기본값=15000
            TF-IDF에서 'name' 컬럼에 사용할 최대 피처 수.

        max_features_desc : int, 기본값=20000
            TF-IDF에서 'item_description' 컬럼에 사용할 최대 피처 수.

        n_components : int, 기본값=150
            TF-IDF 결과를 SVD로 축소할 차원 수.

        fasttext_size : int, 기본값=100
            FastText 단어 벡터의 차원 크기.

        fasttext_window : int, 기본값=5
            FastText 학습 시 윈도우 크기.

        fasttext_min_count : int, 기본값=2
            FastText 학습 시 최소 단어 등장 횟수.

        bert_model_name : str, 기본값="all-MiniLM-L6-v2"
            BERT 임베딩에 사용할 sentence-transformers 모델 이름.

        반환값
        -------
        None
            내부적으로 `self.train_vectorized`와 `self.test_vectorized`에
            벡터화된 학습/테스트 데이터를 DataFrame 형태로 저장합니다.

        사용 예시
        --------
        # 기본 TF-IDF 벡터화
        analyzer.vectorize_text()

        # FastText 벡터화
        analyzer.vectorize_text(method="fasttext", fasttext_size=200)

        # BERT 벡터화
        analyzer.vectorize_text(method="bert", bert_model_name="all-MiniLM-L12-v2")

        참고
        -----
        - 한 번에 하나의 벡터화 방법만 적용됩니다.
        - TF-IDF는 빠르고 메모리 효율적입니다.
        - FastText는 희귀 단어와 오타 처리에 강점이 있습니다.
        - BERT는 문맥과 의미를 잘 반영하지만 메모리 사용량이 큽니다.

        """

        if method == "tfidf":
            print("🔍 TF-IDF 벡터화 시작...")
            # 기존 TF-IDF 코드 그대로 사용
            # ...
            # self.train_vectorized, self.test_vectorized 생성

        elif method == "fasttext":
            print("🔍 FastText 벡터화 시작...")
            from gensim.models import FastText
            sentences = []
            for col in text_columns:
                self.train[col] = self.train[col].fillna("").astype(str)
                self.test[col] = self.test[col].fillna("").astype(str)
                sentences += [str(x).split() for x in pd.concat([self.train[col], self.test[col]])]

            ft_model = FastText(sentences, vector_size=fasttext_size, window=fasttext_window, min_count=fasttext_min_count, sg=1)

            def get_vector(text):
                words = text.split()
                vectors = [ft_model.wv[w] for w in words if w in ft_model.wv]
                return np.mean(vectors, axis=0) if vectors else np.zeros(fasttext_size)

            train_features, test_features = [], []
            for col in text_columns:
                train_features.append(np.vstack(self.train[col].apply(get_vector)))
                test_features.append(np.vstack(self.test[col].apply(get_vector)))

            train_vec = np.hstack(train_features)
            test_vec = np.hstack(test_features)

            self.train_vectorized = pd.DataFrame(train_vec)
            self.test_vectorized = pd.DataFrame(test_vec)

        elif method == "bert":
            print(f"🔍 BERT 임베딩 시작... (모델={bert_model_name})")
            from sentence_transformers import SentenceTransformer
            bert_model = SentenceTransformer(bert_model_name)

            train_features, test_features = [], []
            for col in text_columns:
                self.train[col] = self.train[col].fillna("").astype(str)
                self.test[col] = self.test[col].fillna("").astype(str)

                train_emb = bert_model.encode(self.train[col].tolist(), show_progress_bar=True)
                test_emb = bert_model.encode(self.test[col].tolist(), show_progress_bar=True)

                train_features.append(train_emb)
                test_features.append(test_emb)

            train_vec = np.hstack(train_features)
            test_vec = np.hstack(test_features)

            self.train_vectorized = pd.DataFrame(train_vec)
            self.test_vectorized = pd.DataFrame(test_vec)

        else:
            raise ValueError("method must be one of ['tfidf', 'fasttext', 'bert']")

        print(f"✅ {method} 벡터화 완료: train {self.train_vectorized.shape}, test {self.test_vectorized.shape}")
    # eof --------------------------------------------------------------------------------------------#
    
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

    def find_and_blend_models(self, top_n=3, sort_metric="R2", use_kaggle_winners=True, use_tqdm=True, include_autogluon=True):
        """
        상위권 모델 탐색 및 블렌딩 (확장 버전)
        - use_kaggle_winners=True: Mercari 대회 상위권 모델 포함
        - include_autogluon=True: AutoGluon에서 찾은 조합 모델 포함
        - use_tqdm=True: tqdm 진행바 표시 (기본값 True)
        """
        if self.setup_result is None:
            raise ValueError("먼저 setup_pycaret()를 실행하세요.")

        if use_kaggle_winners:
            print("🏆 Mercari Kaggle 상위권 모델 학습")

            # 기본 Kaggle 상위권 모델
            model_names = ["lightgbm", "ridge", "catboost", "xgboost", "et"]

            # AutoGluon 조합 확장
            if include_autogluon:
                print("➕ AutoGluon 조합 모델 추가")
                model_names.extend(["lightgbm", "rf"])  # LightGBMXT → lightgbm, RandomForestMSE → rf

            top_models = []
            if use_tqdm:
                from tqdm import tqdm
                for name in tqdm(model_names, desc="모델 학습 진행"):
                    model = create_model(name, verbose=False)
                    top_models.append(model)
            else:
                for name in model_names:
                    model = create_model(name, verbose=False)
                    top_models.append(model)

            print(f"✅ {len(top_models)}개 모델 학습 완료")

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
    # eof --------------------------------------------------------------------------------------
        
    def compare_optuna_vs_hyperopt(self, n_iter=50, optimize_metric="R2"):
        """
        Optuna vs Hyperopt 튜닝 결과 자동 비교
        - 두 가지 튜닝을 실행하고 성능 지표를 비교하여 더 좋은 모델을 선택
        """
        if self.best_model is None:
            raise ValueError("먼저 find_and_blend_models()를 실행하세요.")

        print("⚖️ Optuna 튜닝 시작...")
        optuna_model = tune_model(
            self.best_model,
            optimize=optimize_metric,
            n_iter=n_iter,
            search_library='optuna',
            search_algorithm='tpe'
        )

        print("⚖️ Hyperopt 튜닝 시작...")
        hyperopt_model = tune_model(
            self.best_model,
            optimize=optimize_metric,
            n_iter=n_iter,
            search_library='hyperopt',
            search_algorithm='tpe'
        )

        # 두 모델 성능 비교
        optuna_pred = predict_model(optuna_model, data=self.train_vectorized.copy())
        hyperopt_pred = predict_model(hyperopt_model, data=self.train_vectorized.copy())

        y_true = np.expm1(self.train["price"].values)
        y_optuna = np.expm1(optuna_pred["prediction_label"].values)
        y_hyperopt = np.expm1(hyperopt_pred["prediction_label"].values)

        r2_optuna = r2_score(y_true, y_optuna)
        r2_hyperopt = r2_score(y_true, y_hyperopt)

        print(f"📊 Optuna R² = {r2_optuna:.4f}")
        print(f"📊 Hyperopt R² = {r2_hyperopt:.4f}")

        # 더 좋은 모델 선택
        if r2_optuna >= r2_hyperopt:
            self.best_model = optuna_model
            print("✅ Optuna 모델 선택!")
        else:
            self.best_model = hyperopt_model
            print("✅ Hyperopt 모델 선택!")

        return self.best_model    

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
    # eof -----------------------------------------------------------  
      


# End of Class ####################################################################################################
```