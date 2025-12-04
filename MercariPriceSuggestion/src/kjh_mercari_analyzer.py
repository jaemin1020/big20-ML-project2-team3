import pandas as pd
import numpy as np
import os
import json
import datetime
import gc
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from pycaret.regression import *

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class MercariPyCaretAnalyzer:
    """
    Mercari Price Suggestion Challenge용 PyCaret 분석기
    - TSV/CSV 데이터 로딩
    - TF-IDF / CountVectorizer + TruncatedSVD 차원 축소
    - 카테고리/브랜드/배송정보 인코딩 후 피처 결합
    - PyCaret setup & compare_models
    - Base model 생성, 시각화, 예측
    - Metrics JSON 저장, submission CSV 저장
    """

    def __init__(
        self,
        data_dir="../data",
        images_dir="../images",
        results_dir="../results",
        use_gpu=True,
    ):
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.results_dir = results_dir
        self.use_gpu = use_gpu

        self.train = None
        self.test = None
        self.best_model = None
        self.setup_result = None
        self.metrics = {}

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # GPU 가용성 체크
        self._check_gpu_availability()

    # ------------------ 데이터 로딩 ------------------
    def load_data(self, train_file="train.tsv", test_file="test.tsv", sep="\t"):
        print("📂 데이터 로딩 시작...")
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)

        self.train = pd.read_csv(train_path, sep=sep)
        self.test = pd.read_csv(test_path, sep=sep)

        print(f"original data shape : train {self.train.shape}, test {self.test.shape}")
        print("✅ Price 외 결측치 처리 및 데이터 전처리 시작...")

        # price 0 제거 + NaN 제거 (train만 해당)
        self.train = self.train[self.train["price"] > 0].dropna(subset=["price"])
        print("Price NaN count:", self.train["price"].isna().sum())

        # 카테고리 대/중/소 분류 함수
        def _split_cat(x):
            if isinstance(x, str):
                parts = x.split("/", 2)  # 최대 3조각
            else:
                parts = []
            # 길이가 3이 되도록 채우기
            while len(parts) < 3:
                parts.append("missing")
            return parts[:3]

        # train과 test 모두 처리
        for df_name, df in [("train", self.train), ("test", self.test)]:
            # category_name 분리 (대/중/소)
            df["main_cat"], df["sub_cat"], df["sub_sub_cat"] = zip(
                *df["category_name"].apply(_split_cat)
            )

            # 결측치 처리 + 타입 통일
            df["brand_name"] = df["brand_name"].fillna("Unknown").astype(str)
            df["category_name"] = df["category_name"].fillna("Unknown").astype(str)
            df["item_description"] = (
                df["item_description"].fillna("No description").astype(str)
            )
            df["name"] = df["name"].fillna("No name").astype(str)

            # 인덱스 리셋
            if df_name == "train":
                self.train = df.reset_index(drop=True)
            else:
                self.test = df.reset_index(drop=True)

        # price log 변환 (train만) - log1p(price)
        self.train["price"] = np.log1p(self.train["price"])

        print("Final Price NaN count:", self.train["price"].isna().sum())
        print("Train length:", len(self.train))
        print("\nTrain head:")
        print(self.train.head())

        print(f"\nTrain info:\n{'='*50}")
        print(self.train.info())

        print(
            f"\n✅ 데이터 로드 완료: train {self.train.shape}, test {self.test.shape}"
        )
      # ------------------ gpu 사용가능 여부 확인 ------------------
    def _check_gpu_availability(self):
        """GPU 사용 가능 여부 확인"""
        if not self.use_gpu:
            print("🖥️  CPU 모드로 실행")
            return

        print("🔍 GPU 가용성 체크 중...")
        gpu_available = False

        # CUDA 체크
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
                print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                gpu_available = True
        except ImportError:
            pass

        # XGBoost GPU 체크
        try:
            import xgboost as xgb
            print(f"✅ XGBoost 버전: {xgb.__version__}")
        except ImportError:
            print("⚠️  XGBoost 미설치")

        # LightGBM GPU 체크
        try:
            import lightgbm as lgb
            print(f"✅ LightGBM 버전: {lgb.__version__}")
        except ImportError:
            print("⚠️  LightGBM 미설치")

        # CatBoost 체크
        try:
            import catboost
            print(f"✅ CatBoost 버전: {catboost.__version__}")
        except ImportError:
            print("⚠️  CatBoost 미설치")

        if not gpu_available:
            print("⚠️  GPU를 찾을 수 없습니다. CPU 모드로 전환합니다.")
            self.use_gpu = False
        else:
            print("🚀 GPU 가속 활성화")
    # ------------------ Undersampling 기능 ------------------
    def apply_undersampling(
        self,
        method="random",
        target_size=None,
        sampling_ratio=0.5,
        n_bins=10,
        random_state=23,
    ):
        """
        가격 데이터에 대한 undersampling 적용

        Parameters:
        -----------
        method : str
            'random' : 무작위 샘플링
            'stratified' : 가격 구간별 층화 샘플링
            'top_n' : 상위 N개만 선택
        target_size : int, optional
            목표 샘플 수 (None이면 sampling_ratio 사용)
        sampling_ratio : float
            샘플링 비율 (0 < ratio <= 1)
        n_bins : int
            stratified 방식 사용 시 가격 구간 수
        random_state : int
            재현성을 위한 random seed
        """
        if self.train is None:
            raise ValueError("먼저 load_data()를 실행하세요.")

        original_size = len(self.train)

        # target_size 결정
        if target_size is None:
            target_size = int(original_size * sampling_ratio)
        else:
            target_size = min(target_size, original_size)

        print(f"\n🎯 Undersampling 시작...")
        print(f"   방법: {method}")
        print(f"   원본 크기: {original_size:,}")
        print(f"   목표 크기: {target_size:,}")
        print(f"   샘플링 비율: {target_size/original_size:.2%}")

        if method == "random":
            # 무작위 샘플링
            sampled_indices = np.random.RandomState(random_state).choice(
                self.train.index, size=target_size, replace=False
            )
            self.train = self.train.loc[sampled_indices].reset_index(drop=True)

        elif method == "stratified":
            # 가격 구간별 층화 샘플링
            self.train["price_bin"] = pd.qcut(
                self.train["price"], q=n_bins, labels=False, duplicates="drop"
            )

            # 각 구간에서 동일 비율로 샘플링
            sampled = self.train.groupby("price_bin", group_keys=False).apply(
                lambda x: x.sample(
                    n=max(1, int(len(x) * sampling_ratio)), random_state=random_state
                )
            )

            self.train = sampled.drop(columns=["price_bin"]).reset_index(drop=True)

        elif method == "top_n":
            # 상위 N개만 선택 (정렬 기준은 price)
            self.train = self.train.nlargest(target_size, "price").reset_index(
                drop=True
            )

        else:
            raise ValueError(f"지원하지 않는 method: {method}")

        final_size = len(self.train)
        print(f"✅ Undersampling 완료")
        print(f"   최종 크기: {final_size:,}")
        print(f"   실제 샘플링 비율: {final_size/original_size:.2%}")
        print(f"   제거된 샘플: {original_size - final_size:,}")

        # 가격 분포 통계 출력
        print(f"\n📊 샘플링 후 가격 분포:")
        print(f"   평균: {self.train['price'].mean():.4f}")
        print(f"   중앙값: {self.train['price'].median():.4f}")
        print(f"   표준편차: {self.train['price'].std():.4f}")
        print(f"   최소: {self.train['price'].min():.4f}")
        print(f"   최대: {self.train['price'].max():.4f}")

    # ------------------ TF-IDF / CountVectorizer + 차원 축소 ------------------
    def vectorize_text(
        self,
        text_columns=["name", "item_description"],
        method="tfidf",
        max_features=50000,
        n_components=100,
    ):
        """
        text_columns: list of columns to vectorize
        method: 'tfidf' or 'count'
        max_features: Vectorizer max features
        n_components: TruncatedSVD components
        """
        print("📝 텍스트 벡터화 및 차원 축소 시작...")
        vectors = []
        feature_names = []

        for col in tqdm(text_columns, desc="Text columns"):
            print(f"▶ 컬럼: {col}")
            if method == "tfidf":
                vec = TfidfVectorizer(max_features=max_features)
            elif method == "count":
                vec = CountVectorizer(max_features=max_features)
            else:
                raise ValueError("method must be 'tfidf' or 'count'")

            combined_text = pd.concat(
                [self.train[col].astype(str), self.test[col].astype(str)], axis=0
            )
            vec.fit(combined_text)

            train_vec = vec.transform(self.train[col].astype(str))
            test_vec = vec.transform(self.test[col].astype(str))

            # 차원 축소
            if n_components < train_vec.shape[1]:
                svd = TruncatedSVD(n_components=n_components, random_state=23)
                train_vec = svd.fit_transform(train_vec)
                test_vec = svd.transform(test_vec)
                print(f"   ▪ 차원 축소 완료: {train_vec.shape[1]} components")
            else:
                train_vec = train_vec.toarray()
                test_vec = test_vec.toarray()

            vectors.append((train_vec, test_vec))
            feature_names.append([f"{col}_{i}" for i in range(train_vec.shape[1])])

            # 메모리 해제
            del combined_text, vec
            gc.collect()

        # 텍스트 피처 합치기
        train_features = np.hstack([v[0] for v in vectors])
        test_features = np.hstack([v[1] for v in vectors])

        self.train_vectorized = pd.DataFrame(
            train_features, columns=[f for sub in feature_names for f in sub]
        )
        self.test_vectorized = pd.DataFrame(
            test_features, columns=[f for sub in feature_names for f in sub]
        )

        # ✅ 카테고리/브랜드/배송 변수 인코딩 및 추가
        categorical_cols = [
            "main_cat",
            "sub_cat",
            "sub_sub_cat",
            "brand_name",
            "shipping",
        ]

        for col in categorical_cols:
            if col in self.train.columns:
                le = LabelEncoder()
                combined = pd.concat(
                    [self.train[col].astype(str), self.test[col].astype(str)], axis=0
                )
                le.fit(combined)

                self.train_vectorized[col] = le.transform(
                    self.train[col].astype(str).reset_index(drop=True)
                )
                self.test_vectorized[col] = le.transform(
                    self.test[col].astype(str).reset_index(drop=True)
                )

                del combined

        print(
            f"✅ 벡터화 + 차원 축소 + 카테고리 인코딩 완료: "
            f"train {self.train_vectorized.shape}, test {self.test_vectorized.shape}"
        )

    # ------------------ PyCaret setup ------------------
    def setup_pycaret(self, session_id=23, experiment_name='mercari_gpu'):
        if not hasattr(self, "train_vectorized"):
            raise ValueError("먼저 vectorize_text()를 실행하세요.")

        print("🔧 PyCaret setup 시작...")

        categorical_cols = [
            "main_cat",
            "sub_cat",
            "sub_sub_cat",
            "brand_name",
            "shipping",
        ]
        existing_categorical = [
            col for col in categorical_cols if col in self.train_vectorized.columns
        ]

        # 이미 price를 log1p로 변환해둔 상태라 transformation=False로 설정
        # GPU 사용 설정
        setup_params = {
            'data': self.train_vectorized.assign(
                price=self.train["price"].reset_index(drop=True)
            ),
            'target': 'price',
            'session_id': session_id,
            'experiment_name': experiment_name,
            'categorical_features': existing_categorical if existing_categorical else None,
            'normalize': True,
            'transformation': False,  # 이미 로그 변환했으므로 추가 변환 X
            'verbose': True,
        }

        # GPU 사용 시 추가 설정
        if self.use_gpu:
            print("🚀 GPU 가속 설정 활성화")
            setup_params['use_gpu'] = True
            
            # n_jobs를 -1로 설정하여 모든 CPU 코어 사용
            setup_params['n_jobs'] = -1

        self.setup_result = setup(**setup_params)
        print("✅ PyCaret setup 완료")

    # ------------------ Base model 탐색 (GPU Enhanced) ------------------
    def find_base_model(self, sort_metric="R2", include_models=None, exclude_models=None):
        """
        Base model 탐색 with GPU support
        
        Parameters:
        -----------
        sort_metric : str
            정렬 기준 메트릭
        include_models : list, optional
            포함할 모델 리스트 (None이면 전체)
        exclude_models : list, optional
            제외할 모델 리스트
        """
        if self.setup_result is None:
            raise ValueError("먼저 setup_pycaret()를 실행하세요.")

        print("🔍 Base model 탐색 시작...")
        
        compare_params = {
            'sort': sort_metric,
            'n_select': 1,
        }
        
        if include_models:
            compare_params['include'] = include_models
        if exclude_models:
            compare_params['exclude'] = exclude_models

        self.best_model = compare_models(**compare_params)
        print(f"🏆 Best model 선택 완료: {self.best_model}")
        return self.best_model

    # ------------------ GPU 최적화된 특정 모델 생성 ------------------
    def create_gpu_model(self, model_name='lightgbm', **kwargs):
        """
        GPU 최적화된 특정 모델 생성
        
        Parameters:
        -----------
        model_name : str
            'lightgbm', 'xgboost', 'catboost' 등
        **kwargs : 
            모델별 추가 파라미터
        """
        if self.setup_result is None:
            raise ValueError("먼저 setup_pycaret()를 실행하세요.")

        print(f"🎯 {model_name} 모델 생성 중 (GPU 최적화)...")

        # GPU 특화 파라미터 설정
        gpu_params = {}
        
        if model_name.lower() == 'lightgbm':
            if self.use_gpu:
                gpu_params = {
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0,
                }
        
        elif model_name.lower() == 'xgboost':
            if self.use_gpu:
                gpu_params = {
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'predictor': 'gpu_predictor',
                }
        
        elif model_name.lower() == 'catboost':
            if self.use_gpu:
                gpu_params = {
                    'task_type': 'GPU',
                    'devices': '0',
                }

        # 사용자 파라미터와 GPU 파라미터 병합
        final_params = {**gpu_params, **kwargs}

        self.best_model = create_model(model_name, **final_params)
        print(f"✅ {model_name} 모델 생성 완료")
        return self.best_model
    
    # ------------------ 모델 튜닝 (GPU Enhanced) ------------------
    def tune_model(self, n_iter=10, optimize='R2', search_library='scikit-learn'):
        """
        모델 하이퍼파라미터 튜닝 (GPU 지원)
        
        Parameters:
        -----------
        n_iter : int
            튜닝 반복 횟수
        optimize : str
            최적화할 메트릭
        search_library : str
            'optuna', 'scikit-learn', 'scikit-optimize'
        """
        if self.best_model is None:
            raise ValueError("먼저 모델을 생성하세요.")

        print(f"⚙️ 모델 튜닝 시작 (n_iter={n_iter}, optimize={optimize})...")
        
        self.best_model = tune_model(
            self.best_model,
            n_iter=n_iter,
            optimize=optimize,
            search_library=search_library,
        )
        
        print("✅ 모델 튜닝 완료")
        return self.best_model


    # ------------------ 모델 성능 저장 ------------------
    def save_metrics(self, metrics_dict=None, model_name=None):
        """
        price 컬럼은 log1p(price_real) 이므로,
        메트릭은 원래 스케일(price_real) 기준으로 계산해서 저장.
        """
        if metrics_dict is None:
            if self.best_model is None:
                raise ValueError("모델이 없습니다.")

            # train 전체에 대해 예측
            pred = predict_model(self.best_model, data=self.train_vectorized.copy())

            # 로그 스케일 타깃/예측
            y_log_true = self.train["price"].values
            y_log_pred = pred["Label"].values

            # 원래 스케일로 되돌리기
            y_true = np.expm1(y_log_true)
            y_pred = np.expm1(y_log_pred)

            r2 = r2_score(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mae = mean_absolute_error(y_true, y_pred)

            metrics_dict = {
                "R2": round(r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
            }

        self.metrics = metrics_dict

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name is None:
            model_name = str(self.best_model).split("(")[0]
        file_path = os.path.join(
            self.results_dir, f"{model_name}_metrics_{timestamp}.json"
        )

        with open(file_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

        print(f"💾 Metrics 저장 완료: {file_path}")

    # ------------------ 시각화 ------------------
    def visualize_model(self, plots=["residuals", "feature"]):
        if self.best_model is None:
            raise ValueError("먼저 find_base_model()로 모델을 선택하세요.")

        print("🎨 시각화 시작...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = str(self.best_model).split("(")[0]

        for p in plots:
            # 'feature_importance' 같은 alias를 'feature'로 매핑
            if p == "feature_importance":
                plot_name = "feature"
            else:
                plot_name = p

            try:
                save_path = os.path.join(
                    self.images_dir, f"{model_name}_{plot_name}_{timestamp}.png"
                )
                plot_model(self.best_model, plot=plot_name, save=True)
                print(f"✅ {plot_name} plot 저장 완료: {save_path}")
            except Exception as e:
                print(f"⚠️ Plot {p} 실패: {e}")

    # ------------------ Test 예측 & submission ------------------
    def predict_test(self, submission_file="submission.csv"):
        if self.best_model is None:
            raise ValueError("먼저 find_base_model()로 모델을 선택하세요.")

        print("📦 Test 데이터 예측 시작...")
        predictions = predict_model(self.best_model, data=self.test_vectorized.copy())

        # Label은 log1p(price_real) 예측값 → 원래 스케일로 되돌리기
        price_log_pred = predictions["Label"].values
        price_pred = np.expm1(price_log_pred)

        submission = pd.DataFrame(
            {"test_id": self.test["test_id"], "price": price_pred}
        )

        submission_path = os.path.join(self.results_dir, submission_file)
        submission.to_csv(submission_path, index=False)
        print(f"💾 Submission 저장 완료: {submission_path}")
        return submission
