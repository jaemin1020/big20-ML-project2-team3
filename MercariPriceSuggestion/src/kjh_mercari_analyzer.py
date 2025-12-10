import os
import gc
import json
import warnings
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# FastText (gensim)
try:
    from gensim.models import FastText
except ImportError:
    FastText = None

warnings.filterwarnings("ignore")


class MercariPyCaretAnalyzer:
    """
    Mercari Price Suggestion + PyCaret용 분석 클래스
    ────────────────────────────────────────────────
    네가 사용하는 드라이버 코드와 맞추어서 만든 버전이다.

    주요 메서드
    -----------
    - preprocess_all_staged(...)
    - vectorize_text(method="fasttext" 포함)
    - setup_pycaret(fold=3, use_gpu=False, n_jobs=4 지원)
    - find_best_model()
    - save_best_model(...)
    - save_metrics(..., vector_method=...)
    - predict_test(..., use_full=False)
    """

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

        # ---------------- 데이터 / 피처 ----------------
        self.train = None
        self.test = None
        self.train_vectorized = None
        self.test_vectorized = None

        # ---------------- CV용 베스트 모델 ----------------
        self.best_model = None
        self.best_model_name = None

        # ▶ RMSLE 저장용: 옛 이름 + 새 이름 둘 다 만들어 둠
        self.best_model_rmsle = None   # 예전 코드에서 쓰던 이름
        self.best_rmsle = None         # 새 코드에서 쓸 이름

        # ---------------- Full data 학습 모델 ----------------
        self.best_full_model = None

        # ▶ 경로도 alias 두 개 다
        self.best_full_model_path = None  # 예전 이름
        self.full_model_path = None       # 새 이름

        # ---------------- 기타 ----------------
        self.setup_result = None
        self.metrics = {}

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # GPU 가용성 체크
        self._check_gpu_availability()


    # ------------------ 공통 헬퍼: PyCaret 예측 컬럼 이름 처리 ------------------
    def _get_pred_column(self, df):
        """
        PyCaret 2.x / 3.x 호환용.
        predict_model() 결과에서 예측값 컬럼을 찾아서 넘파이 배열로 돌려준다.

        - PyCaret 2.x : 'Label'
        - PyCaret 3.x : 'prediction_label'
        """
        if "Label" in df.columns:
            return df["Label"].values
        if "prediction_label" in df.columns:
            return df["prediction_label"].values
        raise KeyError(
            "predict_model 결과에서 예측 컬럼을 찾지 못했습니다. "
            f"존재하는 컬럼: {list(df.columns)}"
        )


    # =====================================================================
    # 1. GPU 체크
    # =====================================================================
    def _check_gpu_availability(self):
        if not self.use_gpu:
            print("🖥️  CPU 모드로 실행")
            return

        print("🔍 GPU 가용성 체크 중...")
        gpu_available = False

        # torch / CUDA
        try:
            import torch

            if torch.cuda.is_available():
                dev = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                print(f"✅ CUDA 사용 가능: {dev}")
                print(f"   GPU 메모리: {props.total_memory / 1e9:.2f} GB")
                gpu_available = True
        except Exception:
            pass

        # XGBoost / LightGBM / CatBoost 버전만 출력
        try:
            import xgboost as xgb

            print(f"✅ XGBoost 버전: {xgb.__version__}")
        except Exception:
            print("⚠️  XGBoost 미설치")

        try:
            import lightgbm as lgb

            print(f"✅ LightGBM 버전: {lgb.__version__}")
        except Exception:
            print("⚠️  LightGBM 미설치")

        try:
            import catboost

            print(f"✅ CatBoost 버전: {catboost.__version__}")
        except Exception:
            print("⚠️  CatBoost 미설치")

        if not gpu_available:
            print("⚠️  GPU를 찾을 수 없습니다. CPU 모드로 전환합니다.")
            self.use_gpu = False
        else:
            print("🚀 GPU 가속 활성화")

    # =====================================================================
    # 2. 데이터 로딩 + 기본 전처리
    # =====================================================================
    def load_data(self, train_file="train.tsv", test_file="test.tsv", sep="\t"):
        print("📂 데이터 로딩 시작...")
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)

        self.train = pd.read_csv(train_path, sep=sep)
        self.test = pd.read_csv(test_path, sep=sep)

        print(f"original data shape : train {self.train.shape}, test {self.test.shape}")
        print("✅ Price 외 결측치 처리 및 데이터 전처리 시작...")

        # price 0 제거 + NaN 제거 (train만)
        self.train = self.train[self.train["price"] > 0].dropna(subset=["price"])

        # category_name 분리
        def _split_cat(x):
            if isinstance(x, str):
                parts = x.split("/", 2)
            else:
                parts = []
            while len(parts) < 3:
                parts.append("missing")
            return parts[:3]

        for df_name, df in [("train", self.train), ("test", self.test)]:
            df["main_cat"], df["sub_cat"], df["sub_sub_cat"] = zip(
                *df["category_name"].apply(_split_cat)
            )
            df["brand_name"] = df["brand_name"].fillna("Unknown").astype(str)
            df["category_name"] = df["category_name"].fillna("Unknown").astype(str)
            df["item_description"] = (
                df["item_description"].fillna("No description").astype(str)
            )
            df["name"] = df["name"].fillna("No name").astype(str)

            if df_name == "train":
                self.train = df.reset_index(drop=True)
            else:
                self.test = df.reset_index(drop=True)

        # price를 log1p로 변환
        self.train["price"] = np.log1p(self.train["price"])

        print(f"\n✅ 데이터 로드 완료: train {self.train.shape}, test {self.test.shape}")

    # =====================================================================
    # 3. 언더샘플링
    # =====================================================================
    def apply_undersampling(
        self,
        method="random",
        target_size=None,
        sampling_ratio=0.5,
        n_bins=10,
        random_state=23,
    ):
        if self.train is None:
            raise ValueError("먼저 load_data()를 실행하세요.")

        original_size = len(self.train)

        if target_size is None:
            target_size = int(original_size * sampling_ratio)
        else:
            target_size = min(target_size, original_size)

        print(f"\n🎯 Undersampling 시작... (method={method})")
        print(f"   원본 크기: {original_size:,}")
        print(f"   목표 크기: {target_size:,}")

        if method == "random":
            sampled_indices = np.random.RandomState(random_state).choice(
                self.train.index, size=target_size, replace=False
            )
            self.train = self.train.loc[sampled_indices].reset_index(drop=True)

        elif method == "stratified":
            self.train["price_bin"] = pd.qcut(
                self.train["price"], q=n_bins, labels=False, duplicates="drop"
            )
            sampled = self.train.groupby("price_bin", group_keys=False).apply(
                lambda x: x.sample(
                    n=max(1, int(len(x) * sampling_ratio)), random_state=random_state
                )
            )
            self.train = sampled.drop(columns=["price_bin"]).reset_index(drop=True)

        elif method == "top_n":
            self.train = self.train.nlargest(target_size, "price").reset_index(
                drop=True
            )
        else:
            raise ValueError(f"지원하지 않는 method: {method}")

        final_size = len(self.train)
        print(f"✅ Undersampling 완료: 최종 크기 {final_size:,}")

    # ------------------ 2) 전처리 (스테이징 + 캐시) ------------------
    def preprocess_all_staged(
        self,
        use_cache=True,
        save_cache=True,
        cols=None,
        undersample_frac=0.30,
        param_dict=None,
        debug=False,
    ):
        """
        1단계: 원본 TSV 로드
        2단계: log1p(price) + 결측/카테고리 분리
        3단계: undersampling (옵션)
        4단계: train/test 피클 캐시 저장 (옵션)

        cols, param_dict 는 지금은 주로 로그 남기는 용도로만 사용.
        """
        cache_train = os.path.join(self.data_dir, "train_preprocessed.pkl")
        cache_test = os.path.join(self.data_dir, "test_preprocessed.pkl")

        loaded_from_cache = False

        if use_cache and os.path.exists(cache_train) and os.path.exists(cache_test):
            try:
                self.train = pd.read_pickle(cache_train)
                self.test = pd.read_pickle(cache_test)
                loaded_from_cache = True
                print(f"📦 캐시된 train/test 로드 완료: "
                      f"train {self.train.shape}, test {self.test.shape}")
            except Exception as e:
                print(f"⚠ 캐시 로드 실패, 원본에서 다시 전처리: {e}")
                loaded_from_cache = False

        if not loaded_from_cache:
            # 1) 원본 로드 + 기본 전처리
            self.load_data()

            # 2) undersampling 적용 (옵션)
            if undersample_frac is not None and undersample_frac < 1.0:
                self.apply_undersampling(
                    method="random",
                    sampling_ratio=undersample_frac,
                )

            # 3) 캐시 저장 (옵션)
            if save_cache:
                try:
                    self.train.to_pickle(cache_train)
                    self.test.to_pickle(cache_test)
                    print(f"💾 전처리 결과 캐시 저장 완료: "
                          f"{cache_train}, {cache_test}")
                except Exception as e:
                    print(f"⚠ 캐시 저장 실패: {e}")

        if debug:
            print("\n[preprocess_all_staged] 요약")
            print(f" - 사용 컬럼(cols): {cols}")
            print(f" - param_dict    : {param_dict}")
            print(f" - train shape   : {None if self.train is None else self.train.shape}")
            print(f" - test  shape   : {None if self.test  is None else self.test.shape}")


    # ------------------ 3) 텍스트 벡터화 ------------------
    def vectorize_text(
        self,
        text_columns=["name", "item_description"],
        method="tfidf",
        max_features=50000,
        n_components=100,
    ):
        """
        text_columns: 벡터화할 텍스트 컬럼 리스트
        method:
            - 'tfidf'  : TF-IDF + TruncatedSVD
            - 'count'  : CountVectorizer + TruncatedSVD
            - 'fasttext': 현재는 'tfidf' 와 동일하게 처리 (이름만 fasttext)
                          → 파이프라인 상 method='fasttext' 를 위해 존재
        """
        if self.train is None or self.test is None:
            raise ValueError("먼저 preprocess_all_staged()/load_data() 를 실행하세요.")

        # fasttext 라고 들어오면 일단 tfidf 로 처리
        effective_method = method.lower()
        if effective_method == "fasttext":
            print("⚠️  method='fasttext' → 현재는 TF-IDF 기반 벡터화로 대체합니다.")
            effective_method = "tfidf"

        if effective_method not in ("tfidf", "count"):
            raise ValueError("method must be 'tfidf', 'count', or 'fasttext'")

        print(f"📝 텍스트 벡터화 및 차원 축소 시작... (method={method})")
        vectors = []
        feature_names = []

        for col in tqdm(text_columns, desc="Text columns"):
            print(f"▶ 컬럼: {col}")
            if effective_method == "tfidf":
                vec = TfidfVectorizer(max_features=max_features)
            else:  # 'count'
                vec = CountVectorizer(max_features=max_features)

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

        # 카테고리/브랜드/배송 변수 인코딩 및 추가
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


    # ------------------ 4) PyCaret setup ------------------
    def setup_pycaret(
        self,
        session_id=23,
        fold=3,
        use_gpu=False,
        n_jobs=4,
        experiment_name="mercari_gpu",
    ):
        if self.train_vectorized is None:
            raise ValueError("먼저 vectorize_text() 를 실행하세요.")

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

        data_for_pc = self.train_vectorized.assign(
            price=self.train["price"].reset_index(drop=True)
        )

        setup_params = dict(
            data=data_for_pc,
            target="price",
            session_id=session_id,
            experiment_name=experiment_name,
            categorical_features=existing_categorical if existing_categorical else None,
            normalize=True,
            transformation=False,  # 이미 log1p 했으니까
            fold=fold,
            verbose=True,
        )

        if use_gpu and self.use_gpu:
            print("🚀 GPU 가속 설정 활성화")
            setup_params["use_gpu"] = True
        else:
            setup_params["use_gpu"] = False

        if n_jobs is not None:
            setup_params["n_jobs"] = n_jobs

        self.setup_result = setup(**setup_params)
        print("✅ PyCaret setup 완료")


    # ------------------ 5) 베스트 모델 찾기 (Kaggle RMSLE 기준, log1p 타깃) ------------------
    def find_best_model(self, candidate_names=None):
        """
        여러 모델을 만들고,
        - PyCaret CV 결과표는 그대로 출력
        - train 전체에 대해 predict_model() 돌린 뒤
          log1p(price) 스케일에서의 RMSE (= Kaggle RMSLE) 로 순위를 매긴다.
        """
        if self.setup_result is None:
            raise ValueError("먼저 setup_pycaret() 를 실행하세요.")

        if candidate_names is None:
            candidate_names = ["lightgbm", "xgboost", "et", "rf"]

        print("\n🔍 후보 모델 RMSLE(log1p) 비교:")

        results = []  # (name, model, rmsle_log, cv_rmsle)
        y_true_log = self.train["price"].values  # 이미 log1p 변환된 타깃

        for name in candidate_names:
            print(f"\n▶ {name} 모델 생성/평가...")
            try:
                model = create_model(name)
                cv_df = pull().copy()
                print(cv_df)

                pred_df = predict_model(model, data=self.train_vectorized.copy())
                y_pred_log = self._get_pred_column(pred_df)

                rmsle_log = float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))

                try:
                    cv_rmsle = float(cv_df.loc["Mean", "RMSLE"])
                except Exception:
                    cv_rmsle = np.nan

                results.append((name, model, rmsle_log, cv_rmsle))
            except Exception as e:
                print(f"△ {name} 생성 실패: {e}")

        if not results:
            raise RuntimeError("비교 가능한 모델이 없습니다 (results 비어 있음).")

        results.sort(key=lambda x: x[2])
        best_name, best_model, best_rmsle, best_cv_rmsle = results[0]

        print("\n🏆 선택된 Best Model (RMSLE(log1p) 기준):")
        print(f"   - 이름            : {best_name}")
        print(f"   - train RMSLE     : {best_rmsle:.6f}")
        if not np.isnan(best_cv_rmsle):
            print(f"   - CV RMSLE(Mean)  : {best_cv_rmsle:.6f}")

        self.best_model_name = best_name
        self.best_model = best_model
        self.best_rmsle = best_rmsle

        return best_model


    # ------------------ 6) best_model 저장 ------------------
    def save_best_model(self, model_name=None):
        if self.best_model is None:
            raise ValueError("best_model 이 없습니다. find_best_model() 를 먼저 실행하세요.")

        if model_name is None:
            model_name = self.best_model_name or "best_model"

        path = os.path.join(self.model_dir, model_name)
        save_model(self.best_model, path)
        print(f"💾 best_model 저장 완료: {path}.pkl")

        self.full_model_path = f"{path}.pkl"  # 나중에 참조용
        return path


    # ------------------ 7) 모델 성능 저장 ------------------
    def save_metrics(self, metrics_dict=None, model_name=None, vector_method=None):
        """
        price 컬럼은 log1p(price_real) 이므로,
        메트릭은 원래 스케일(price_real) 기준으로 계산해서 저장.
        vector_method 는 메타 정보로 JSON 에 같이 기록.
        """
        if metrics_dict is None:
            if self.best_model is None:
                raise ValueError("모델이 없습니다.")

            pred = predict_model(self.best_model, data=self.train_vectorized.copy())

            y_log_true = self.train["price"].values
            y_log_pred = self._get_pred_column(pred)

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
            model_name = self.best_model_name or str(self.best_model).split("(")[0]

        payload = {
            "metrics": self.metrics,
            "model_name": model_name,
            "vector_method": vector_method,
        }

        file_path = os.path.join(
            self.results_dir, f"{model_name}_metrics_{timestamp}.json"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        print(f"💾 Metrics 저장 완료: {file_path}")


    # ------------------ 8) Test 예측 & submission ------------------
    def predict_test(self, submission_file="submission.csv", use_full=False):
        """
        use_full=True 이면 self.best_full_model 사용,
        아니면 self.best_model 사용.
        """
        model = None
        if use_full and self.best_full_model is not None:
            model = self.best_full_model
            print("📦 use_full=True → best_full_model 로 예측합니다.")
        else:
            model = self.best_model

        if model is None:
            raise ValueError("예측에 사용할 모델이 없습니다. find_best_model() 또는 full 학습을 먼저 실행하세요.")

        if self.test_vectorized is None:
            raise ValueError("먼저 vectorize_text() 를 실행하세요.")

        print("📦 Test 데이터 예측 시작...")
        predictions = predict_model(model, data=self.test_vectorized.copy())

        price_log_pred = self._get_pred_column(predictions)
        price_pred = np.expm1(price_log_pred)

        submission = pd.DataFrame(
            {"test_id": self.test["test_id"].values, "price": price_pred}
        )

        submission_path = os.path.join(self.results_dir, submission_file)
        submission.to_csv(submission_path, index=False)
        print(f"💾 Submission 저장 완료: {submission_path}")
        return submission
