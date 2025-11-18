## 🧱 Santander 스타일 이진 분류 프로젝트 구조

### 📁 1. 디렉토리 구조

```
santander_project/
├── data/
│   ├── train.csv
│   ├── test.csv
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_stack.ipynb
│   ├── 04_shap_analysis.ipynb
│   ├── 05_pca_experiment.ipynb
├── models/
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   ├── catboost_model.pkl
├── submissions/
│   ├── submission_stack.csv
│   ├── submission_pca.csv
├── utils/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
├── README.md
```

---

### 🧪 2. 주요 모듈 설명

#### `preprocessing.py`
- 결측치 처리
- 스케일링 (StandardScaler)
- 언더샘플링 / 오버샘플링

#### `feature_engineering.py`
- 통계 기반 파생 변수 생성
- 상관관계 기반 피처 제거
- SHAP 기반 중요도 선택

#### `modeling.py`
- XGBoost, LightGBM, CatBoost 모델 정의
- StackingClassifier 구성
- Optuna 또는 GridSearchCV 튜닝

#### `evaluation.py`
- ROC-AUC 계산
- 교차검증
- SHAP 시각화
- PCA 시각화

---

### 📊 3. 실험 노트북 흐름

1. **01_preprocessing.ipynb**  
   → 데이터 로딩, 언더샘플링, 스케일링

2. **02_feature_engineering.ipynb**  
   → 파생 변수 생성, 피처 선택, SHAP 분석

3. **03_modeling_stack.ipynb**  
   → XGB + LGBM + CatBoost 스태킹, 제출 파일 생성

4. **04_shap_analysis.ipynb**  
   → SHAP summary plot, waterfall plot

5. **05_pca_experiment.ipynb**  
   → PCA 적용 후 모델 성능 비교 및 2D 시각화

---

### 🧠 확장 가능성
- ✅ AutoML 도입 (e.g. H2O, PyCaret)
- ✅ Streamlit 대시보드로 결과 시각화
- ✅ 모델 서빙 (Flask, FastAPI)

---

아래는 **01_preprocessing.ipynb ~ 05_pca_experiment.ipynb**까지의 핵심 코드 흐름과 설명을 담은 노트북 템플릿입니다. 각 단계는 실전 대회 스타일로 구성되어 있어, 바로 실행하거나 커스터마이징하기 쉬워요.

---

## 📘 01_preprocessing.ipynb

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 데이터 로딩
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 피처와 타겟 분리
X = train.drop(['ID', 'TARGET'], axis=1)
y = train['TARGET']
X_test = test.drop(['ID'], axis=1)

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 언더샘플링
neg = train[train.TARGET == 0].sample(n=20000, random_state=42)
pos = train[train.TARGET == 1]
train_balanced = pd.concat([neg, pos])
X_bal = train_balanced.drop(['ID', 'TARGET'], axis=1)
y_bal = train_balanced['TARGET']
```

---

## 📘 02_feature_engineering.ipynb

```python
# 통계 기반 파생 변수 생성
def add_features(df):
    df['sum'] = df.sum(axis=1)
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    df['zero_count'] = (df == 0).sum(axis=1)
    df['negative_count'] = (df < 0).sum(axis=1)
    return df

X_bal = add_features(X_bal)
X_test = add_features(X_test)

# 상관관계 기반 피처 제거
corr_matrix = X_bal.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_bal.drop(columns=to_drop, inplace=True)
X_test.drop(columns=to_drop, inplace=True)
```

---

## 📘 03_modeling_stack.ipynb

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 모델 정의
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgbm = LGBMClassifier(random_state=42)
catboost = CatBoostClassifier(verbose=0, random_state=42)
meta_model = LogisticRegression()

# 스태킹 앙상블
stacking_model = StackingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('catboost', catboost)],
    final_estimator=meta_model,
    cv=5
)

# 학습 및 예측
stacking_model.fit(X_bal, y_bal)
test_preds = stacking_model.predict_proba(X_test)[:, 1]

# 제출 파일 생성
submission = pd.DataFrame({'ID': test['ID'], 'TARGET': test_preds})
submission.to_csv('submissions/submission_stack.csv', index=False)
```

---

## 📘 04_shap_analysis.ipynb

```python
import shap
import matplotlib.pyplot as plt

# SHAP 분석
explainer = shap.Explainer(xgb, X_bal)
shap_values = explainer(X_bal)

# summary plot
shap.summary_plot(shap_values, X_bal, plot_type="bar")

# dependence plot 예시
shap.dependence_plot("sum", shap_values.values, X_bal)
```

---

## 📘 05_pca_experiment.ipynb

```python
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# PCA 적용
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_bal)
X_test_pca = pca.transform(X_test)

# 모델 학습 및 평가
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for train_idx, val_idx in skf.split(X_pca, y_bal):
    X_tr, X_val = X_pca[train_idx], X_pca[val_idx]
    y_tr, y_val = y_bal.iloc[train_idx], y_bal.iloc[val_idx]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc_scores.append(roc_auc_score(y_val, val_pred))

print("PCA 적용 후 평균 AUC:", np.mean(auc_scores))

# 2D 시각화
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_bal)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_bal, cmap='coolwarm', alpha=0.5)
plt.title("PCA 2D Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='TARGET')
plt.show()
```
---
아래는 `notebooks/06_feature_selection_statsmodels.ipynb`의 구성입니다.  
이 노트북은 **statsmodels를 활용한 로지스틱 회귀 분석**과 **p-value 기반 피처 선택**을 중심으로 구성되어 있어요.

---

## 📘 06_feature_selection_statsmodels.ipynb

### 🔹 1. 데이터 로딩 및 전처리

```python
import pandas as pd
from utils.preprocessing import load_data, undersample, split_features_target
from utils.feature_engineering import add_statistical_features

# 데이터 로딩
train, _ = load_data('data/train.csv', 'data/test.csv')
balanced = undersample(train)
X_bal, y_bal = split_features_target(balanced)

# 파생 변수 추가
X_bal = add_statistical_features(X_bal)
```

---

### 🔹 2. statsmodels 로지스틱 회귀 분석

```python
import statsmodels.api as sm

# 상수항 추가
X_const = sm.add_constant(X_bal)

# 모델 적합
logit_model = sm.Logit(y_bal, X_const)
result = logit_model.fit()

# 회귀표 출력
print(result.summary())
```

---

### 🔹 3. p-value 기반 피처 선택

```python
# p-value 기준 피처 선택
p_values = result.pvalues
selected_features = p_values[p_values < 0.05].index.tolist()

# 상수항 제거
selected_features = [f for f in selected_features if f != 'const']

# 선택된 피처만 추출
X_selected = X_bal[selected_features]
```

---

### 🔹 4. 선택된 피처로 모델 재학습 및 평가

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for train_idx, val_idx in skf.split(X_selected, y_bal):
    X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_tr, y_val = y_bal.iloc[train_idx], y_bal.iloc[val_idx]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc_scores.append(roc_auc_score(y_val, val_pred))

print("선택된 피처 기반 평균 AUC:", np.mean(auc_scores))
```
여기서는 **statsmodels 기반 로지스틱 회귀 분석 결과를 시각화하는 방법**을 소개할게요. 특히 다음 두 가지를 중심으로 설명드릴게요:

---

## 📊 1. 회귀계수 시각화 (p-value 기반 중요도)

### 🔧 막대그래프: 유의한 피처만 시각화

```python
import matplotlib.pyplot as plt

# 회귀계수와 p-value 추출
coefs = result.params
pvals = result.pvalues

# 유의한 피처만 필터링 (p < 0.05)
significant = pvals[pvals < 0.05].index
coefs_sig = coefs[significant].drop('const', errors='ignore')

# 시각화
plt.figure(figsize=(10, 6))
coefs_sig.sort_values().plot(kind='barh', color='skyblue')
plt.title("유의한 피처의 회귀계수 (p < 0.05)")
plt.xlabel("회귀계수")
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## 📈 2. p-value 분포 시각화

### 🔧 히스토그램: 전체 피처의 p-value 분포 확인

```python
plt.figure(figsize=(8, 5))
plt.hist(pvals, bins=30, color='salmon', edgecolor='black')
plt.axvline(0.05, color='blue', linestyle='--', label='p = 0.05')
plt.title("전체 피처의 p-value 분포")
plt.xlabel("p-value")
plt.ylabel("피처 수")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## 🧠 해석 팁

- **회귀계수가 양수**이면 해당 피처가 `TARGET=1`일 확률을 **높이는 방향**으로 작용
- **음수**이면 **낮추는 방향**
- **p-value < 0.05**는 통계적으로 유의미한 피처로 간주

---

이 시각화는 모델 해석과 피처 선택에 매우 유용해요. 원하시면 이 시각화 결과를 자동 저장하거나 Streamlit 대시보드로 보여주는 방법도 알려드릴 수 있어요!

---

### 📌 요약

- `statsmodels`로 피처 유의성 분석
- `p-value < 0.05` 기준으로 피처 선택
- 선택된 피처로 모델 재학습 → 성능 비교

---



---

 아래는 `utils/` 폴더에 들어갈 모듈화된 Python 코드입니다. 각 파일은 프로젝트의 재사용성과 유지보수성을 높이기 위해 기능별로 분리되어 있어요.

---

## 📁 `utils/preprocessing.py`

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def split_features_target(train_df, target_col='TARGET'):
    X = train_df.drop(['ID', target_col], axis=1)
    y = train_df[target_col]
    return X, y

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def undersample(train_df, target_col='TARGET', n_majority=20000, random_state=42):
    majority = train_df[train_df[target_col] == 0].sample(n=n_majority, random_state=random_state)
    minority = train_df[train_df[target_col] == 1]
    balanced = pd.concat([majority, minority])
    return balanced
```

---

## 📁 `utils/feature_engineering.py`

```python
import numpy as np

def add_statistical_features(df):
    df['sum'] = df.sum(axis=1)
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    df['zero_count'] = (df == 0).sum(axis=1)
    df['negative_count'] = (df < 0).sum(axis=1)
    return df

def drop_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop
```

---

## 📁 `utils/modeling.py`
- pip install catboost  @ ml_dev 에 필요함! 

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

def get_base_models():
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    lgbm = LGBMClassifier(random_state=42)
    catboost = CatBoostClassifier(verbose=0, random_state=42)
    return xgb, lgbm, catboost

def get_stacking_model(xgb, lgbm, catboost):
    meta_model = LogisticRegression()
    stacking_model = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('catboost', catboost)],
        final_estimator=meta_model,
        cv=5
    )
    return stacking_model
```

---

## 📁 `utils/evaluation.py`

- pip install shap @ ml_dev # install 필요함

```python
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def evaluate_model_cv(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y.iloc[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        auc = roc_auc_score(y.iloc[val_idx], preds)
        aucs.append(auc)
    return aucs

def plot_shap_summary(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, plot_type="bar")

def plot_pca_2d(X_2d, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', alpha=0.5)
    plt.title("PCA 2D Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label='TARGET')
    plt.show()
```

---

이제 `utils/` 폴더에 이 네 개의 모듈을 배치하면, 각 노트북에서 다음과 같이 불러와 사용할 수 있어요:

```python
from utils.preprocessing import load_data, split_features_target, scale_data, undersample
from utils.feature_engineering import add_statistical_features, drop_highly_correlated_features
from utils.modeling import get_base_models, get_stacking_model
from utils.evaluation import evaluate_model_cv, plot_shap_summary, plot_pca_2d
```

`models/` 폴더에 있는 `.pkl` 파일들은 **학습된 머신러닝 모델을 저장한 파일**입니다. 이 파일들은 모델을 다시 학습하지 않고도 불러와서 예측에 사용할 수 있도록 만들어진 거예요.

---

## 🧠 `.pkl` 파일 생성 방법

### 🔧 사용 라이브러리: `joblib` 또는 `pickle`

#### ✅ 예시: `joblib`을 사용한 모델 저장

```python
import joblib
from xgboost import XGBClassifier

# 모델 학습
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_bal, y_bal)

# 모델 저장
joblib.dump(model, 'models/xgb_model.pkl')
```

#### ✅ 예시: `pickle`을 사용한 저장

```python
import pickle

with open('models/xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

---

## 📥 모델 불러오기

```python
# joblib 방식
model = joblib.load('models/xgb_model.pkl')

# pickle 방식
with open('models/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## 📁 `models/` 폴더에 들어가는 파일 예시

| 파일명             | 설명                                 |
|--------------------|--------------------------------------|
| `xgb_model.pkl`     | XGBoost로 학습한 모델                |
| `lgbm_model.pkl`    | LightGBM으로 학습한 모델             |
| `catboost_model.pkl`| CatBoost로 학습한 모델               |
| `stacking_model.pkl`| 스태킹 앙상블 모델 전체 저장         |

---

이렇게 저장된 모델은 추후 예측, 배포, 또는 실험 재현에 매우 유용해요. 원하시면 `models/`에 저장된 모델을 활용한 예측 스크립트도 만들어드릴 수 있어요!

 `.pkl` 파일을 만드는 과정은 **`utils/modeling.py`에서 정의한 모델을 불러와서 학습한 뒤 저장하는 단계**에서 수행됩니다. 하지만 제가 앞서 제공한 `modeling.py`는 **모델 정의만 포함**되어 있고, **모델 학습과 저장은 별도로 실행해야 해요.**

---

## ✅ `.pkl` 파일 생성 흐름

### 1. `utils/modeling.py`에서 모델 정의
```python
xgb, lgbm, catboost = get_base_models()
stacking_model = get_stacking_model(xgb, lgbm, catboost)
```

### 2. 노트북 또는 스크립트에서 모델 학습
```python
stacking_model.fit(X_bal, y_bal)
```

### 3. 학습된 모델을 `.pkl`로 저장
```python
import joblib
joblib.dump(stacking_model, 'models/stacking_model.pkl')
```

---

## 📁 추천 저장 스크립트 (`save_models.py`)

```python
from utils.modeling import get_base_models, get_stacking_model
from utils.preprocessing import load_data, split_features_target, scale_data, undersample
from utils.feature_engineering import add_statistical_features
import joblib

# 데이터 로딩 및 전처리
train, test = load_data('data/train.csv', 'data/test.csv')
balanced = undersample(train)
X_bal, y_bal = split_features_target(balanced)
X_bal = add_statistical_features(X_bal)
X_test = add_statistical_features(test.drop(['ID'], axis=1))
X_bal_scaled, X_test_scaled, _ = scale_data(X_bal, X_test)

# 모델 정의 및 학습
xgb, lgbm, catboost = get_base_models()
stacking_model = get_stacking_model(xgb, lgbm, catboost)
stacking_model.fit(X_bal_scaled, y_bal)

# 저장
joblib.dump(xgb, 'models/xgb_model.pkl')
joblib.dump(lgbm, 'models/lgbm_model.pkl')
joblib.dump(catboost, 'models/catboost_model.pkl')
joblib.dump(stacking_model, 'models/stacking_model.pkl')
```

---

이렇게 하면 `models/` 폴더에 `.pkl` 파일들이 생성되고, 이후 예측이나 배포에 바로 활용할 수 있어요. 원하시면 `load_models.py`도 만들어드릴 수 있어요!
 아래는 `models/` 폴더에 저장된 `.pkl` 파일들을 불러와서 예측에 활용할 수 있는 **`load_models.py` 스크립트**입니다. 이 스크립트는 저장된 모델을 로딩하고, 새로운 데이터에 대해 예측을 수행하며, 결과를 제출 파일로 저장하는 데 사용됩니다.

---

## 📁 `load_models.py`

```python
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from utils.feature_engineering import add_statistical_features

# 1. 테스트 데이터 로딩 및 전처리
test = pd.read_csv('data/test.csv')
X_test = test.drop(['ID'], axis=1)
X_test = add_statistical_features(X_test)

# 2. 스케일링 (훈련 시 사용한 스케일러와 동일하게 적용해야 함)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)  # 또는 저장된 스케일러를 불러와 적용

# 3. 모델 불러오기
xgb_model = joblib.load('models/xgb_model.pkl')
lgbm_model = joblib.load('models/lgbm_model.pkl')
catboost_model = joblib.load('models/catboost_model.pkl')
stacking_model = joblib.load('models/stacking_model.pkl')

# 4. 예측 수행
xgb_preds = xgb_model.predict_proba(X_test_scaled)[:, 1]
lgbm_preds = lgbm_model.predict_proba(X_test_scaled)[:, 1]
catboost_preds = catboost_model.predict_proba(X_test_scaled)[:, 1]
stacking_preds = stacking_model.predict_proba(X_test_scaled)[:, 1]

# 5. 결과 저장
pd.DataFrame({
    'ID': test['ID'],
    'XGB': xgb_preds,
    'LGBM': lgbm_preds,
    'CatBoost': catboost_preds,
    'Stacking': stacking_preds
}).to_csv('submissions/predictions_from_loaded_models.csv', index=False)
```

---

## 🧠 팁
- `StandardScaler()`는 훈련 시 저장해둔 스케일러를 `joblib.dump()`로 저장하고 여기서 `joblib.load()`로 불러오는 것이 더 정확합니다.
- 예측 결과를 비교하거나 앙상블 평균을 다시 계산할 수도 있어요:
  ```python
  blended_preds = (xgb_preds + lgbm_preds + catboost_preds) / 3
  ```

---



