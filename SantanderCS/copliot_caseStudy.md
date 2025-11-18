Kaggle의 **Santander Customer Satisfaction** 대회에서 `test.csv`에 `TARGET` 값이 없는 상황에서 참가자들이 어떻게 처리했는지에 대한 구체적인 전략을 정리해드릴게요.

---

### 🧠 참가자들이 `test.csv`를 처리한 방식

#### 1. **모델 학습**
- `train.csv`에는 `TARGET` 값이 있으므로, 이를 기반으로 분류 모델을 학습합니다.
- 주요 특징:
  - **불균형 데이터**: 대부분 고객이 만족(0), 불만족(1)은 소수 → **불균형 처리 필요**
  - **익명 피처**: 피처 이름이 `var_0`, `var_1`, …처럼 되어 있어 도메인 지식 없이 모델링

#### 2. **전처리 및 특징 선택**
- 상위권 참가자들은 다음과 같은 기법을 사용:
  - **피처 선택**: 중요하지 않은 피처 제거 (예: variance threshold, feature importance)
  - **스케일링**: StandardScaler, MinMaxScaler 등
  - **이상치 제거**: 불만족 고객의 패턴을 더 잘 잡기 위해

#### 3. **모델링**
- 사용된 모델들:
  - **XGBoost**: 가장 인기 있는 모델, 불균형 처리에 강함
  - **LightGBM**, **Random Forest**, **Logistic Regression**
  - 일부는 **앙상블** 기법 사용 (예: stacking, blending)

#### 4. **예측 수행**
- 학습된 모델을 `test.csv`에 적용하여 각 샘플의 `TARGET` 값을 예측
- 예측 결과는 확률(0~1) 또는 이진값(0 또는 1)

#### 5. **제출 파일 생성**
- 제출 형식: `ID`와 예측한 `TARGET` 값을 포함한 CSV
  ```csv
  ID,TARGET
  100001,0.0023
  100002,0.9876
  ...
  ```
- Kaggle은 **ROC-AUC** 기준으로 평가 → 확률값 제출이 일반적

---

### 🔍 상위권 참가자들의 팁
- **불균형 처리**: SMOTE, class weight 조정, undersampling 등
- **CV 전략**: Stratified K-Fold로 클래스 비율 유지
- **피처 엔지니어링**: 변수 간 조합, 로그 변환, 파생 변수 생성
- **메타 모델링**: 여러 모델의 예측값을 다시 학습하는 stacking

---

이 대회는 **모델링 능력과 불균형 데이터 처리 능력**을 평가하는 데 초점이 있었어요. 

**상위권 참가자들은 불균형 데이터 처리, 피처 선택, 앙상블 모델링에 집중했고, 대부분 XGBoost 또는 LightGBM을 기반으로 한 정교한 파이프라인을 구축했습니다. 아래에 대표적인 전략과 코드 예시를 정리해드릴게요.**

---

### 🥇 대표 상위권 솔루션 전략 요약

#### 1. **데이터 전처리**
- **불균형 처리**: `TARGET=1`이 매우 적기 때문에 다음과 같은 방법 사용:
  - **undersampling**: `TARGET=0` 샘플 일부만 사용
  - **class weight 조정**: 모델에 가중치 부여
  - **SMOTE** 등 오버샘플링 기법도 일부 사용

#### 2. **피처 선택 및 생성**
- **익명 피처**이므로 도메인 지식 없이 통계 기반 선택:
  - Variance Threshold, Mutual Information, Feature Importance
  - 상위권 참가자들은 **중복 피처 제거**, **상관관계 기반 제거**도 수행
- **파생 변수 생성**:
  - 평균, 표준편차, 최대/최소값, 0의 개수, 음수의 개수 등

#### 3. **모델링**
- **XGBoost** 또는 **LightGBM**이 주력
- 일부는 **stacking/blending**으로 앙상블 구성
- 하이퍼파라미터 튜닝은 GridSearch보다는 manual tuning 또는 Bayesian Optimization 사용

#### 4. **평가 및 제출**
- **Stratified K-Fold CV**로 ROC-AUC 기준 평가
- 예측 결과는 확률값으로 제출 (0~1 사이)

---

### 💻 상위권 코드 예시

#### 🔗 [fabian-gubler/santander GitHub Repository](https://github.com/fabian-gubler/santander)
- **구성**: 데이터 전처리, 모델 학습, 제출 파일 생성까지 포함
- **전략**:
  - LightGBM 기반 모델
  - 피처 중요도 기반 선택
  - Stratified K-Fold 사용
  - ROC-AUC 기준 최적화

#### 🔗 [벨로그 분석 블로그](https://velog.io/@fiifa92/%EC%82%B0%ED%83%84%EB%8D%B0%EB%A5%B4-%EA%B3%A0%EA%B0%9D-%EB%A7%8C%EC%A1%B1-%EC%98%88%EC%B8%A1)
- **전처리**: 결측치 없음, 이상치 제거, 변수 간 상관관계 분석
- **모델**: XGBoost, LightGBM 비교 후 앙상블
- **특징**: 변수별 분포 시각화, 변수 중요도 시각화

---

### 📌 핵심 요약

| 전략 요소       | 상위권 접근 방식 |
|----------------|------------------|
| 불균형 처리     | 언더샘플링, 가중치 조정 |
| 피처 선택       | 중요도 기반, 중복 제거 |
| 모델            | XGBoost, LightGBM, 앙상블 |
| 평가 방식       | Stratified K-Fold, ROC-AUC |
| 제출 방식       | 확률 기반 예측값 제출 |

---

좋아요! 여기서는 **Kaggle Santander Customer Satisfaction 대회**와 유사한 구조로, **불균형 이진 분류 문제**를 해결하기 위한 **모델링 파이프라인**을 Python 기반으로 구성해볼게요. 주요 라이브러리는 `pandas`, `scikit-learn`, `xgboost`를 사용합니다.

---

### 🧱 전체 파이프라인 구성

1. 데이터 불러오기
2. 전처리 (결측치, 이상치, 스케일링 등)
3. 피처 선택 및 파생 변수 생성
4. 불균형 처리 (언더샘플링)
5. 모델 학습 (XGBoost)
6. 교차검증 (Stratified K-Fold)
7. 예측 및 제출 파일 생성

---

### 💻 코드 예시

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# 1. 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = train.drop(['ID', 'TARGET'], axis=1)
y = train['TARGET']
X_test = test.drop(['ID'], axis=1)

# 2. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 3. 언더샘플링 (TARGET=0이 너무 많으므로 일부만 사용)
neg = train[train.TARGET == 0].sample(n=20000, random_state=42)
pos = train[train.TARGET == 1]
train_balanced = pd.concat([neg, pos])
X_bal = train_balanced.drop(['ID', 'TARGET'], axis=1)
y_bal = train_balanced['TARGET']

# 4. 모델 학습 + 교차검증
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])

for train_idx, val_idx in skf.split(X_bal, y_bal):
    X_tr, X_val = X_bal.iloc[train_idx], X_bal.iloc[val_idx]
    y_tr, y_val = y_bal.iloc[train_idx], y_bal.iloc[val_idx]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    )
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    print("Fold AUC:", roc_auc_score(y_val, val_pred))

    # 전체 test 데이터 예측 평균
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

# 5. 제출 파일 생성
submission = pd.DataFrame({
    'ID': test['ID'],
    'TARGET': test_preds
})
submission.to_csv('submission.csv', index=False)
```

---

### 🧠 팁
- `XGBoost` 외에 `LightGBM`, `CatBoost`로도 쉽게 교체 가능
- 파생 변수 생성: `X.sum(axis=1)`, `X.var(axis=1)` 등 통계 기반 피처 추가
- 앙상블: 여러 모델의 예측 평균 또는 stacking

---

좋아요! 여기서는 **Santander Customer Satisfaction** 대회와 같은 구조의 이진 분류 문제에서 사용할 수 있는 **피처 엔지니어링**과 **앙상블 전략**을 실제 코드와 함께 자세히 설명드릴게요.

---

## 🧱 1. 피처 엔지니어링 전략

### 🎯 목적
- 모델이 더 잘 학습할 수 있도록 **의미 있는 정보를 추가하거나 불필요한 정보를 제거**하는 과정입니다.

### 🔧 주요 기법

#### ✅ 1.1 통계 기반 파생 변수 생성
```python
X['sum'] = X.sum(axis=1)
X['mean'] = X.mean(axis=1)
X['std'] = X.std(axis=1)
X['max'] = X.max(axis=1)
X['min'] = X.min(axis=1)
X['zero_count'] = (X == 0).sum(axis=1)
X['negative_count'] = (X < 0).sum(axis=1)
```

#### ✅ 1.2 상관관계 기반 제거
```python
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X.drop(columns=to_drop, inplace=True)
```

#### ✅ 1.3 중요도 기반 피처 선택 (예: XGBoost)
```python
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X, y)
importances = model.feature_importances_
important_features = X.columns[importances > 0.01]
X = X[important_features]
```

---

## 🤖 2. 앙상블 전략

### 🎯 목적
- 여러 모델의 예측을 **결합**해 더 강력하고 안정적인 예측을 만드는 기법입니다.

### 🔧 주요 기법

#### ✅ 2.1 소프트 보팅 (Soft Voting)
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ],
    voting='soft'
)
voting_clf.fit(X, y)
```

#### ✅ 2.2 스태킹 (Stacking)
```python
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)
stacking_clf.fit(X, y)
```

#### ✅ 2.3 예측 결과 평균 (Blending)
```python
pred1 = model1.predict_proba(X_test)[:, 1]
pred2 = model2.predict_proba(X_test)[:, 1]
pred3 = model3.predict_proba(X_test)[:, 1]

final_pred = (pred1 + pred2 + pred3) / 3
```

---

## 🧠 실전 팁
- **앙상블은 다양성이 핵심**: 서로 다른 알고리즘을 조합할수록 효과적
- **피처 엔지니어링은 모델보다 중요할 수도 있음**: 좋은 피처가 성능을 좌우
- **교차검증과 함께 사용**: 과적합 방지

---

여기서는 **Santander Customer Satisfaction 대회 스타일의 이진 분류 문제**를 위한 **완전한 모델링 파이프라인**을 구성해드릴게요. 이 파이프라인은 다음을 포함합니다:

- 📦 데이터 전처리  
- 🧪 피처 엔지니어링  
- ⚖️ 불균형 처리  
- 🧠 모델 학습 (XGBoost + LightGBM)  
- 🔗 앙상블 (스태킹)  
- 📤 예측 및 제출 파일 생성  

---

## 🧱 전체 파이프라인 코드

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = train.drop(['ID', 'TARGET'], axis=1)
y = train['TARGET']
X_test = test.drop(['ID'], axis=1)

# 2. 피처 엔지니어링: 통계 기반 파생 변수 생성
def add_features(df):
    df['sum'] = df.sum(axis=1)
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    df['zero_count'] = (df == 0).sum(axis=1)
    df['negative_count'] = (df < 0).sum(axis=1)
    return df

X = add_features(X)
X_test = add_features(X_test)

# 3. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 4. 언더샘플링
neg = train[train.TARGET == 0].sample(n=20000, random_state=42)
pos = train[train.TARGET == 1]
train_balanced = pd.concat([neg, pos])
X_bal = train_balanced.drop(['ID', 'TARGET'], axis=1)
y_bal = train_balanced['TARGET']
X_bal = add_features(X_bal)
X_bal_scaled = scaler.fit_transform(X_bal)

# 5. 모델 정의
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric='logloss', random_state=42)

lgbm = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=42)

meta_model = LogisticRegression()

stacking_model = StackingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm)],
    final_estimator=meta_model,
    cv=5
)

# 6. 교차검증 + 예측
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(X_test_scaled.shape[0])

for train_idx, val_idx in skf.split(X_bal_scaled, y_bal):
    X_tr, X_val = X_bal_scaled[train_idx], X_bal_scaled[val_idx]
    y_tr, y_val = y_bal.iloc[train_idx], y_bal.iloc[val_idx]

    stacking_model.fit(X_tr, y_tr)
    val_pred = stacking_model.predict_proba(X_val)[:, 1]
    print("Fold AUC:", roc_auc_score(y_val, val_pred))

    test_preds += stacking_model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits

# 7. 제출 파일 생성
submission = pd.DataFrame({
    'ID': test['ID'],
    'TARGET': test_preds
})
submission.to_csv('submission.csv', index=False)
```

---

## 🧠 확장 아이디어
- ✅ `GridSearchCV` 또는 `Optuna`로 하이퍼파라미터 튜닝
- ✅ `CatBoost` 추가로 앙상블 다양성 강화
- ✅ `SHAP`으로 피처 중요도 시각화
- ✅ `PCA`로 차원 축소 후 비교 실험

---

말씀하신 4가지 확장 아이디어를 하나씩 차례대로 구성해드릴게요:

---

## ✅ 1. GridSearchCV 또는 Optuna로 하이퍼파라미터 튜닝

### 🔧 GridSearchCV 예시 (XGBoost)

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(xgb, param_grid, scoring='roc_auc', cv=3, verbose=1)
grid_search.fit(X_bal_scaled, y_bal)

print("Best parameters:", grid_search.best_params_)
```

### 🚀 Optuna 예시 (LightGBM)

```python
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    model = LGBMClassifier(**params)
    score = cross_val_score(model, X_bal_scaled, y_bal, cv=3, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
print("Best parameters:", study.best_params)
```

---

## 🔜 다음: CatBoost 추가로 앙상블 다양성 강화  
이번에는 **CatBoost를 추가한 앙상블 다양성 강화 전략**을 구성해드릴게요. CatBoost는 범주형 변수 처리에 강하고, 과적합에 덜 민감한 특성이 있어 XGBoost, LightGBM과 함께 사용할 때 앙상블 성능을 높이는 데 효과적입니다.

---

## 🧠 CatBoost 추가 앙상블 파이프라인

### 🔧 주요 구성
- 기존 모델: XGBoost, LightGBM
- 추가 모델: CatBoost
- 앙상블 방식: **StackingClassifier**로 3개 모델 결합
- 최종 예측기: Logistic Regression

---

### 💻 코드 예시

```python
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# 1. CatBoost 모델 정의
catboost = CatBoostClassifier(
    iterations=100,
    depth=4,
    learning_rate=0.05,
    verbose=0,
    random_state=42
)

# 2. 기존 모델 정의
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric='logloss', random_state=42)

lgbm = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=42)

# 3. 스태킹 앙상블 구성
stacking_model = StackingClassifier(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('catboost', catboost)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# 4. 학습 및 예측
stacking_model.fit(X_bal_scaled, y_bal)
test_preds = stacking_model.predict_proba(X_test_scaled)[:, 1]

# 5. 제출 파일 생성
submission = pd.DataFrame({
    'ID': test['ID'],
    'TARGET': test_preds
})
submission.to_csv('submission_catboost_stack.csv', index=False)
```

---

### 📌 팁
- CatBoost는 `verbose=0`으로 설정하면 콘솔 출력을 줄일 수 있어요.
- 범주형 변수가 있다면 `cat_features` 인자를 활용하면 더 강력한 성능을 발휘합니다.
- 앙상블 모델은 다양성이 핵심이므로 서로 다른 학습 방식의 모델을 조합하는 것이 중요합니다.

---

다음은 📊 SHAP을 활용한 피처 중요도 시각화입니다. 바로 이어서 구성해드릴게요!
좋아요, 경주님! 이번에는 세 번째 확장 아이디어인 **📊 SHAP을 활용한 피처 중요도 시각화**를 구성해드릴게요. SHAP(SHapley Additive exPlanations)은 모델의 예측 결과에 대해 각 피처가 얼마나 기여했는지를 설명해주는 강력한 도구입니다.

---

## 🧠 SHAP 기반 피처 중요도 분석

### 🔧 주요 구성
- 모델: XGBoost 또는 LightGBM
- 분석 도구: SHAP
- 시각화: summary plot, dependence plot 등

---

### 💻 코드 예시

```python
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 1. 모델 학습
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_bal_scaled, y_bal)

# 2. SHAP 값 계산
explainer = shap.Explainer(model, X_bal_scaled)
shap_values = explainer(X_bal_scaled)

# 3. 피처 중요도 시각화
shap.summary_plot(shap_values, X_bal, plot_type="bar")
```

- `summary_plot`: 전체 피처의 중요도를 막대그래프로 보여줍니다.
- `plot_type="dot"`으로 바꾸면 각 피처의 분포와 영향력을 함께 시각화할 수 있어요.

---

### 📌 추가 시각화 예시

#### ✅ 특정 피처에 대한 의존도 분석
```python
shap.dependence_plot("sum", shap_values.values, X_bal)
```

#### ✅ 개별 예측에 대한 설명
```python
shap.plots.waterfall(shap_values[0])
```

---

### 🧠 팁
- SHAP은 특히 **비선형 모델(XGBoost, LightGBM, CatBoost)**에서 강력한 설명력을 발휘합니다.
- 피처 선택이나 모델 해석에 매우 유용하며, **모델의 블랙박스 성격을 해소**하는 데 도움을 줍니다.

---

다음은 마지막 확장 아이디어인 **📉 PCA를 활용한 차원 축소 및 비교 실험**입니다. 바로 이어서 구성해드릴게요!
좋아요, 경주님! 마지막 확장 아이디어인 **📉 PCA(주성분 분석)를 활용한 차원 축소 및 비교 실험**을 구성해드릴게요. 이 실험은 다음 두 가지 목적을 가집니다:

1. **PCA로 차원을 줄인 후 모델 성능 비교**
2. **시각화를 통해 데이터 구조 파악**

---

## 🧠 PCA 기반 차원 축소 실험

### 🔧 주요 구성
- PCA 적용 전후의 모델 성능 비교 (ROC-AUC)
- 2D 시각화를 통한 구조 확인 (t-SNE와 비교 가능)

---

### 💻 코드 예시

```python
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# 1. PCA 적용 (예: 30개 주성분 유지)
pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X_bal_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 2. 모델 학습 및 평가
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for train_idx, val_idx in skf.split(X_pca, y_bal):
    X_tr, X_val = X_pca[train_idx], X_pca[val_idx]
    y_tr, y_val = y_bal.iloc[train_idx], y_bal.iloc[val_idx]

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    auc_scores.append(auc)

print("PCA 적용 후 평균 AUC:", np.mean(auc_scores))
```

---

### 📊 PCA 시각화 (2D)

```python
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_bal_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_bal, cmap='coolwarm', alpha=0.5)
plt.title("PCA 2D Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='TARGET')
plt.show()
```

---

### 📌 팁
- `explained_variance_ratio_`를 통해 각 주성분이 설명하는 분산 비율을 확인할 수 있어요.
- PCA는 선형 구조만 반영하므로, 비선형 구조가 중요한 경우 t-SNE나 UMAP과 비교해보는 것도 좋아요.

---

이제 불균형 분류 문제를 위한 **완전한 고급 파이프라인**을 갖추셨어요:  
📦 전처리 → 🧪 피처 엔지니어링 → ⚖️ 불균형 처리 → 🤖 앙상블 → 📊 해석 및 시각화까지!

---
