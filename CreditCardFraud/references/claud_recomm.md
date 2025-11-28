신용카드 사기 탐지는 **극심한 클래스 불균형** 문제가 특징인 좋은 프로젝트입니다. 단계별로 가이드해드리겠습니다.

## 1. 데이터 이해 및 탐색

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 데이터 로드
df = pd.read_csv('creditcard.csv')

# 기본 정보
print(df.shape)
print(df.info())
print(df.isnull().sum())

# 클래스 분포 확인 (핵심!)
print(df['Class'].value_counts())
print(f"사기 비율: {df['Class'].sum() / len(df) * 100:.3f}%")

# 시각화
plt.figure(figsize=(8, 4))
df['Class'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.show()
```

## 2. 데이터 전처리

```python
# Amount와 Time 스케일링 (V1-V28은 이미 PCA 처리됨)
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# 불필요한 컬럼 제거
df_processed = df.drop(['Amount', 'Time'], axis=1)

# 특성과 타겟 분리
X = df_processed.drop('Class', axis=1)
y = df_processed['Class']

# 학습/테스트 분할 (stratify 중요!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## 3. 클래스 불균형 처리 방법

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# 방법 1: SMOTE (오버샘플링)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 방법 2: 언더샘플링
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

# 방법 3: 혼합 방법
smt = SMOTETomek(random_state=42)
X_train_combined, y_train_combined = smt.fit_resample(X_train, y_train)
```

## 4. 모델 학습 및 평가

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 평가 함수
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 평가 지표
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return y_pred_proba

# 모델 정의
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), 
                             random_state=42),
    'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42),
}

# 각 모델 학습 및 평가
results = {}
for name, model in models.items():
    proba = evaluate_model(model, X_train_smote, y_train_smote, X_test, y_test, name)
    results[name] = proba
```

## 5. ROC Curve 비교

```python
plt.figure(figsize=(10, 6))
for name, y_pred_proba in results.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

## 6. 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

# Random Forest 예시
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train_smote, y_train_smote)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")

# 최적 모델로 평가
best_model = grid_search.best_estimator_
evaluate_model(best_model, X_train_smote, y_train_smote, X_test, y_test, "Tuned Random Forest")
```

## 7. 중요 평가 지표

사기 탐지에서는 **Precision, Recall, F1-Score, ROC-AUC**가 중요합니다:

- **Recall (재현율)**: 실제 사기를 얼마나 잡아냈는가 (중요!)
- **Precision (정밀도)**: 사기로 예측한 것 중 실제 사기 비율
- **ROC-AUC**: 전반적인 분류 성능

## 8. 주요 팁

1. **클래스 불균형 처리는 필수** - SMOTE, class_weight 활용
2. **Recall을 우선시** - 사기를 놓치는 것이 더 위험
3. **임계값 조정** 고려:
```python
# 임계값 낮추기 (Recall 향상)
y_pred_custom = (y_pred_proba > 0.3).astype(int)
```
4. **앙상블 모델**이 일반적으로 성능이 좋음
5. **교차 검증** 사용 (StratifiedKFold)

필요한 부분이 있으면 더 자세히 설명해드릴게요!

신용카드 사기 탐지에 효과적인 앙상블 모델 조합을 추천해드리겠습니다.

## 1. 기본 앙상블 - Voting Classifier

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 다양한 특성을 가진 모델 조합
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('xgb', XGBClassifier(scale_pos_weight=100, random_state=42)),
        ('lgbm', LGBMClassifier(class_weight='balanced', random_state=42))
    ],
    voting='soft'  # 확률 기반 투표
)

voting_clf.fit(X_train_smote, y_train_smote)
y_pred = voting_clf.predict(X_test)
y_pred_proba = voting_clf.predict_proba(X_test)[:, 1]

print(f"Voting Ensemble ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

## 2. Stacking Classifier (추천!)

```python
from sklearn.ensemble import StackingClassifier

# Base models (다양성 확보)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=100, random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# Meta model
meta_model = LogisticRegression(class_weight='balanced')

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba'
)

stacking_clf.fit(X_train_smote, y_train_smote)
y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]

print(f"Stacking Ensemble ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

## 3. 가중치 기반 앙상블 (커스텀)

```python
# 개별 모델 학습
models = {
    'rf': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'xgb': XGBClassifier(n_estimators=200, scale_pos_weight=100, random_state=42),
    'lgbm': LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'catboost': CatBoostClassifier(iterations=200, class_weights=[1, 100], verbose=0, random_state=42)
}

predictions = {}
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    predictions[name] = model.predict_proba(X_test)[:, 1]
    print(f"{name} ROC-AUC: {roc_auc_score(y_test, predictions[name]):.4f}")

# 가중 평균 (성능 기반 가중치)
weights = {
    'rf': 0.2,
    'xgb': 0.3,
    'lgbm': 0.3,
    'catboost': 0.2
}

weighted_pred = sum(predictions[name] * weight for name, weight in weights.items())
print(f"\nWeighted Ensemble ROC-AUC: {roc_auc_score(y_test, weighted_pred):.4f}")
```

## 4. 추천 조합별 특징

### **조합 A: 균형잡힌 앙상블**
```python
estimators = [
    ('lr', LogisticRegression(class_weight='balanced')),  # 선형 모델
    ('rf', RandomForestClassifier(class_weight='balanced')),  # 배깅
    ('xgb', XGBClassifier(scale_pos_weight=100)),  # 부스팅
    ('svm', SVC(probability=True, class_weight='balanced'))  # 커널 기반
]
```
- **장점**: 다양한 알고리즘 특성 활용
- **단점**: SVM은 느림

### **조합 B: 트리 기반 앙상블 (추천!)**
```python
estimators = [
    ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced')),
    ('xgb', XGBClassifier(n_estimators=200, scale_pos_weight=100)),
    ('lgbm', LGBMClassifier(n_estimators=200, class_weight='balanced')),
    ('catboost', CatBoostClassifier(iterations=200, class_weights=[1, 100], verbose=0))
]
```
- **장점**: 모두 트리 기반이지만 다른 방식 (배깅 vs 부스팅)
- **추천 이유**: 빠르고 성능 우수

### **조합 C: 부스팅 앙상블**
```python
estimators = [
    ('xgb', XGBClassifier(scale_pos_weight=100)),
    ('lgbm', LGBMClassifier(class_weight='balanced')),
    ('catboost', CatBoostClassifier(class_weights=[1, 100], verbose=0)),
    ('gb', GradientBoostingClassifier())
]
```
- **장점**: 모두 강력한 부스팅 알고리즘
- **단점**: 과적합 가능성

## 5. 최적 가중치 찾기

```python
from scipy.optimize import minimize

def objective(weights, predictions, y_true):
    weighted_pred = sum(pred * w for pred, w in zip(predictions.values(), weights))
    return -roc_auc_score(y_true, weighted_pred)  # 음수 (최소화)

# 초기 가중치
initial_weights = [0.25] * len(predictions)

# 제약 조건: 가중치 합 = 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * len(predictions)

result = minimize(
    objective,
    initial_weights,
    args=(predictions, y_test),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
print("Optimal weights:", dict(zip(predictions.keys(), optimal_weights)))

optimal_pred = sum(pred * w for pred, w in zip(predictions.values(), optimal_weights))
print(f"Optimized Ensemble ROC-AUC: {roc_auc_score(y_test, optimal_pred):.4f}")
```

## 6. 전체 파이프라인 예시

```python
from sklearn.pipeline import Pipeline

# Stacking with preprocessing
stacking_pipeline = Pipeline([
    ('stacking', StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, 
                                         class_weight='balanced', random_state=42)),
            ('xgb', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                 scale_pos_weight=100, random_state=42)),
            ('lgbm', LGBMClassifier(n_estimators=200, num_leaves=31,
                                   class_weight='balanced', random_state=42))
        ],
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=5
    ))
])

stacking_pipeline.fit(X_train_smote, y_train_smote)
y_pred_proba = stacking_pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, (y_pred_proba > 0.5).astype(int)))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

## 7. 성능 비교 함수

```python
def compare_ensembles(X_train, y_train, X_test, y_test):
    results = {}
    
    # 1. Voting
    voting = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
            ('xgb', XGBClassifier(scale_pos_weight=100, random_state=42)),
            ('lgbm', LGBMClassifier(class_weight='balanced', random_state=42))
        ],
        voting='soft'
    )
    voting.fit(X_train, y_train)
    results['Voting'] = roc_auc_score(y_test, voting.predict_proba(X_test)[:, 1])
    
    # 2. Stacking
    stacking = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
            ('xgb', XGBClassifier(scale_pos_weight=100, random_state=42)),
            ('lgbm', LGBMClassifier(class_weight='balanced', random_state=42))
        ],
        final_estimator=LogisticRegression(class_weight='balanced')
    )
    stacking.fit(X_train, y_train)
    results['Stacking'] = roc_auc_score(y_test, stacking.predict_proba(X_test)[:, 1])
    
    return pd.DataFrame(results, index=['ROC-AUC']).T.sort_values('ROC-AUC', ascending=False)

comparison = compare_ensembles(X_train_smote, y_train_smote, X_test, y_test)
print(comparison)
```

## 추천 전략

**사기 탐지에 최적화된 조합**:
1. **XGBoost** (강력한 부스팅)
2. **LightGBM** (빠르고 효율적)
3. **Random Forest** (안정적인 배깅)
4. **CatBoost** (범주형 처리 우수)

**Meta model**: Logistic Regression (해석 가능성)

이 조합이 **속도, 성능, 안정성**의 균형이 가장 좋습니다!