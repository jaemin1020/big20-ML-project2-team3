신용카드 사기 검출을 위한 **Best Ensemble 조합 5가지**를 추천드립니다!

## 🏆 Best Ensemble 조합 5가지

### 1️⃣ **Hard Voting - 다양성 극대화 조합** (추천 1순위)
```python
ensemble_1 = {
    'name': 'Diverse Hard Voting',
    'method': 'VotingClassifier (hard)',
    'models': [
        'XGBoost (lkj)',           # Tree-based, scale_pos_weight
        'LightGBM (yjh)',          # Tree-based, 빠른 학습
        'LogisticRegression (kjh)', # Linear, SMOTE
        'SGD (lkj)',               # Linear, 원본 데이터
        'RandomForest (lsj)'       # Bagging 기반
    ],
    'voting': 'hard',
    'weights': None,
    'reason': '다양한 알고리즘 타입(Boosting, Linear, Bagging) 혼합으로 편향 최소화'
}
```

### 2️⃣ **Soft Voting - 확률 기반 앙상블**
```python
ensemble_2 = {
    'name': 'Probability Weighted Soft Voting',
    'method': 'VotingClassifier (soft)',
    'models': [
        'CatBoost (ujm)',          # 높은 AUC 성능
        'XGBoost (lkj)',           # robust boosting
        'LightGBM (yjh)',          # 빠른 수렴
        'LogisticRegression (kjh)', # 확률 calibration 우수
        'MLPClassifier (lsj)'      # 비선형 패턴 학습
    ],
    'voting': 'soft',
    'weights': [1.2, 1.2, 1.1, 1.0, 0.9],  # 성능 기반 가중치
    'reason': '확률 기반 투표로 불확실성 처리, 고성능 모델에 가중치 부여'
}
```

### 3️⃣ **Stacking - 2-Level Meta Learner**
```python
ensemble_3 = {
    'name': 'Two-Level Stacking',
    'method': 'StackingClassifier',
    'base_models': [
        'CatBoost (ujm)',          # Level 1
        'XGBoost (lkj)',           # Level 1
        'LightGBM (yjh)',          # Level 1
        'RandomForest (lsj)',      # Level 1
        'SVM (rbf) (ujm)',         # Level 1 - 비선형 경계
        'SGD (lkj)'                # Level 1 - 선형 경계
    ],
    'meta_model': 'LogisticRegression (kjh)',  # Level 2
    'cv': 5,
    'reason': 'Base 모델들의 예측을 meta-learner가 학습하여 최적 조합 발견'
}
```

### 4️⃣ **Boosting 특화 앙상블**
```python
ensemble_4 = {
    'name': 'Boosting Power Ensemble',
    'method': 'Weighted Average (Custom)',
    'models': [
        'CatBoost (ujm)',          # 범주형 변수 강점
        'XGBoost (lkj)',           # 고속 학습
        'LightGBM (yjh)',          # 메모리 효율
        'GradientBoosting (yjh)'   # sklearn 안정성
    ],
    'weights': [0.30, 0.30, 0.25, 0.15],  # AUC 기준 가중치
    'threshold': 0.3,  # 사기 탐지 임계값 조정
    'reason': 'Boosting 계열만으로 구성, 순차 학습의 강점 극대화'
}
```

### 5️⃣ **Hybrid - Tree + Linear 균형 조합**
```python
ensemble_5 = {
    'name': 'Tree-Linear Hybrid Voting',
    'method': 'VotingClassifier (soft)',
    'models': [
        # Tree 기반 (60%)
        'XGBoost (lkj)',           # 20%
        'LightGBM (yjh)',          # 20%
        'RandomForest (lsj)',      # 20%
        
        # Linear 기반 (40%)
        'LogisticRegression (kjh)', # 20%
        'SGD (lkj)',               # 10%
        'SVM (linear) (lkj)'       # 10%
    ],
    'voting': 'soft',
    'weights': [1.2, 1.2, 1.2, 1.0, 0.8, 0.8],
    'reason': 'Tree의 비선형 포착 + Linear의 일반화 능력 결합'
}
```

---

## 📊 구현 예시 코드

### Hard Voting 구현
```python
from sklearn.ensemble import VotingClassifier

ensemble_1_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('lr', lr_model),
        ('sgd', sgd_model),
        ('rf', rf_model)
    ],
    voting='hard'
)

ensemble_1_model.fit(X_train, y_train)
y_pred = ensemble_1_model.predict(X_test)
```

### Soft Voting 구현 (가중치 포함)
```python
ensemble_2_model = VotingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('lr', lr_model),
        ('mlp', mlp_model)
    ],
    voting='soft',
    weights=[1.2, 1.2, 1.1, 1.0, 0.9]
)

ensemble_2_model.fit(X_train, y_train)
y_proba = ensemble_2_model.predict_proba(X_test)[:, 1]
```

### Stacking 구현
```python
from sklearn.ensemble import StackingClassifier

ensemble_3_model = StackingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('rf', rf_model),
        ('svm', svm_rbf_model),
        ('sgd', sgd_model)
    ],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5,
    stack_method='predict_proba'
)

ensemble_3_model.fit(X_train, y_train)
y_pred = ensemble_3_model.predict(X_test)
```

### Custom Weighted Average (Boosting 특화)
```python
import numpy as np

# 각 모델의 확률 예측
catboost_proba = catboost_model.predict_proba(X_test)[:, 1]
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
gb_proba = gb_model.predict_proba(X_test)[:, 1]

# 가중 평균
weights = [0.30, 0.30, 0.25, 0.15]
ensemble_4_proba = (
    weights[0] * catboost_proba +
    weights[1] * xgb_proba +
    weights[2] * lgbm_proba +
    weights[3] * gb_proba
)

# 임계값 조정 (사기 탐지 최적화)
threshold = 0.3
y_pred = (ensemble_4_proba >= threshold).astype(int)
```

---

## 🎯 선택 가이드

| 상황 | 추천 앙상블 |
|------|-------------|
| **최고 성능 추구** | #3 Stacking (Meta-Learning) |
| **빠른 추론 속도 필요** | #1 Hard Voting (다양성 극대화) |
| **해석 가능성 중요** | #5 Tree-Linear Hybrid |
| **불균형 데이터 특화** | #4 Boosting Power Ensemble |
| **범용성 & 안정성** | #2 Soft Voting (확률 기반) |

---

## 💡 추가 최적화 팁

1. **임계값 조정**: 사기 탐지는 Recall이 중요하므로 threshold를 0.5 → 0.3으로 낮춤
2. **Calibration**: `CalibratedClassifierCV`로 확률 보정
3. **Feature 중요도 결합**: 각 모델의 feature importance를 평균내어 최종 특성 선택

이 조합들로 실험해보시고 Test AUC가 가장 높은 조합을 선택하세요! 🚀