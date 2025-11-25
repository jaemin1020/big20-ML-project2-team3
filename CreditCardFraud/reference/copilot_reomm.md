**빠른 답변:** 캐글 신용카드 사기 데이터셋은 **극도로 불균형(0.172%만 사기)**하기 때문에, 단순 정확도보다 **재현율(Recall), AUC, F1-score** 같은 지표를 중심으로 모델을 평가해야 합니다. 여러 머신러닝 모델을 적용하려면 데이터 전처리 → 불균형 처리 → 다양한 모델 학습 → 성능 비교 → 비용 민감 학습까지 단계적으로 진행하는 것이 가장 효과적입니다.  

---

## 📌 분석 가이드라인

### 1. 데이터 이해
- **출처**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **특징**: 284,807건 중 492건만 사기 → **불균형 데이터 문제**  
- **변수**: 대부분 PCA로 변환된 익명 변수, `Amount`, `Time`, `Class`(0=정상, 1=사기)

### 2. 데이터 전처리
- **스케일링**: `Amount`, `Time`은 표준화 필요  
- **EDA**: 사기 거래의 분포, 거래 금액 패턴 확인  
- **불균형 처리**:  
  - **언더샘플링**: 정상 거래 줄이기 → 데이터 손실 위험  
  - **오버샘플링 (SMOTE)**: 사기 거래 증식 → 과적합 주의  
  - **Class weight 조정**: 모델 학습 시 가중치 부여  

### 3. 모델 선택
- **Baseline**: Logistic Regression (해석력 높음)  
- **Tree-based**: Random Forest, XGBoost, LightGBM (불균형에 강함)  
- **Neural Network**: MLPClassifier (복잡 패턴 탐지 가능)  
- **Ensemble**: Stacking, Bagging → 여러 모델 결합으로 성능 향상  

### 4. 성능 평가
- **Accuracy는 무의미** (거의 정상만 맞추는 모델도 99% 정확도)  
- **중요 지표**:  
  - Recall (사기 탐지율)  
  - Precision (탐지된 사기 중 진짜 비율)  
  - F1-score (균형 지표)  
  - ROC-AUC (전체 분류 성능)  
- **비용 민감 학습**: False Negative(사기 놓침)의 비용이 훨씬 크므로 Recall을 우선시  

### 5. 하이퍼파라미터 튜닝
- **GridSearchCV / RandomizedSearchCV** 활용  
- XGBoost: `max_depth`, `learning_rate`, `scale_pos_weight`  
- Logistic Regression: `C`, `penalty`, `solver`  
- Random Forest: `n_estimators`, `max_features`, `class_weight`

### 6. 모델 비교 및 결론
- 여러 모델 성능을 **같은 평가 지표**로 비교  
- 최종적으로는 **Recall 중심 + AUC 보조**로 선택  
- 실제 적용 시에는 **실시간 탐지 가능성**과 **비용 민감도**도 고려해야 함  

---

## 🚀 추천 워크플로우
1. 데이터 로드 및 전처리  
2. 불균형 처리 (SMOTE, class_weight)  
3. Logistic Regression → baseline  
4. Random Forest / XGBoost / LightGBM → 트리 기반  
5. Neural Network → 추가 실험  
6. Stacking → 최종 앙상블  
7. Recall, Precision, AUC 비교 후 최적 모델 선택  

---

제가 원한다면, 위 단계별로 **실제 코드 예시**를 짜서 보여드릴 수 있습니다. 예를 들어 Logistic Regression과 XGBoost를 비교하는 파이프라인을 만들어드릴 수도 있어요. 원하나요?

# Credit card fraud analysis with multiple models

아래 코드는 추천한 순서대로 실행되는, 불균형 처리와 다양한 모델 비교를 포함한 end-to-end 파이프라인입니다. 바로 실행 가능한 형태로 구성했으며, 필요에 따라 SMOTE on/off, 하이퍼파라미터 튜닝 범위 등을 조절할 수 있습니다.

---

## Setup and data loading

```python
# !pip install imbalanced-learn xgboost

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # pipeline with samplers

# 1) Load data
df = pd.read_csv("creditcard.csv")  # Kaggle dataset file
X = df.drop(columns=["Class"])
y = df["Class"]

# 2) Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23, stratify=y
)

# 3) Preprocess: scale only non-PCA columns (Amount, Time)
scale_cols = ["Amount", "Time"]
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, scale_cols)],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
```

---

## Imbalance handling options

```python
# Toggle SMOTE usage
USE_SMOTE = True

# Build base pipeline factory (with or without SMOTE)
def make_pipeline(estimator):
    if USE_SMOTE:
        # SMOTE inside pipeline to avoid leakage (fit only on training folds)
        return ImbPipeline(steps=[
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=23)),
            ("clf", estimator)
        ])
    else:
        return Pipeline(steps=[
            ("preprocess", preprocess),
            ("clf", estimator)
        ])
```

---

## Baseline: logistic regression

```python
# 4) Baseline model
log_reg = LogisticRegression(
    random_state=23,
    max_iter=1000,
    C=0.1,
    penalty="l2",
    solver="lbfgs",
    class_weight="balanced"  # helpful even with SMOTE off; with SMOTE on you can set None
)

pipe_log = make_pipeline(log_reg)
pipe_log.fit(X_train, y_train)

# Evaluation helpers
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n=== {name} ===")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    if y_proba is not None:
        roc = roc_auc_score(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")

evaluate_model("Logistic Regression (baseline)", pipe_log, X_test, y_test)
```

---

## Tree-based models: random forest and XGBoost

```python
# 5) Random Forest
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=23,
    class_weight="balanced_subsample"  # helps with imbalance
)
pipe_rf = make_pipeline(rf)
pipe_rf.fit(X_train, y_train)
evaluate_model("Random Forest", pipe_rf, X_test, y_test)

# 6) XGBoost
# Tip: scale_pos_weight ≈ (neg / pos) on training set
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / max(pos, 1)

xgb = XGBClassifier(
    random_state=23,
    n_estimators=400,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.0,
    reg_lambda=1.0,
    n_jobs=-1,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight
)
pipe_xgb = make_pipeline(xgb)
pipe_xgb.fit(X_train, y_train)
evaluate_model("XGBoost", pipe_xgb, X_test, y_test)
```

---

## Neural network and stacking ensemble

```python
# 7) MLP (simple architecture; tune carefully to avoid overfitting)
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate="adaptive",
    max_iter=50,           # increase with early stopping if desired
    random_state=23
)
pipe_mlp = make_pipeline(mlp)
pipe_mlp.fit(X_train, y_train)
evaluate_model("MLPClassifier", pipe_mlp, X_test, y_test)

# 8) Stacking ensemble (logistic meta-learner)
from sklearn.ensemble import StackingClassifier

# Base learners should expose predict_proba for better stacking
estimators = [
    ("log", LogisticRegression(
        random_state=23, max_iter=1000, C=0.2, penalty="l2",
        solver="lbfgs", class_weight="balanced"
     )),
    ("rf", RandomForestClassifier(
        n_estimators=300, random_state=23, n_jobs=-1, class_weight="balanced_subsample"
     )),
    ("xgb", XGBClassifier(
        random_state=23, n_estimators=300, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1, objective="binary:logistic",
        eval_metric="auc", scale_pos_weight=scale_pos_weight
     )),
]

# Important: preprocess once at the top, then fit stacking on processed features
# We'll wrap stacking inside a pipeline so preprocess applies to all base learners consistently.
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(
        random_state=23, max_iter=1000, C=0.5, penalty="l2", solver="lbfgs"
    ),
    stack_method="predict_proba",
    passthrough=False,            # set True to include original features with meta-features
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
)

pipe_stack = make_pipeline(stack)
pipe_stack.fit(X_train, y_train)
evaluate_model("Stacking Ensemble", pipe_stack, X_test, y_test)
```

---

## Optional: hyperparameter tuning templates

```python
# 9) GridSearch examples (use small grids first to avoid long runs)

# Logistic Regression grid
log_grid = {
    "clf__C": [0.05, 0.1, 0.2],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "liblinear"],
    # If using SMOTE pipeline, "clf__class_weight": [None, "balanced"] can be explored
}
gs_log = GridSearchCV(make_pipeline(LogisticRegression(max_iter=1000, random_state=23)),
                      log_grid, cv=5, scoring="roc_auc", n_jobs=-1)
gs_log.fit(X_train, y_train)
print("Best LOG params:", gs_log.best_params_)

# Random Forest grid
rf_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 6, 10],
    "clf__max_features": ["sqrt", "log2"],
}
gs_rf = GridSearchCV(make_pipeline(RandomForestClassifier(
    random_state=23, n_jobs=-1, class_weight="balanced_subsample")),
    rf_grid, cv=5, scoring="roc_auc", n_jobs=-1)
gs_rf.fit(X_train, y_train)
print("Best RF params:", gs_rf.best_params_)

# XGBoost grid
xgb_grid = {
    "clf__n_estimators": [250, 400],
    "clf__max_depth": [3, 4, 5],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample": [0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.8, 0.9, 1.0],
}
gs_xgb = GridSearchCV(make_pipeline(XGBClassifier(
    random_state=23, n_jobs=-1, objective="binary:logistic", eval_metric="auc",
    scale_pos_weight=scale_pos_weight)),
    xgb_grid, cv=5, scoring="roc_auc", n_jobs=-1)
gs_xgb.fit(X_train, y_train)
print("Best XGB params:", gs_xgb.best_params_)
```

---

## Notes and next steps

- **Metric focus:** accuracy는 무의미합니다. ROC-AUC와 특히 PR-AUC, recall을 중점적으로 보세요.  
- **Threshold tuning:** 최종 배포 전에는 분류 임계값을 조정해 비용(거짓음성 비용이 큼)에 맞춘 최적의 운영점으로 설정하세요.  
- **SMOTE vs class_weight:** 동시에 쓰면 과도한 보정이 될 수 있습니다. 둘 다 켜고 결과를 비교한 뒤 더 안정적인 설정을 선택하세요.  
- **Reproducibility:** random_state를 일관되게 유지하세요.  
- 필요하면 LightGBM, CatBoost도 쉽게 추가할 수 있습니다. 원하는 모델이 있으면 말씀해 주세요.
