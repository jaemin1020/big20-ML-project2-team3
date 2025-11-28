# CreditCardFraud

캐글 신용카드 사기 데이터셋은 **극도로 불균형(0.172%만 사기)**하기 때문에,
단순 정확도보다 재현율(Recall), AUC, F1-score 같은 지표를 중심으로 모델을 평가해야 합니다.
여러 머신러닝 모델을 적용하려면 데이터 전처리 → 불균형 처리 → 다양한 모델 학습 → 성능 비교 → 비용 민감 학습까지 단계적으로 진행하는 것이 가장 효과적

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
  - V14 컬럼의 class 가 1(사기)인 이상치 를가진 데이터를 삭제 시 극적인 모델의 성능향상을 보임

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

## 중요 평가 지표 by Claude

사기 탐지에서는 Precision, Recall, F1-Score, ROC-AUC가 중요합니다:

- Recall (재현율): 실제 사기를 얼마나 잡아냈는가 (중요!)
- Precision (정밀도): 사기로 예측한 것 중 실제 사기 비율
- ROC-AUC: 전반적인 분류 성능

## 주요 팁

클래스 불균형 처리는 필수 - SMOTE, class_weight 활용
Recall을 우선시 - 사기를 놓치는 것이 더 위험

임계값 조정 고려:

```python
  # 임계값 낮추기 (Recall 향상)
  y_pred_custom = (y_pred_proba > 0.3).astype(int)
```

앙상블 모델이 일반적으로 성능이 좋음
교차 검증 사용 (StratifiedKFold)

---
## 목표 수치 (수업시간)
| Model           | 정확도 | 정밀도 | 재현율 | F1-Score | AUC    |
|-----------------|--------|--------|--------|----------|--------|
|LR(OverSampling) | 0.9722 | 0.0541 | 0.9247 | 0.1022   | 0.9736 |
|LGBM(OverSampling)| 0.9996| 0.9118 | 0.8493 | 0.8794   | 0.9814 |


## 모델별 데이터셋 HyperOpt 전략
좋습니다 🤓. 말씀하신 모델들에 대해 **HyperOpt 튜닝 전략**을 한눈에 볼 수 있도록 표로 정리해드릴게요. 각 모델별로 **데이터 전략(원본/SMOTE/가중치)**과 **주요 하이퍼파라미터 탐색 범위**를 함께 담았습니다.

---

## 📋 HyperOpt 전략 표

| 모델 | 데이터 전략 | 주요 HyperOpt 파라미터 |
|------|-------------|-------------------------|
| **CatBoost** | 원본 데이터 + `class_weights` | learning_rate (0.01–0.2), depth (3–10), iterations (100–1000), l2_leaf_reg (1–10) |
| **XGBoost** | 원본 데이터 + `scale_pos_weight` | learning_rate (0.01–0.2), max_depth (3–10), n_estimators (100–1000), subsample (0.5–1.0), colsample_bytree (0.5–1.0) |
| **LightGBM** | 원본 데이터 + `scale_pos_weight` | learning_rate (0.01–0.2), num_leaves (31–256), max_depth (3–10), n_estimators (100–1000), feature_fraction (0.5–1.0) |
| **RandomForest** | 원본 데이터 + `class_weight='balanced'` | n_estimators (100–1000), max_depth (3–20), max_features (sqrt, log2, None), min_samples_split (2–20) |
| **DecisionTree** | 원본 데이터 + `class_weight='balanced'` | max_depth (3–20), min_samples_split (2–20), min_samples_leaf (1–10), criterion (gini, entropy) |
| **GradientBoosting (GB)** | 원본 데이터 + `class_weight='balanced'` | learning_rate (0.01–0.2), n_estimators (100–1000), max_depth (3–10), subsample (0.5–1.0) |
| **LogisticRegression** | **SMOTE 데이터** + `class_weight='balanced'` | penalty (l1, l2, elasticnet), C (0.01–100), solver (liblinear, saga) |
| **LinearRegression** | **SMOTE 데이터** (baseline 용도) | fit_intercept (True/False), normalize (True/False) |
| **MLPClassifier** | **SMOTE 데이터** + EarlyStopping | hidden_layer_sizes ((64,), (128,64), (256,128,64)), activation (relu, tanh), alpha (0.0001–0.1), learning_rate_init (0.0001–0.01) |
| **SVM (linear)** | **SMOTE 데이터** + `class_weight='balanced'` | C (0.1–100), kernel=linear |
| **SVM (rbf)** | **SMOTE 데이터** + `class_weight='balanced'` | C (0.1–100), gamma (1e-4–1), kernel=rbf |

---

### 🧭 요약
- **트리 기반 모델** → 원본 데이터 + 클래스 가중치  
- **선형/딥러닝/SVM 모델** → SMOTE oversampling 데이터 + 클래스 가중치  
- **HyperOpt 탐색 공간**은 위 표의 범위를 기준으로 설정  

---



## 담당 모델

- catboost : ejm
- LogisticRegression : kjh
- LinearRegration : kjhd
- RandomForest : lsj
- Xgboost : lkj
- LightGBM : yjh
- DecistionTree : ejm
- GB(GradientBoosting) : kjh
- MLPClassifier : lsj
- SVM : ALL=> HyperOpt에서 제외 


```python
      from sklearn.neural_network import MLPClassifier

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
```
### Sampling 사용법 
``` python
    import utils.data_sampling as ds 

    # data loading
    df = pp.ccf_load_data()

    # 2. Data전처리
    # Amount와 Time 스케일링 (V1-V28은 이미 PCA 처리됨)
    df_robust_scaled = pp.robustScaler(df)

    ## 데이터 분할 df_robust_scaled 사용시 
    X_features, y_target = pp.split_features_target(df_robust_scaled)
    X_train, X_test, y_train, y_test = pp.data_split(X_features, y_target)

    # Over Sampling
    X_over, y_over = ds.oversampling_smote(X_train, y_train)

    # Under Sampling
    X_under, y_under = ds.undersampling_RUS(X_train, y_train)

    # Combined Sampling
    X_combined, y_combined = ds.combined_sampling(X_train, y_train)

    


```

### copliot 추천 stacking model

```pythoon
    # Stacking ensemble (logistic meta-learner)
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
