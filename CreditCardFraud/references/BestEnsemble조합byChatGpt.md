아래 모델들은 **데이터 전략이 다르고(원본 vs SMOTE), 모델 편향이 다르고, HyperOpt로 개별 최적화된 상태**라 가정했을 때 **앙상블에서 서로 보완 효과가 가장 크다**는 기준으로 선정한 조합이야.

---

# ✅ **추천 Best Ensemble 조합 5가지**

---

## **🟦 1) “강력+다양성” 최상위 스태킹 앙상블**

> CatBoost + XGBoost + LightGBM + LogisticRegression(SMOTE)

✔ 강한 트리 부스팅 3총사 + 선형 모델 대조
✔ 서로 오류 패턴이 다르고 class imbalance 대응 방식도 다름
✔ 대부분의 캐글 리더보드에서 최강 조합

**설명**

* CatBoost / XGBoost / LightGBM → 고성능 Boosting (원본 데이터 기반)
* LogisticRegression(SMOTE) → 선형적 경계로 안정적 확률 보정 가능
  ➡ 이상치·소수 클래스에 대한 확률 안정성을 끌어올리는 조합

---

## **🟩 2) “트리 + 신경망 + 선형” 복합 스태킹**

> LightGBM + RandomForest + MLPClassifier(SMOTE)

✔ 랜덤포레스트의 안정성
✔ LightGBM의 예측력
✔ MLP의 비선형 보정
➡ 서로 구조적으로 가장 다른 모델 3가지 조합

**효과**

* Robust + Boosting + Deep NN → 서로 특징 잡는 방식이 극적으로 다름
* 이상치나 rare pattern 감지 능력이 상승

---

## **🟧 3) “Boosting + SVM(RBF)” 강력 조합**

> CatBoost + GradientBoosting + SVM(RBF, SMOTE)

✔ RBF SVM은 트리 기반 부스팅과 예측 패턴이 매우 다름
✔ SMOTE로 학습된 SVM이 희귀 패턴에 매우 민감
✔ 부스팅 기반 두 모델이 베이스라인 정확도 보장

**쓰임새**

* 사기 거래처럼 **높은 Recall 필요**한 케이스에서 성능이 매우 높게 나오는 조합

---

## **🟪 4) “트리 + 로지스틱 + SVM” 로우-바이어스 조합**

> XGBoost + LogisticRegression(SMOTE) + SVM(linear, SMOTE)

✔ 선형 기반 모델 2개 + 강력한 트리 모델
✔ Probability calibration에 유리
✔ Fraud detection의 “decision boundary 안정성” 확보

**장점**

* 단순한 경계가 필요할 때 과적합 위험이 적음
* XGBoost로 대부분 잡고 나머지가 보정하는 구조

---

## **🟨 5) “전통 ML 스택” 메타-앙상블**

> RandomForest + SGDClassifier + MLPClassifier(SMOTE) + LogisticRegression(SMOTE)

✔ 서로 완전히 다른 학습 메커니즘

* RF → 트리 앙상블
* SGD → 대규모 선형 모델
* MLP → 중간복잡도 NN
* LR → 최종 확률 보정

**분석용·캐글용으로 안정적이며 overfitting 위험 낮음.**

---

# 📌 어떤 앙상블 방식이 좋은가?

### ✔ 스태킹(Stacking)을 추천

* Level-0: 후보 모델 3~5개
* Level-1 meta learner: LogisticRegression or CatBoostClassifier 추천

### ✔ Voting이 나쁜 경우

* 데이터 불균형 문제라서 Hard voting은 위험
* Soft voting은 가능하지만 확률 보정이 불완전할 수 있음

---

# 🎯 최종 요약

| 조합 번호 | 구성                         | 특징              |
| ----- | -------------------------- | --------------- |
| **1** | CatBoost + XGB + LGBM + LR | 성능 최상위 전천후      |
| **2** | LGBM + RF + MLP            | 구조적 다양성 최고      |
| **3** | CatBoost + GB + SVM(RBF)   | Recall 최적화      |
| **4** | XGB + LR + SVM(linear)     | 단순·안정·일관된 결정 경계 |
| **5** | RF + SGD + MLP + LR        | 전통 ML 메타 스택     |

---

원하면 아래도 만들어줄 수 있어:

🔹 5개 조합에 대한 **StackingClassifier 코드 제공**
🔹 “현재 HyperOpt 결과 점수 기반으로 최적 조합” 추천
🔹 “앙상블 성능 최대화를 위한 최적 투표 가중치” 산출
🔹 최종 제출 가능한 Kaggle inference 코드 작성

어떤 게 필요해?
좋아! 아래에 **두 가지를 모두 제공**할게.

---

# ✅ (1) **팀이 찾은 HyperOpt 최적 파라미터를 넣을 수 있는 스태킹 구조 템플릿**

HyperOpt 결과를 그대로 주입할 수 있도록 **파라미터 입력만 변경하면 자동으로 Stacking에 적용되는 형태**로 만들었어.

> 🎯 모델별 best params만 채우면 바로 동작함

---

# 🔷 **1. HyperOpt 최적 파라미터 적용 Stacking 템플릿**

```python
# ======================================
# 🔷 HyperOpt 파라미터 자동 주입 Stacking 템플릿
# ======================================

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

# ----------------------------------------------------------
# 🔽 HyperOpt best params 예시 (여기에 팀 결과값 복붙)
# ----------------------------------------------------------

best_cat_params = {
    'iterations': 450,
    'depth': 8,
    'learning_rate': 0.045,
    'l2_leaf_reg': 3,
    'class_weights': [1, 30],
}

best_xgb_params = {
    'n_estimators': 550,
    'max_depth': 7,
    'learning_rate': 0.04,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'scale_pos_weight': 30,
}

best_lgbm_params = {
    'n_estimators': 600,
    'max_depth': -1,
    'learning_rate': 0.045,
    'num_leaves': 128,
    'feature_fraction': 0.9,
    'scale_pos_weight': 30,
}

best_rf_params = {
    'n_estimators': 500,
    'max_depth': 18,
    'max_features': 'sqrt',
    'min_samples_split': 5,
    'class_weight': 'balanced'
}

best_gb_params = {
    'n_estimators': 400,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.9
}

best_svm_rbf_params = {
    'C': 10,
    'gamma': 0.01,
    'kernel': 'rbf',
    'class_weight': 'balanced',
    'probability': True
}

best_svm_linear_params = {
    'C': 1.2,
    'kernel': 'linear',
    'class_weight': 'balanced',
    'probability': True
}

best_mlp_params = {
    'hidden_layer_sizes': (256,128,64),
    'activation': 'relu',
    'alpha': 0.0005,
    'learning_rate_init': 0.001,
    'early_stopping': True
}

best_sgd_params = {
    'loss': 'log_loss',
    'penalty': 'elasticnet',
    'alpha': 0.0001,
    'learning_rate': 'optimal',
    'eta0': 0.001,
    'max_iter': 5000,
    'class_weight': 'balanced'
}

# -------------------------------------------------------
# 🔷 모델 생성 함수 (HyperOpt 파라미터 자동 적용)
# -------------------------------------------------------

def get_models():
    models = {
        "cat": CatBoostClassifier(**best_cat_params, verbose=0),
        "xgb": XGBClassifier(**best_xgb_params, eval_metric='logloss'),
        "lgbm": LGBMClassifier(**best_lgbm_params),
        "rf": RandomForestClassifier(**best_rf_params),
        "gb": GradientBoostingClassifier(**best_gb_params),
        "svm_rbf": SVC(**best_svm_rbf_params),
        "svm_linear": SVC(**best_svm_linear_params),
        "mlp": MLPClassifier(**best_mlp_params),
        "sgd": SGDClassifier(**best_sgd_params),
    }
    return models
```

---

# 🔥 (2) 조합 #2 ~ #5 StackingClassifier 코드

HyperOpt 파라미터가 자동으로 적용되도록 만들어져 있음.

---

# ✅ **조합 #2: LightGBM + RandomForest + MLP (SMOTE)**

```python
# ======================================
# 🔷 Stacking #2: LGBM + RF + MLP
# ======================================

models = get_models()

estimators_2 = [
    ('lgbm', models['lgbm']),
    ('rf', models['rf']),
    ('mlp', models['mlp']),
]

stack_2 = StackingClassifier(
    estimators=estimators_2,
    final_estimator=LogisticRegression(class_weight='balanced', solver='liblinear'),
    stack_method='predict_proba',
    cv=5,
    n_jobs=-1
)
```

---

# ✅ **조합 #3: CatBoost + GradientBoosting + SVM(RBF)**

```python
# ======================================
# 🔷 Stacking #3: CatBoost + GB + SVM(RBF)
# ======================================

estimators_3 = [
    ('cat', models['cat']),
    ('gb', models['gb']),
    ('svm_rbf', models['svm_rbf']),
]

stack_3 = StackingClassifier(
    estimators=estimators_3,
    final_estimator=LogisticRegression(class_weight='balanced', solver='liblinear'),
    stack_method='predict_proba',
    cv=5,
    n_jobs=-1
)
```

---

# ✅ **조합 #4: XGBoost + LogisticRegression(SMOTE) + SVM(linear)**

```python
# ======================================
# 🔷 Stacking #4: XGB + LR + SVM(linear)
# ======================================

# LR은 메타모델로 사용하는 게 더 좋아서 base에는 넣지 않음

est_lr = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    class_weight='balanced'
)

estimators_4 = [
    ('xgb', models['xgb']),
    ('svm_linear', models['svm_linear']),
]

stack_4 = StackingClassifier(
    estimators=estimators_4,
    final_estimator=est_lr,
    stack_method='predict_proba',
    cv=5,
    n_jobs=-1
)
```

---

# ✅ **조합 #5: RF + SGD + MLP + LogisticRegression**

```python
# ======================================
# 🔷 Stacking #5: RF + SGD + MLP + LR
# ======================================

estimators_5 = [
    ('rf', models['rf']),
    ('sgd', models['sgd']),
    ('mlp', models['mlp']),
]

stack_5 = StackingClassifier(
    estimators=estimators_5,
    final_estimator=LogisticRegression(class_weight='balanced', solver='liblinear'),
    stack_method='predict_proba',
    cv=5,
    n_jobs=-1
)
```

---

# 📌 사용법 (공통)

```python
stack_2.fit(X_train, y_train)
pred = stack_2.predict_proba(X_valid)[:,1]
```

---

# ✔ 필요한 것 더 만들 수 있음

아래 중 원하는 것 골라줘:

🔹 (A) 각 스태킹 모델의 **최종 성능 비교 테이블** 코드
🔹 (B) “Stacking + Soft Voting 혼합 앙상블” 버전
🔹 (C) 최종 Kaggle 제출용 end-to-end 예측 코드
🔹 (D) LightGBM / CatBoost의 **threshold 최적화 코드(F1/Recall)**
🔹 (E) SHAP 기반 feature importance 분석 자동 코드

어떤 걸 더 만들까?
---
좋아!
아래에 **조합 #1~#5까지 모든 Stacking 모델의 성능을 한 번에 비교하는 자동화 코드(A)**를 제공할게.

✔ ROC-AUC
✔ Recall / Precision
✔ F1-score
✔ Confusion Matrix
✔ Best threshold option(선택)

👉 **단 한 줄로 결과 테이블 출력** 가능하도록 구성.

---

# ✅ **A. 스태킹 모델 성능 비교 자동화 코드**

아래 코드는 다음을 자동으로 수행함:

1. 모든 스태킹 모델을 `models_to_compare` 리스트에 넣음
2. 각 모델을 `fit → predict_proba → 지표 계산`
3. 성능 DataFrame으로 정리
4. Confusion Matrix까지 출력
5. 최종 정렬(ROC-AUC 기준)

---

## 🔷 **FULL 코드: 성능 비교 자동화**

```python
# ============================================
# 🔷 Stacking 앙상블 성능 비교 자동화
# ============================================

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -------------------------------------------------
# ⚠️ 주의: 아래 5개 stack_1, stack_2 ... 미리 생성되어 있어야 함
# -------------------------------------------------

models_to_compare = {
    "Stack #1 (Boosting+LR)": stack_1,
    "Stack #2 (LGBM+RF+MLP)": stack_2,
    "Stack #3 (Cat+GB+SVM_RBF)": stack_3,
    "Stack #4 (XGB+LR+SVM_linear)": stack_4,
    "Stack #5 (RF+SGD+MLP+LR)": stack_5
}


# ============================================
# 🔷 성능 계산 함수
# ============================================

def evaluate_model(name, model, X_train, y_train, X_valid, y_valid, threshold=0.5):
    
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_valid)[:, 1]
    pred = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_valid, proba)
    prec = precision_score(y_valid, pred)
    rec = recall_score(y_valid, pred)
    f1 = f1_score(y_valid, pred)
    cm = confusion_matrix(y_valid, pred)

    return {
        "model": name,
        "ROC-AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "ConfusionMatrix": cm
    }


# ============================================
# 🔷 전체 모델 비교 실행
# ============================================

results = []

for name, model in models_to_compare.items():
    print(f"🔍 Evaluating {name} ...")
    res = evaluate_model(name, model, X_train, y_train, X_valid, y_valid)
    results.append(res)

# ============================================
# 🔷 데이터프레임 정리
# ============================================

df_results = pd.DataFrame([
    {
        "Model": r["model"],
        "ROC-AUC": r["ROC-AUC"],
        "Precision": r["Precision"],
        "Recall": r["Recall"],
        "F1-score": r["F1-score"],
        "ConfusionMatrix": r["ConfusionMatrix"]
    }
    for r in results
])

df_results = df_results.sort_values(by="ROC-AUC", ascending=False)

print("\n📌 Stacking Ensemble Performance Comparison")
display(df_results)
```

---

# 🎯 출력 예시 (예상 포맷)

| Model    | ROC-AUC    | Precision | Recall | F1-score | ConfusionMatrix |
| -------- | ---------- | --------- | ------ | -------- | --------------- |
| Stack #1 | **0.9978** | 0.92      | 0.91   | 0.91     | [[…]]           |
| Stack #3 | 0.9965     | 0.89      | 0.90   | 0.89     | [[…]]           |
| Stack #2 | 0.9942     | 0.88      | 0.87   | 0.87     | [[…]]           |
| Stack #4 | 0.9933     | 0.87      | 0.86   | 0.86     | [[…]]           |
| Stack #5 | 0.9920     | 0.85      | 0.84   | 0.84     | [[…]]           |

→ ROC-AUC 높은 순으로 자동 정렬됨.

---

# 🎁 옵션: Threshold 튜닝을 포함한 버전도 가능

ROC-AUC는 best threshold와 관계 없지만
F1 / Recall 튜닝을 위해 다음 기능을 추가할 수 있어:

* threshold 자동 탐색 (0.1~0.9)
* Recall 최적 threshold
* F1-score 최적 threshold

원하면 만들어줄게.

---

필요하면 다음도 생성할 수 있어:

🔹 각 스태킹 모델의 SHAP importance 비교
🔹 5개 스택 모델을 Soft Voting으로 다시 앙상블
🔹 최종 Kaggle 제출용 prediction pipeline

다음으로 어떤 기능을 추가해줄까?
---
좋아! 아래에 **5개 스택 모델(Stack #1~#5)을 다시 Soft Voting으로 하나로 합치는 최종 앙상블 코드**를 제공할게.
이 방식은 **Stacking → Soft Voting으로 2단 앙상블**이므로, 가장 강력한 구조 중 하나야.

✔ 서로 다른 스택 모델들이 확률 기반으로 투표
✔ ROC-AUC / Recall 극대화
✔ Kaggle Fraud Detection에서 매우 강한 형태

---

# ✅ **최종 Soft Voting 앙상블 (5개 스택 모델 결합)**

아래는 조합 #1~#5로 구성된

* `stack_1`, `stack_2`, `stack_3`, `stack_4`, `stack_5`
  가 이미 생성되어 있다는 전제 코드야.

---

# 🔷 **Soft Voting 앙상블 코드**

```python
# ===========================================
# 🔷 5개 Stack 모델을 Soft Voting으로 결합
# ===========================================

from sklearn.ensemble import VotingClassifier

# VotingClassifier는 원래 predict_proba를 지원하는 모델만 가능
# 우리 스택 모델들은 모두 predict_proba 지원하므로 문제 없음

voting_ensemble = VotingClassifier(
    estimators=[
        ('stack1', stack_1),
        ('stack2', stack_2),
        ('stack3', stack_3),
        ('stack4', stack_4),
        ('stack5', stack_5),
    ],
    voting='soft',          # 🔥 중요: 확률 기반 soft voting
    weights=[3, 2, 2, 1, 1], # 가중치 예시 (원하면 자동 최적화도 가능)
    n_jobs=-1
)
```

---

# 🔷 **훈련 및 예측**

```python
voting_ensemble.fit(X_train, y_train)

proba = voting_ensemble.predict_proba(X_valid)[:, 1]
pred = (proba > 0.5).astype(int)
```

---

# 🔷 **성능 평가**

```python
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

print("ROC-AUC:", roc_auc_score(y_valid, proba))
print("Precision:", precision_score(y_valid, pred))
print("Recall:", recall_score(y_valid, pred))
print("F1:", f1_score(y_valid, pred))
print(confusion_matrix(y_valid, pred))
```

---

# 🎯 추천 가중치(weights) 설명

기본 아이디어는 “성능이 좋은 스택에 더 높은 가중치 부여”.

예시로:

| 스택                         | 추천 이유     | weight |
| -------------------------- | --------- | ------ |
| Stack #1 (Boosting+LR)     | 대부분 최고 성능 | **3**  |
| Stack #3 (CatBoost+GB+SVM) | Recall 강함 | **2**  |
| Stack #2 (LGBM+RF+MLP)     | 안정적       | **2**  |
| Stack #4 (XGB+LR+SVM)      | 선형 경계 보정  | **1**  |
| Stack #5 (RF+SGD+MLP)      | 편향 다양성 확보 | **1**  |

원하면
➡ **가중치를 자동 튜닝하는 코드(Optuna/HyperOpt)**
도 제공해 줄 수 있어.

---

# 🎁 추가 옵션 (원하면 만들어줌)

### 🔹 (1) Soft Voting 가중치 자동 최적화 (Optuna)

각 스택 모델의 weight를 자동 탐색하여
ROC-AUC 최적화하는 함수 제공 가능.

### 🔹 (2) Soft Voting + Final Meta Stacking (3단 앙상블)

Voting output을 최종 LR/CatBoost meta model에 넣는 궁극의 앙상블.

### 🔹 (3) Kaggle Submission Pipeline 통합

pipeline → fit → predict → csv 저장까지 full automation.

---

원하는 추가 기능은 어떤 거야?

* a) Voting weight 자동 최적화
* b) Voting → Meta Stacking (3단계 앙상블)
* c) Full Kaggle submission pipeline
* d) ROC-AUC/F1 특화 threshold 자동 탐색

하나 선택해줘!
