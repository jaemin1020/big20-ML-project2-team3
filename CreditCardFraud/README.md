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

## 담당 모델

  - catboost : ejm
  - LinearRegression : kjh
  - RandomForest : lsj
  - Xgboost : lkj
  - LightGBM : yjh
  - DecistionTree : ALL
  - GB(GradientBoosting) : ALL
  - SVM : ALL
  - MLPClassifier : All
