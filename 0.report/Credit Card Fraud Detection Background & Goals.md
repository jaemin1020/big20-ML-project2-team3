# Credit Card Fraud Detection Background & Goals

## 문제 정의 및 데이터 이해

- [ ] # **프로젝트 배경 및 목표**

        
- Kaggle 신용카드 사기 검출 데이터셋의 목적은 금융 거래에서 발생하는 사기 행위를 효과적으로 탐지하기 위한 머신러닝 모델을 개발·실습하는 것이며, 배경은 실제 금융권에서 급증하는 사기 문제와 기존 규칙 기반 시스템의 한계를 극복하기 위해 마련  
- 이번 프로젝트 실습을 통해서 주요 피처 분석은 **데이터의 불균형 문제를 이해**하고, 어떤 변수들이 사기 탐지에 가장 큰 영향을 주는지 파악  
  - 수업시간 보다 **모델 성능을 고도화**하고, **실제 금융 보안 전략에 적용**할 수 있는 인사이트 도출을 목표로 설정

1) ## 목적

* **사기 거래 탐지 모델 개발** : 정상 거래와 사기 거래를 구분하는 분류 모델을 학습·평가  
* **불균형 데이터 처리 연습** : 전체 거래 중 사기거래가 약 0.173%에 불과  
  * 데이터 불균형 문제를 다룰 수 있는 좋은 실습 예제  
* **머신러닝 기법 비교** : 로지스틱 회귀, 랜덤포레스트, XGBoost, LightGBM 등 다양한 알고리즘을 적용해 성능 비교  
* **실무 적용 가능성 검토** : 실제 금융 서비스에서 발생하는 \*False Positive(정상 거래를 사기로 잘못 탐지)와 \*False Negative(사기를 놓치는 경우)의 비용을 고려한 모델 설계


2) ## 배경

* **금융권의 현실적 문제** : 신용카드 사기로 매년 전세계적으로 수십억 달러 규모의 손실 유발하며, 소비자와 기업 모두에게 큰 피해 발생시킴  
* **기존 시스템의 한계** : 전통적 규칙 기반(rule-based) 탐지 시스템은 새로운 사기 패턴에 적응하기 어렵고, 오탐(false positive)이 많아 고객 불편을 초래  
* **데이터셋 구성 이유** : 실제 금융 거래 데이터를 익명화(PCA 변환)하여 제공, 연구자와 개발자가 개인정보 침해 없이 모델을 실습할 수 있도록 설계된것으로 추측  
* **연구 및 교육 목적** : 머신러닝, 데이터 불균형 처리(SMOTE, 언더샘플링, 오버샘플링) 기법을 학습하는 대표적 교육용 데이터셋

3) ## 문제 정의 및 중점 정리

* 데이터셋은 **카드 거래 데이터**를 기반으로 하며, 개인정보 보호를 위해 **PCA로 변환된 변수(V1\~V28)**를 포함  
  * 주요 피처를 중심으로 모델을 최적화 → 탐지 성능 고도화  
* **Class 변수** : 이번 프로젝트 실습 데이터셋의 **레이블(답)**이며, 0 \= 정상거래, 1 \= 사기 거래  
* 금융 기관은 단순히 “모델이 사기라고 판단했다” 가 아니라, *어떤 변수 때문에 사기 가능성이 높다고 판단했는지* 설명할 수 있어야함

- [ ] # **데이터셋 문제와 평가지표 선정**

        
- Kaggle 신용카드 사기 검출 데이터셋은 극심한 클래스 불균형 이라는 문제를 가지고 있으며, 단순 정확도(Accuracy)로는 성능을 제대로 평가할 수 없음  
- 따라서, **Precision, Recall, F1-score, F2-score ROC-AUC, PR-AUC** 같은 불균형 데이터에 적합한 평가지표를 선정이 요구됨

1) ## 데이터셋 문제점

* **극심한 불균형** : 전체 284,315건 중 사기 거래는 492건(0.173%)에 불과  
  * 대부분 정상 거래로 학습되며, 모델이 “전부 정상” 이라고 예측해도, Accuracy가 99% 이상 나올 수 있음  
* **데이터 익명화** : 개인정보 보호를 위해 PCA로 변환된 변수만 제공   
  * 실제 의미 해석 어려움  
* **시간적 특성 미반영** : 거래 시간(Time) 변수 있지만, 실제 시계열적 패턴을 반영 제한적  
* **샘플링 필요성** : 불균형 문제 해결을 위해 **언더샘플링, 오버샘플링(SMOTE)** 기법 필요

![][image1]  
\[ 그림 \] 신용카드 사기검출 데이터셋 Class 피처 실제 사기와 정상 거래 분포 시각화

2) ## 평가지표 선정

* **비즈니스 목표에 따라 지표 선택** :  
  * 은행은 Recall 을 높여 사기를 놓치지 않게 하는 것이 중요  
  * 하지만, Precision 이 낮으면 정상 거래가 과도하게 차단되어 고객 불편 발생 초래  
* **Threshold 조정** : 모델의 예측 확률을 조정해 Precision vs Recall 균형을 맞추는 전략 필요  
* **비용 기반 평가** : False Negative(사기 놓침)과 False Positive(정상 거래 차단)의 비용을 고려한 Cost-sensitive Learning 도 활용 가능

- 불균형 데이터셋에서 모델이 대다수 클래스만 잘 맞추고, 소수 클래스는 완전히 무시하는 ‘정확도의 역설(Accuracy Paradox)’ 현상이 발생할 수 있음  
- 정확도의 한계를 극복하고 모델이 희소 클래스를 얼마나 잘 처리하는지 종합적으로 판단하기 위해 다음 지표들이 요구됨

\< 표 \> 불균형 데이터에서의 다양한 평가지표 필요성

| 지 표 | 정 의 | 필 요 성 |
| :---: | ----- | ----- |
| **Accuracy** | 전체 예측 중 맞춘 비율 | 불균형 데이터에서는 대부분 ‘정상’으로 예측해도 높은 값이 나오므로 의미 제한적 |
| **Precision** | 예측한 Positive 중 실제 Positive 비율 | False Positive(정상 거래 사기로 잘 못 탐지)를 줄이는데 중요 → 고객불편 최소화 |
| **Recall** | 실제 Positive 중 모델이 잡아낸 비율 | False Negative(사기를 놓침)를 줄이는데 중요. 사기 탐지에서 핵심 지표 |
| **F1-score** | Precision과 Recall의 조화 평균 | Precision과 Recall 간 균형 평가. 불균형 데이터에서 단일 지표로 성능 비교 |
| **F2-score** | Recall에 더 큰 가중치를 둔 조화평균 (Recall을 Precision 보다 2배 중요시) | 사기 탐지에서 놓치면 큰 피해가 발생하는 문제에서 Recall을 더 강조할 때 적합 |
| **ROC-AUC** | 다양한 Threshold에서의 TPR vs FPR 곡선 아래 면적 | 전체 분류 성능을 평가하지만, 극단적 불균형에서는 과대평가 가능 |
| **PR-AUC** | Precision-Recall 곡선 아래 면적 | Positive 클래스(사기거래)에 집중한 평가. 불균형 데이터에서 더 적합한 지표 |

3) ## 데이터셋 문제 및 핵심 지표 정리

* **Accuracy는 불균형 데이터에서 신뢰할 수 없음**  
* **Precision ↔ Recall 균형** → **F2-score**  
* **Recall**을 더 강조해야 하는 경우 → F2-score

- 따라서 단일 지표에 의존하기 보다는 재현율, 정밀도, F1점수, F2점수 등을 함께 사용하여 모델이 희소하지만 중요한 사건을 얼마나 효과적으로 감지하는지 다각도로 평가가 요구됨

- [ ] # **데이터 설명 및 구조**

1) ## 데이터셋 개요

* 출처 : [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
* 총 거래 수 : 284,807 건  
* 사기 거래 수 : 492건 (0.172%) → 극도로 불균형한 데이터  
* 특징(Features) : 30개 변수  
  * 대부분 PCA(주성분 분석)로 변환된 익명화된 변수 (V1 \~ V28)  
  * ‘Time’ : 첫 거래로부터 경과 시간  
  * ‘Amount’ : 거래 금액  
  * ‘Class’ : 레이블 (0 \= 정상, 1 \= 사기)

    

2) ## 데이터 구조

* 입력변수(Features) : Time, V1\~V28, Amount  
* 출력변수(Target) : Class  
- 익명화 처리된 V1\~V28은 PCA 변환되어 해석이 불가능하므로, 단순히 모델 입력값으로만 사용  
- 이번 신용카드 사기검출 데이터셋은 극도로 불균형하므로, 모델 학습 시 샘플링 기법을 고려해보아야 함


\< 표 \> 데이터 구조 및 컬럼 설명

| 컬럼명 | 설 명 |
| :---: | ----- |
| **Time** | 첫 거래로부터 경과 시간(초 단위) → 데이터셋 내 거래 순서를 나타냄 |
| **V1 \~ V28** | PCA(주성분 분석) 변환된 익명화된 변수들 : 원래는 카드 소유자의 민감한 정보 였으나 익명화 |
| **Amount** | 거래 금액, 모델 학습 시 중요한 변수 중 하나 |
| **Class** | 레이블(목표 변수, Target). 0 \= 정상 거래, 1 \= 사기 거래 |

- [ ] # **주요 피처분석**

        
- 데이터의 불균형 문제를 이해하고, 어떤 변수들이 사기 탐지에 가장 큰 영향을 주는지 파악  
- describe() 결과를 기반으로 시각화를 통해 변수의 분포, 스케일 차이, 클래스 불균형을 직관적으로 확인 가능.  
    
* 데이터 특성 이해  
  * 대부분의 피처가 PCA(주성분 분석)으로 변환된 익명 변수  
  * 원래 의미를 알 수 없지만, 변수 간 분포와 상관관계를 분석하는 것이 필요  
* 클래스 불균형 문제  
  * 전체 거래 중 사기 거래는 약 0.17%에 불과  
  * 불균형을 고려하지 않으면 모델이 정상 거래만 예측하는 쪽으로 치우칠 수 있음  
  * 따라서, 피처 중요도 분석은 소수 클래스(사기 거래)를 잘 구분하는 변수를 찾는데 핵심  
* 모델 해석 가능성 확보  
  * 이 주제에 적합한 알고리즘(트리 계열 모델)을 활용하여 각 피처의 중요도를 계산  
  * 사기 거래 탐지에 기여하는 주요 변수를 확인하고, 금융기관이 규칙 기반 탐지 시스템을 개선하는데 활용할 수 있음


- **데이터 불균형 문제 해결, 사기 패턴 이해, 실무 적용 가능성 확보** 세 가지 측면에서 매우 중요한 역할을 함

1. Amount (거래 금액)  
   1. 거래 금액은 사기 탐지에서 직관적으로 중요한 변수  
   2. 사기 거래는 특정 금액 구간에서 집중되는 경향  
   3. 모델 학습 시 로그 변환이나 표준화를 통해 스케일링 조정 필요

      

2. Time (거래 시간)  
   1. 거래 발생 시점  
   2. 단독으로는 큰 의미 없지만, 시간대별 패턴 분석에 활용 가능

      

3. PCA 변환 변수 (V1 \~ V28)  
1. 원래 민감한 카드 정보와 거래 특성을 PCA로 변환한 값  
2. 직접적인 해석은 불가능하지만, 통계적 분포와 상관관계를 통해 중요변수 파악  
3. 각 피처들과 사기 거래의 상관도 분석을 통해 상관성 높은 주요 피처 도출

![][image2]  
\[ 그림 \] 데이터셋 피처들과 사기 거래 상관도 분석 중 Top10 주요 피처  


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAloAAAFNCAYAAADYYMFUAAAsbUlEQVR4Xu3dC7TVZZ3/8VREEhE1LgILDjcRUkiDBsWZwQsz2ChIgOmYa4EmBAmEMYoSI16ILEQYnWSiEsFrKmgSalyGyzIGR0HFyiQxBIGSm2PAAbl8///P4+xfv9/D4fA8yHHiPO/XWp/F3s9vn30Om/L7Wc/vt/f5jAEAAKBKfMZfAAAAwOFB0QIAAKgiFC0AAIAqQtECAACoIhQtAACAKkLRAgAAqCIULQAAgCpC0QIAAKgiFC0A+/nMZz5jTz/9tL8c5ZM+x9SpU61u3br+8l+tYcOG+UufSO/evbPb/fr1s8suuyx39PD4r//6L3vppZf8ZQCHEUULqMZUVg7FJy1J4j/Ht771LbvmmmtyjzA7/vjj7V/+5V9s8eLF9tZbb7k/x44day+88EKVFy3/Z6nIggUL3N/j9ddftz/84Q/229/+1q6//nqrWbOmfe1rX7O9e/f6X3JQ+jvF/rscjqKl179r167+MoAqRtECqrHYgV7il6RDkX+OP/7xj1a7dm1XVkoeeOABe/nll7P7vqouWvpZ9HNVplS0tm7dWljXTtBnP/tZGz9+fGE9xMGK1p49e/wlihZwBKNoAdVYZQN99erV1rx5c1cY2rdvb7/4xS+yYyoXM2fOtB/84AfWrFkzO+644+ycc85xBaNk3759NmHCBGvRooXb4ZkzZ052TPJFa9y4cdarV6/sWHl5uTVs2DC7XxG/aKmATJw40U444QRr3Lixffvb37aPPvooO37VVVdZ/fr13S5Z586dbceOHW5dJalRo0ZWq1Yt69Chg7377rvZ13z3u9/NblfkQEVLRo0aZU2aNMnuq8iU6PGDBw/Ovq++p/4+eq58xowZk5Wohx9+2Bo0aOB+dtF6SekxS5cutXPPPdeaNm1qd911V3Zc/BKln0HfQ38HyX/fsrIyt/bqq68Wyu/u3bvda9KmTRv3bzpw4EDbvHlzdlyP1dd/4xvfsJNOOslOPfVUu+mmmyoshwA+RtECqrHKitZ9991nL774ov3617+2QYMGuYKyfv16d0zDtEuXLnbllVfar371K1uyZIldeumlVqdOnezrR44c6YatvodOrWkwr1q1KjueL1oXXnihTZ48OTumU4M1atTI7lfEL1oqDiohb775pv3sZz9zP++kSZOy43379nU/x3//93/b7bffbtu2bXPrPXr0cGvLly93Re2dd97JvkY/V2UqK1p6Th0rFZV80dL3/Pu///vs++p7/vnPf3aP1WuonTDd1vOqRJ199tnWrVs39/hly5a55/CLlkqi8tRTT7kCfOyxx9r999+fPeZgRUunSvX66fuuXbvWrflF64orrnBl9cEHH7Q33njDTjvtNOvYsWNWaEtF63vf+557rf/93//d/Tv+x3/8R/YcAIooWkA1VlnR2rVrV3Z7586dbtdKBUY0TM8///zsuOjxpZ0QDdyjjz7annzyyey4SouuXyrJFy2dNtRgLrnnnnusVatW2f2K+EVL10Pld05UPr785S+72xs3bnTXd1VE3/tAKjsmlRUtnXbUsdIuX75o6XmnT5+e3c/zTx3q76HStGHDhr886H/X87dVZFeuXJmt3XLLLW5nr+RgRauiU4f5oqXXT49XqS5RQdTP9sgjj7j7paKVp4J70UUXFdYA/AVFC6jGKitaoiGsXRGdhtMpxH/7t39z6xqmFe1S6FSS/OhHP3K7WfmLwXUa7G//9m+z+/mipdvvv/9+dkynEtu2bZvdr4hftEoeeugh973OOusstxMkOo3ZqVOnwqnNku7du7sdtIro5yrtfFWksqKlXSEd046V5IuWvmfr1q0r/L4VFa2/+Zu/+csDcuv52zp1m6dClH9d/RIVW7R0KvTzn/984bhoZ670715R0brzzjvdqUYAFaNoAdXYgYrWli1bXLHK0ykjnVoTDdPSLkbeV7/6VfenThlpN0Wn8fJ5++23s8f6RSt/rY+uWdKOWGX8oqXTWiowJTfeeKN94QtfyO6LioCu1dIujH6ekj/96U82fPhwO+qoo2zu3LnZun6uDz/8MLvvq6xo/fM//3O2oyb5olVS+r7571lR0aroQne/aP3DP/zDXw7+f88991zhdc2/NlIqgqFFS6/nmWeeWTgu+ppvfvOb7nZFRUv/myntdALYH0ULqMYOVLRK1xeVvPbaa+5+vmhdffXV2XHRQC+datPjVZR+97vfFR6Tly9aOi2pj0bI69+/v61Zs6awlucXLe226KMfRDtpX/ziF/crWqLdrZNPPtl+8pOf+Ifc40ePHp3d14XqlTlQ0dJz6xqxFStWZGsVFa2S/PesV69eYbcwtGhpB3HTpk3Zmoqerp8que66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/AFLiAIAiBfSMShaifEHLCEKACBeSMegaCXGH7CEKACAeCEdg6KVGH/AEqIAAOKFdAyKVmL8AUuIAgCIF9IxKFqJ8QcsIQoAIF5Ix6BoJcYfsIQoAIB4IR2DopUYf8ASogAA4oV0DIpWYvwBS4gCAIgX0jEoWonxBywhCgAgXkjHoGglxh+whCgAgHghHYOilRh/wBKiAADihXSMqKK1b98+mzZtmp1zzjnZ2vLly61z587WrFkza9eunc2ZMyc7NnHiRGvVqpU1btzYevXqZZs2bXLr+rNv377WtGlT93V333139jVz5861Dh06uGMdO3a0ZcuWufW9e/fazTffbGVlZe75rrnmGisvL8++7kBCXoSU+AOWEAUAEC+kYwQXreeff97OPPNMV5xOP/30bL1JkyauHMnChQutbt26tmHDBnf/7LPPts2bN9uePXts0KBB1rt3b7f+5S9/2W677TZX3NatW+fK07PPPuuONWzY0F5//XV3+5FHHnHPr0L1/e9/3y6++GLbsWOH7dy50y699FL79re/7R5XmZAXISX+gCVEAQDEC+kYwUWrZMGCBYWiNX78+NxRs7Fjx1r//v1t9+7dtnHjxsKx1q1buzJ2ySWXFNZffPFFa968ubv95JNPFo5dd911rpSdcMIJrpiVaIerZs2atnr16tyj9xfyIqTEH7CEKACAeCEd4xMXrenTp+eOmk2ZMsV69OiR7WrldenSxWbMmGHXXnttYX3lypVWp04dd3vx4sWFY6NGjbKhQ4day5YtC+uiU4ilU4t52vHSX1pZu3btQV+ElPgDlhAFABDvUylaumYrb/LkydazZ09XcPI7UKJruWbOnOmur8p788037cQTT3S3Fy1aVDg2cuRIGzZsmLVo0aKwLjrNqGvEfGPGjHF/8XwqexFS4g9YQhQAQLxPpWhNmDAhd9TsjjvusIEDB9q2bdvc9Vl52pVasmSJ2/HK0y5WmzZt3G3teOVp92vcuHHZjleJSlyNGjVs/fr1hXVhR+vA/AFLiAIAiPepFK0+ffrkjpq7YH3q1Knu9qxZs7J1FaLatWvb9u3brUGDBu4aq5K77rrL+vXr526PGDEiW5e2bdu676mi9cYbb2TrS5cudRfRH0zIi5ASf8ASogAA4oV0jE9ctE466SSbP3++uz179mxXfrSbJZ06dbKtW7farl27XJEa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33Dre/fudffHjRvn/ly3bp2dcsop1rlz54+fBEkJ6RgUrcT4A5YQBYBZkyZNrH379lnR6tu3b+H48ccfbxMnTiysyahRo9yOF9IT0jEoWonxBywhCpCylStXWo0aNWz27NmuSJWKltaee+45t4s1evRoN0u6deu239dq52vo0KGFdaQhpGNQtBLjD1hCFCBlp556qvXp08fdzhetK6+80l2jpcLVrl07a9CggX3lK1/Jvu7xxx93Jeu8887L1pCWkI5B0UqMP2AJUYCU1a9fP7v2Kl+0fDp1eOedd7rb/fv3dxfJDx8+3HsUUhLSMShaifEHLCEKkDLNiIqSp1OIWiu9u1DvNpw3b17hMUhPSMc4oorWjh07bMCAAdasWTN30eKNN95o+/bt8x9WEPIipMQfsIQoAD6W39F6/fXX3Z+vvvqq1alTp3DacMiQIdltpCukYxxRRWvw4MH29a9/3Xbv3m0ffPCBderUye69917/YQUhL0JK/AFLiALgY/7F8Nq50v3LL7+88DidNtSxfN55553CY1D9hXSMI6ZoLVy40M4444zC2nvvvef+j1CZkBchJf6AJUQBAMQL6RhHTNF67LHH7MILLyysffTRR/udR5edO3e6v7SyZs0a95i1a9dmaymn6fAnCNkv/v9OCCGEHDzqFuoYOst2IPu3lL9SDz30kF1wwQWFtfLy8gqL1pgxY/a7qJEQQgghpCqiwnUg+7eUv1L6IDl9Ym+edqtq1apVWJP8jtbWrVtt1apVrm36TZSkndL/Ofx1QgjxU9q54L8ZJB91C/1vovTxIBU5YorWhg0b3AWJW7Zsydb0YXFdu3b9y4OACPqPpv6PAgAHo/9W8N8MHIojpmhJz5493W9T17sON27c6Ha4nn76af9hQBD+owkgFEULh+qIKloqVypb9erVs7KyMrvvvvv8hwDB+I8mgFAULRyqI6poAQAAHEkoWgAAAFWEogUAAFBFKFoAAABVhKIFAABQRShaSMq+ffvch9hW9usSAAA4XChaqPZuuukm+9KXvmQzZ860devWuc9h028P0AfedujQwe6//37/SwAAOCwoWqj2mjZtah9++KG/7Oiz2Vq0aOEvAwBwWFC0UO3pw211yrAie/bssfr16/vLABJ37rnnWrt27Q4YIBRFC9XewIEDbcCAAftdl6X73/zmN+2yyy4rrAPAtGnTbOHChQcMEIqihWqvvLzcBg8ebLVr13a/H7Nbt2520UUXWa1ataxXr162adMm/0sAwJ5//nl/CYhG0UIytm3bZosWLXIXwT/88MP2xz/+0X8IAGT0Bhrgk6JoAQAAVBGKFgAAQBWhaAEAAFQRihaAak/vML3lllvc2/KbN29uDRo0sM6dO9txxx3nPxQADiuKFoBqTwXr9ttvzz7iQ5+rtnjxYooWgCpH0QJQ7Y0bN85fcihaAKoaRQtAtbZ69Wr3WWoVyRetuXPnutOJzZo1cyl566233GevtWzZ0k499dQK15544ons8QCQR9ECUK09+uij/lLG39Hau3evrVy50p577jmbMWOGW1OhGjJkiG3fvj17XEVrAFARihaAau2ll15yv9OyIvmiNX78eGvTpo317t3bRowYYY899phb37Bhg/Xr188aNmxoI0eOrHBtx44d2fMAQB5FC0C1pl2qH/7wh/6yUypab7/9tp188sn24YcfZsdKRavk/ffft0suuaTCtUGDBhXWAaCEogWg2qtXr5498MADtmvXLnd/9+7d9uyzz2ZF67e//a3VqVPH3nvvPXdfv1C4VLTmzZvnyprcdtttFa5dffXV7jYA+ChaAKq9FStWWJ8+faxJkyZWVlbmcuWVVxZOHd56663WqFEjd4H7d77znaxo9e3b1+rXr2+nnXaaXXbZZRWurV+/PnseAMijaAEAAFQRihYAAEAVoWgBAABUEYoWAABAFaFoAQAAVBGKFgAAQBWhaAEAAFQRihYAAEAVoWgBAABUEYoWAABAFaFoAQAAVJH/B/8OKHrE7C/dAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,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>