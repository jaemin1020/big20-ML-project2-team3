# 클래스 불균형 처리 방법으로서의 data sampling utils
import pandas as pd
from imblearn.over_sampling  import SMOTE
from imblearn.combine        import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler  
from imblearn.under_sampling import TomekLinks
    
def oversampling_smote(X_train, y_train, random_state=23, sampling_strategy='auto', k_neighbors=5):
    """
    SMOTE를 이용한 오버샘플링
    
    소수 클래스의 샘플을 합성하여 생성하는 방법으로, 
    기존 샘플의 k-nearest neighbors를 이용해 새로운 샘플을 만듦
    
    Parameters:
    -----------
    X_train : DataFrame or array-like
        학습 데이터의 특성
    y_train : Series or array-like
        학습 데이터의 타겟
    random_state : int, default=23
        재현성을 위한 랜덤 시드
    sampling_strategy : str or dict, default='auto'
        - 'auto': 소수 클래스를 다수 클래스 수준으로 맞춤
        - 'minority': 소수 클래스만 샘플링
        - dict: {class_label: n_samples} 형태로 직접 지정
    k_neighbors : int, default=5
        새 샘플 생성 시 참고할 이웃 수
    
    Returns:
    --------
    X_resampled : DataFrame or array
        리샘플링된 특성 데이터
    y_resampled : Series or array
        리샘플링된 타겟 데이터
    
    장점:
    -----
    - 정보 손실 없음
    - 다수 클래스 데이터 유지
    - 과적합 위험이 언더샘플링보다 낮음
    
    단점:
    -----
    - 학습 시간 증가
    - 노이즈에 민감할 수 있음
    
    사용 예시:
    ----------
    >>> X_smote, y_smote = oversampling_smote(X_train, y_train)
    >>> print(f"Before: {len(y_train)}, After: {len(y_smote)}")
    >>> print(y_smote.value_counts())
    """

    
    try:
        smote = SMOTE(
            random_state=random_state,
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # 샘플링 결과 출력
        print(f"✅ SMOTE 오버샘플링 완료")
        print(f"   원본 샘플 수: {len(y_train)} (Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)})")
        print(f"   샘플링 후: {len(y_resampled)} (Class 0: {sum(y_resampled==0)}, Class 1: {sum(y_resampled==1)})")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"❌ SMOTE 오버샘플링 중 오류 발생: {e}")
        return X_train, y_train
# eof ----------------------------------------------------------- #


def undersampling_RUS(X_train, y_train, random_state=23, sampling_strategy='auto'):
    """
    RandomUnderSampler를 이용한 언더샘플링    
    다수 클래스의 샘플을 무작위로 제거하여 클래스 균형을 맞추는 방법
    
    Parameters:
    -----------
    X_train : DataFrame or array-like
        학습 데이터의 특성
    y_train : Series or array-like
        학습 데이터의 타겟
    random_state : int, default=23
        재현성을 위한 랜덤 시드
    sampling_strategy : str or dict, default='auto'
        - 'auto': 다수 클래스를 소수 클래스 수준으로 맞춤
        - 'majority': 다수 클래스만 샘플링
        - dict: {class_label: n_samples} 형태로 직접 지정
    
    Returns:
    --------
    X_resampled : DataFrame or array
        리샘플링된 특성 데이터
    y_resampled : Series or array
        리샘플링된 타겟 데이터
    
    장점:
    -----
    - 학습 시간 단축
    - 메모리 효율적
    - 구현이 간단하고 빠름
    
    단점:
    -----
    - 정보 손실 (다수 클래스의 중요한 샘플 제거 가능)
    - 데이터가 적을 경우 성능 저하
    
    주의사항:
    ---------
    데이터가 충분히 많을 때 사용 권장. 
    소수 클래스 샘플이 매우 적다면 오버샘플링 고려.
    
    사용 예시:
    ----------
    >>> X_under, y_under = undersampling_RUS(X_train, y_train)
    >>> print(f"Before: {len(y_train)}, After: {len(y_under)}")
    >>> print(y_under.value_counts())
    """
    
    try:
        rus = RandomUnderSampler(
            random_state=random_state,
            sampling_strategy=sampling_strategy
        )
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
        # 샘플링 결과 출력
        print(f"✅ 랜덤 언더샘플링 완료")
        print(f"   원본 샘플 수: {len(y_train)} (Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)})")
        print(f"   샘플링 후: {len(y_resampled)} (Class 0: {sum(y_resampled==0)}, Class 1: {sum(y_resampled==1)})")
        print(f"   제거된 샘플: {len(y_train) - len(y_resampled)}개")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"❌ 언더샘플링 중 오류 발생: {e}")
        return X_train, y_train
# eof ----------------------------------------------------------- #


def combined_sampling(X_train, y_train, random_state=23, sampling_strategy='auto'):
    """
    SMOTETomek을 이용한 혼합 샘플링
    
    SMOTE로 오버샘플링 후, Tomek Links로 경계선의 노이즈 제거
    두 방법의 장점을 결합한 접근법
    
    Parameters:
    -----------
    X_train : DataFrame or array-like
        학습 데이터의 특성
    y_train : Series or array-like
        학습 데이터의 타겟
    random_state : int, default=23
        재현성을 위한 랜덤 시드
    sampling_strategy : str or dict, default='auto'
        SMOTE의 샘플링 전략
    
    Returns:
    --------
    X_resampled : DataFrame or array
        리샘플링된 특성 데이터
    y_resampled : Series or array
        리샘플링된 타겟 데이터
    
    작동 원리:
    ----------
    1. SMOTE: 소수 클래스를 합성 샘플로 증가
    2. Tomek Links: 서로 다른 클래스의 가장 가까운 이웃 쌍 제거
       → 클래스 경계를 명확하게 만듦
    
    장점:
    -----
    - SMOTE의 오버샘플링 장점 + 노이즈 제거
    - 클래스 경계가 더 명확해짐
    - 모델 성능 향상 가능
    
    단점:
    -----
    - 처리 시간이 가장 오래 걸림
    - 매개변수 튜닝이 복잡
    
    적합한 상황:
    ------------
    - 데이터에 노이즈가 많을 때
    - 클래스 경계가 불명확할 때
    - 시간적 여유가 있을 때
    
    사용 예시:
    ----------
    >>> X_combined, y_combined = combined_sampling(X_train, y_train)
    >>> print(f"Before: {len(y_train)}, After: {len(y_combined)}")
    >>> print(y_combined.value_counts())
    """    
    try:
        smt = SMOTETomek(
            random_state=random_state,
            smote=SMOTE(sampling_strategy=sampling_strategy, random_state=random_state),
            tomek=TomekLinks(sampling_strategy='all')
        )
        X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
        
        # 샘플링 결과 출력
        print(f"✅ SMOTETomek 혼합 샘플링 완료")
        print(f"   원본 샘플 수: {len(y_train)} (Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)})")
        print(f"   샘플링 후: {len(y_resampled)} (Class 0: {sum(y_resampled==0)}, Class 1: {sum(y_resampled==1)})")
        print(f"   최종 변화: {len(y_resampled) - len(y_train):+d}개")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"❌ 혼합 샘플링 중 오류 발생: {e}")
        return X_train, y_train
# eof ----------------------------------------------------------- #


# 보너스: 샘플링 방법 비교 함수
def compare_sampling_methods(X_train, y_train, random_state=23):
    """
    세 가지 샘플링 방법의 결과 비교
    
    Parameters:
    -----------
    X_train : DataFrame or array-like
        학습 데이터의 특성
    y_train : Series or array-like
        학습 데이터의 타겟
    random_state : int, default=23
        재현성을 위한 랜덤 시드
    
    Returns:
    --------
    dict : 각 방법별 X, y 쌍을 담은 딕셔너리
    
    사용 예시:
    ----------
    >>> results = compare_sampling_methods(X_train, y_train)
    >>> X_smote, y_smote = results['smote']
    >>> X_under, y_under = results['under']
    >>> X_combined, y_combined = results['combined']
    """
    
    
    print("="*60)
    print("샘플링 방법 비교")
    print("="*60)
    
    # 원본
    print(f"\n[원본 데이터]")
    print(f"샘플 수: {len(y_train)}")
    print(f"클래스 분포:\n{pd.Series(y_train).value_counts()}")
    
    # SMOTE
    print(f"\n{'='*60}")
    X_smote, y_smote = oversampling_smote(X_train, y_train, random_state)
    
    # 언더샘플링
    print(f"\n{'='*60}")
    X_under, y_under = undersampling_RUS(X_train, y_train, random_state)
    
    # 혼합
    print(f"\n{'='*60}")
    X_combined, y_combined = combined_sampling(X_train, y_train, random_state)
    
    print(f"\n{'='*60}")
    print("비교 완료!")
    print("="*60)
    
    return {
        'original': (X_train, y_train),
        'smote': (X_smote, y_smote),
        'under': (X_under, y_under),
        'combined': (X_combined, y_combined)
    }
# eof ----------------------------------------------------------- #