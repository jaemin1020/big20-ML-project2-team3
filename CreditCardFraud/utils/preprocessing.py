# preprocessing.py 데이터 전처리 함수 
import os
import pandas as pd
import numpy  as np 
import datetime
import matplotlib.pyplot as plt
import seaborn           as sns

from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import RobustScaler
from sklearn.model_selection import train_test_split


# sof : Start of function ------------------------------------------------------ #
def ccf_load_data(train_path="../data/creditcard.csv"):
    try:
        # 파일 존재 확인 
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {train_path}")
        
        train = pd.read_csv(train_path)
        print(f"데이터 로드 성공: {train.shape}")
        return train
    
    except FileNotFoundError as e:
        print(f"❌ 파일 오류: {e}")
        return None
    
    except pd.errors.EmptyDataError:
        print(f"❌ 빈 파일입니다: {train_path}")
        return None
    
    except pd.errors.ParserError as e:
        print(f"❌ CSV 파싱 오류: {e}")
        return None
    
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return None
# eof : End of Function --------------------------------------------------------- #

# soF --------------------------------------------------------------------------
def checkBasicInfo(df, target_cols='Class', isInfo=True, isNullShow=True, isGraph=True):
    """
    데이터프레임의 기본 정보와 타겟 분포 확인
    
    Args:
        df: 데이터프레임
        target_cols: 타겟 컬럼명 (str 또는 list)
        isInfo: .info() 출력 여부
        isNullShow: 결측치 정보 출력 여부
        
    사용예시 : 
    # 기본 사용
    checkBasicInfo(df)

    # 선택적 사용
    checkBasicInfo(df, isInfo=False)  # info 제외
    checkBasicInfo(df, isNullShow=False)  # 결측치 확인 제외
    checkBasicInfo(df, target_cols=None)  # 타겟 분포 제외

    # 다른 타겟 컬럼
    checkBasicInfo(df, target_cols='Fraud')

    # 여러 타겟 컬럼
    checkBasicInfo(df, target_cols=['Class', 'Type', 'Category'])

    # 모든 기능 끄기
    checkBasicInfo(df, isInfo=False, isNullShow=False, target_cols=None)        
    """
    try:
        # DataFrame validation
        if df is None:
            print("❌ 오류: df가 None입니다.")
            return
        
        if not isinstance(df, pd.DataFrame):
            print(f"❌ 오류: df는 pandas DataFrame이어야 합니다. 현재 타입: {type(df).__name__}")
            return
        
        if df.empty:
            print("⚠️ 경고: 빈 데이터프레임입니다.")
            return
        
        if len(df.columns) == 0:
            print("⚠️ 경고: 컬럼이 없는 데이터프레임입니다.")
            return
        
        # 1. 기본 정보
        if isInfo:
            print(f"== 기본 정보 {'='*70}")
            df.info()            
            print(f"\nShape: {df.shape}")
        
        # 2. 결측치
        if isNullShow:
            print(f" 결측치 정보 {'='*70}")
            null_sum = df.isnull().sum()
            if null_sum.sum() > 0:
                print(null_sum[null_sum > 0])
            else:
                print("✅ 결측치 없음")
        
        # 3. 타겟 분포
        if target_cols:
            print(f"\n== 타겟 분포 {'='*70}")
            
            cols = [target_cols] if isinstance(target_cols, str) else target_cols
            
            for col in cols:
                if col not in df.columns:
                    print(f"⚠️ '{col}' 컬럼이 없습니다.")
                    continue
                
                print(f"\n[{col}] 개수:")
                print(df[col].value_counts().sort_index())
                
                print(f"\n[{col}] 비율(%):")
                proportions = df[col].value_counts(normalize=True).sort_index() * 100
                for idx, val in proportions.items():
                    print(f"  Class {idx}: {val:.2f}%")
                
                # 불균형 체크
                if len(df[col].value_counts()) == 2:
                    counts = df[col].value_counts()
                    ratio = counts.min() / counts.max() * 100
                    if ratio < 10:
                        print(f"  ⚠️ 심각한 불균형 ({ratio:.2f}%)")
        

        # 4. 시각화
        if isGraph:  
            plt.figure(figsize=(8, 4))
            df[target_cols].value_counts().plot(kind='bar')
            ax = df[target_cols].value_counts().plot(kind='bar')
            plt.title(f'Label({target_cols}) Distribution')

            # 막대 위에 값 표시
            for container in ax.containers:
                ax.bar_label(container, fmt='%d')
            plt.show()

        print(f"\n{'='*70}\n")        
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
# eof ----------------------------------------------------------- #



def remove_zero_columns(df, threshold_rate=0.99, save_report=True, isPrint=True):
    """
    0값이 threshold_rate 이상인 컬럼들을 제거

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    threshold_rate : float, default=0.99
        0값 비율 임계값 (0~1 사이, 예: 0.99 = 99%)
    save_report : bool, default=True
        제거된 컬럼 정보를 텍스트 파일로 저장할지 여부
    isPrint : bool, default=True
        제거된 컬럼 정보를 화면에 출력할지 여부     

    Returns:
    --------
    pd.DataFrame : 컬럼이 제거된 데이터프레임

    Example:
    --------
    >>> clean_df = remove_zero_columns(df, threshold_rate=0.95)
    >>> print(f"Original: {df.shape}, Cleaned: {clean_df.shape}")
    """

    # 입력 검증
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df는 pandas DataFrame이어야 합니다.")

    if not 0 <= threshold_rate <= 1:
        raise ValueError("threshold_rate는 0과 1 사이의 값이어야 합니다.")

    row_count = df.shape[0]
    if isPrint:
        print(f'\n{"="*102}')
        print(f"Current Data Status :  (Threshold: {threshold_rate*100}% )")
        print("=" * 102)
        print(f"Total rows: {row_count:,}")
        print(f"Total columns: {len(df.columns):,}")
        print("=" * 102)

    ## Summary DataFrame 생성
    summary_data = []

    for col in df.columns:
        zero_count = (df[col] == 0).sum()
        zero_rate = zero_count / len(df)
        mode_freq = df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
        mode_rate = mode_freq / len(df)

        summary_data.append(
            {
                "ColumnName": col,
                "na_Sum": df[col].isna().sum(),
                "nUnique": df[col].nunique(),
                "mode": df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                "modeFreq": mode_freq,
                "modeFreqRate": f"{mode_rate * 100:.2f}%",
                "zero_count": zero_count,
                "zero_count_rate": zero_rate,  # ✅ 숫자 (비교용)
                "zero_count_rate_display": f"{zero_rate * 100:.2f}%",  # ✅ 문자 (출력용)
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Summary 정보 출력 (표시용 컬럼 사용)
    display_cols = [
        "ColumnName",
        "na_Sum",
        "nUnique",
        "mode",
        "modeFreq",
        "modeFreqRate",
        "zero_count",
        "zero_count_rate_display",
    ]
    if isPrint:
        print(f'\n{"Summary 정보 (zero_count 내림차순)":^102}')
        print("=" * 102)
        print(
            summary_df[display_cols]
            .sort_values(by="zero_count", ascending=False)
            .to_string(index=False)
        )

    # ✅ 간단하게: 이미 만들어진 숫자형으로 바로 비교
    remove_cols = summary_df[summary_df["zero_count_rate"] > threshold_rate][
        "ColumnName"
    ].tolist()

    # 결과 저장
    if save_report and len(remove_cols) > 0:
        os.makedirs("../doc", exist_ok=True)
        file_path = f"../doc/remove_cols_{threshold_rate}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(remove_cols))

        print(f"\n제거 대상 컬럼 목록 ({len(remove_cols)}개):")
        print(remove_cols)
        print(f"\n✓ {len(remove_cols)}개의 컬럼명을 {file_path}에 저장하였습니다.")

    # 컬럼 제거
    if len(remove_cols) > 0:
        clean_df = df.drop(remove_cols, axis=1)

        print(f'\n{"="*102}')
        print(f"결과 요약")
        print("=" * 102)
        print(f"Threshold: {threshold_rate*100}%")
        print(f"제거된 컬럼 수: {len(remove_cols):,}")
        print(f"남은 컬럼 수: {clean_df.shape[1]:,}")
        print(f"원본 shape: {df.shape}")
        print(f"정제 후 shape: {clean_df.shape}")
        print(f"제거 비율: {len(remove_cols)/len(df.columns)*100:.2f}%")
        print("=" * 102)
    else:
        print(f"\n✓ Threshold({threshold_rate*100}%)를 초과하는 컬럼이 없습니다.")
        clean_df = df.copy()

    return clean_df
# EOF ---------------------------------------------------------------------------------------

def remove_zero_columns2(train_df, test_df, threshold_rate=0.99, save_report=True, isPrint=True):
    """
    0값이 threshold_rate 이상인 컬럼들을 train_df 기준으로 train_df와 test_df에서 모두 제거

    Parameters:
    -----------
    train_df : pd.DataFrame
        기준이 되는 학습 데이터프레임
    test_df : pd.DataFrame
        테스트 데이터프레임
    threshold_rate : float, default=0.99
        0값 비율 임계값 (0~1 사이, 예: 0.99 = 99%)
    save_report : bool, default=True
        제거된 컬럼 정보를 텍스트 파일로 저장할지 여부
    isPrint : bool, default=True
        제거된 컬럼 정보를 화면에 출력할지 여부            
        

    Returns:
    --------
    tuple : (clean_train_df, clean_test_df) - 컬럼이 제거된 데이터프레임들

    Example:
    --------
    >>> clean_train, clean_test = remove_zero_columns2(train_df, test_df, threshold_rate=0.95)
    >>> print(f"Train: {train_df.shape} -> {clean_train.shape}")
    >>> print(f"Test: {test_df.shape} -> {clean_test.shape}")
    """

    # 입력 검증
    if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
        raise ValueError("train_df와 test_df는 모두 pandas DataFrame이어야 합니다.")

    if not 0 <= threshold_rate <= 1:
        raise ValueError("threshold_rate는 0과 1 사이의 값이어야 합니다.")

    row_count = train_df.shape[0]

    if isPrint:
        print(f'\n{"="*102}')
        print(f"Train Data Analysis (Threshold: {threshold_rate*100}% )")
        print("=" * 102)
        print(f"Train rows: {train_df.shape[0]:,}, columns: {train_df.shape[1]:,}")
        print(f"Test rows: {test_df.shape[0]:,}, columns: {test_df.shape[1]:,}")
        print("=" * 102)

    ## Train 데이터 기준으로 Summary DataFrame 생성
    summary_data = []

    for col in train_df.columns:
        zero_count = (train_df[col] == 0).sum()
        zero_rate = zero_count / len(train_df)
        mode_freq = (
            train_df[col].value_counts().iloc[0] if len(train_df[col]) > 0 else 0
        )
        mode_rate = mode_freq / len(train_df)

        summary_data.append(
            {
                "ColumnName": col,
                "na_Sum": train_df[col].isna().sum(),
                "nUnique": train_df[col].nunique(),
                "mode": (
                    train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else None
                ),
                "modeFreq": mode_freq,
                "modeFreqRate": f"{mode_rate * 100:.2f}%",
                "zero_count": zero_count,
                "zero_count_rate": zero_rate,  # ✅ 숫자 (비교용)
                "zero_count_rate_display": f"{zero_rate * 100:.2f}%",  # ✅ 문자 (출력용)
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Summary 정보 출력 (표시용 컬럼 사용)
    display_cols = [
        "ColumnName",
        "na_Sum",
        "nUnique",
        "mode",
        "modeFreq",
        "modeFreqRate",
        "zero_count",
        "zero_count_rate_display",
    ]
    
    if isPrint:
        print(f'\n{"Train Summary (zero_count 내림차순)":^102}')
        print("=" * 102)
        print(
            summary_df[display_cols]
            .sort_values(by="zero_count", ascending=False)
            .to_string(index=False)
        )

    # ✅ 간단하게: 이미 만들어진 숫자형으로 바로 비교
    remove_cols = summary_df[summary_df["zero_count_rate"] > threshold_rate][
        "ColumnName"
    ].tolist()

    # 결과 저장
    if save_report and len(remove_cols) > 0:
        os.makedirs("../doc", exist_ok=True)
        file_path = f"../doc/remove_cols_train_{threshold_rate}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(remove_cols))

        print(f"\n제거 대상 컬럼 목록 ({len(remove_cols)}개):")
        print(remove_cols)
        print(f"\n✓ {len(remove_cols)}개의 컬럼명을 {file_path}에 저장하였습니다.")

    # Train과 Test에서 동일한 컬럼 제거
    if len(remove_cols) > 0:
        clean_train_df = train_df.drop(remove_cols, axis=1)
        clean_test_df = test_df.drop(remove_cols, axis=1)

        print(f'\n{"="*102}')
        print(f"결과 요약")
        print("=" * 102)
        print(f"Threshold: {threshold_rate*100}%")
        print(f"제거된 컬럼 수: {len(remove_cols):,}")
        print(f"\nTrain 데이터:")
        print(f"  원본 shape: {train_df.shape}")
        print(f"  정제 후 shape: {clean_train_df.shape}")
        print(f"\nTest 데이터:")
        print(f"  원본 shape: {test_df.shape}")
        print(f"  정제 후 shape: {clean_test_df.shape}")
        print(f"\n제거 비율: {len(remove_cols)/len(train_df.columns)*100:.2f}%")
        print("=" * 102)
    else:
        print(f"\n✓ Threshold({threshold_rate*100}%)를 초과하는 컬럼이 없습니다.")
        clean_train_df = train_df.copy()
        clean_test_df = test_df.copy()

    return clean_train_df, clean_test_df
# -- EOF ----------------------------------------------------------------------------------------------


def split_features_target(train_df, cols=None, target_col='Class'):
    """
    train_df를 X, y로 분리
    target_col이 None이면 y는 None 반환
    
    # 사용방법
    # 케이스 1: cols 없음 (타겟만 제거)
    X, y = split_features_target(train_df)

    # 케이스 2: 단일 컬럼
    X, y = split_features_target(train_df, cols='ID')

    # 케이스 3: 여러 컬럼 (리스트)
    X, y = split_features_target(train_df, cols=['ID', 'Time'])

    # 케이스 4: 타겟 컬럼명 변경
    X, y = split_features_target(train_df, cols='ID', target_col='Fraud')    
    """
    drop_cols = []
    
    # cols 처리
    if cols is not None:
        if isinstance(cols, str):
            drop_cols.append(cols)
        else:  # list
            drop_cols.extend(cols)
    
    # target_col 처리
    if target_col is not None:
        drop_cols.append(target_col)
    
    # X 생성
    if drop_cols:
        X = train_df.drop(drop_cols, axis=1)
    else:
        X = train_df.copy()
    
    # y 생성
    y = train_df[target_col] if target_col is not None else None
    
    return X, y
# eof ----------------------------------------------------------- #


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
# eof ----------------------------------------------------------- #

def scale_selected_columns(X_train, X_test=None, columns=None):
    """
    특정 컬럼만 StandardScaler로 스케일링하는 함수
    
    Parameters
    ----------
    X_train : pd.DataFrame
        학습용 데이터
    X_test : pd.DataFrame or None, optional
        테스트용 데이터 (없으면 None)
    columns : list
        스케일링할 컬럼명 리스트
    
    Returns
    -------
    X_train_scaled : pd.DataFrame
        스케일링된 학습 데이터
    X_test_scaled : pd.DataFrame or None
        스케일링된 테스트 데이터 (입력 None이면 None 반환)
    scaler : StandardScaler
        학습된 스케일러 객체
    """
    # ✅ 유효성 검사
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train은 반드시 pandas DataFrame이어야 합니다.")
    
    if X_test is not None and not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test는 None 또는 pandas DataFrame이어야 합니다.")
    
    if not isinstance(columns, (list, tuple)):
        raise TypeError("columns는 리스트나 튜플이어야 합니다.")
    
    missing_cols_train = [col for col in columns if col not in X_train.columns]
    if missing_cols_train:
        raise ValueError(f"X_train에 없는 컬럼: {missing_cols_train}")
    
    if X_test is not None:
        missing_cols_test = [col for col in columns if col not in X_test.columns]
        if missing_cols_test:
            raise ValueError(f"X_test에 없는 컬럼: {missing_cols_test}")
    
    # ✅ 스케일링
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[columns] = scaler.fit_transform(X_train[columns])
    
    if X_test is not None:
        X_test_scaled = X_test.copy()
        X_test_scaled[columns] = scaler.transform(X_test[columns])
    else:
        X_test_scaled = None
    
    return X_train_scaled, X_test_scaled, scaler

# eof ----------------------------------------------------------- #

def undersample(train_df, target_col="TARGET", n_majority=20000, random_state=42):
    majority = train_df[train_df[target_col] == 0].sample(
        n=n_majority, random_state=random_state
    )
    minority = train_df[train_df[target_col] == 1]
    balanced = pd.concat([majority, minority])
    return balanced
# eof ----------------------------------------------------------- #

# learn/test data sperate


def data_split(X_features, y_target, size=0.2, rs=23):
    """
    X_features와 y_target을 넣으면 기본 8:2로 분할하는 함수
    사용 예제:
        X_train, X_test, y_train, y_test = data_split(X_features, y_target)
    """
    X_train = X_test = y_train = y_test = None

    if (
        X_features is not None
        and y_target is not None
        and len(X_features) > 0
        and len(y_target) > 0
    ):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_target, test_size=size, random_state=rs, stratify=y_target
            )
        except ValueError as e:
            print(f"데이터 분할 중 오류 발생: {e}")
    else:
        print("데이터를 입력하세요")

    return X_train, X_test, y_train, y_test
# eof ----------------------------------------------------------- #

def robustScaler(df, cols=['Amount', 'Time'], isDropCols=True):
    """
    RobustScaler를 사용하여 지정된 컬럼들을 스케일링
    
    RobustScaler는 중앙값(median)과 IQR(Interquartile Range)을 사용하여
    이상치(outlier)에 강건한 스케일링을 수행합니다.
    
    Parameters:
    -----------
    df : DataFrame
        스케일링할 데이터프레임
    cols : list, default=['Amount', 'Time']
        스케일링할 컬럼 리스트
    isDropCols : bool, default=True
        원본 컬럼 제거 여부
        - True: 원본 컬럼 제거, '_scaled' 컬럼만 유지
        - False: 원본 컬럼 유지, '_scaled' 컬럼 추가
    
    Returns:
    --------
    DataFrame
        스케일링이 완료된 데이터프레임 복사본
    
    주의사항:
    ---------
    - 원본 df는 변경되지 않고, 복사본을 반환
    - 각 컬럼마다 별도의 scaler를 fit (독립적 스케일링)
    - 수치형 컬럼에만 사용 가능
    
    사용 예시:
    ----------
    >>> # 기본 사용 (원본 컬럼 제거)
    >>> df_scaled = robustScaler(df)
    
    >>> # 원본 컬럼 유지
    >>> df_scaled = robustScaler(df, isDropCols=False)
    
    >>> # 특정 컬럼만 스케일링
    >>> df_scaled = robustScaler(df, cols=['Amount'])
    
    >>> # 여러 컬럼 스케일링
    >>> df_scaled = robustScaler(df, cols=['Amount', 'Time', 'V1', 'V2'])
    """
    try:
        # 1. DataFrame validation
        if df is None:
            raise ValueError("df가 None입니다.")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df는 pandas DataFrame이어야 합니다. 현재 타입: {type(df).__name__}")
        
        if df.empty:
            raise ValueError("빈 데이터프레임입니다.")
        
        # 2. cols validation
        if not cols:
            raise ValueError("스케일링할 컬럼이 지정되지 않았습니다.")
        
        if not isinstance(cols, list):
            raise TypeError(f"cols는 list 타입이어야 합니다. 현재 타입: {type(cols).__name__}")
        
        # 3. 원본 보호를 위한 복사본 생성
        df_scaled = df.copy()
        
        # 4. 존재하지 않는 컬럼 확인
        missing_cols = [col for col in cols if col not in df_scaled.columns]
        if missing_cols:
            raise KeyError(f"다음 컬럼이 데이터프레임에 없습니다: {missing_cols}")
        
        # 5. 각 컬럼 스케일링
        scaled_cols = []
        for col in cols:
            try:
                # 수치형 확인
                if not pd.api.types.is_numeric_dtype(df_scaled[col]):
                    print(f"⚠️ 경고: '{col}'은 수치형이 아닙니다. 스킵합니다. (dtype: {df_scaled[col].dtype})")
                    continue
                
                # 결측치 확인
                if df_scaled[col].isnull().any():
                    print(f"⚠️ 경고: '{col}'에 결측치가 있습니다. (개수: {df_scaled[col].isnull().sum()})")
                    # 결측치가 있어도 스케일링은 진행 (RobustScaler가 처리)
                
                # 스케일링 수행 (각 컬럼마다 별도 scaler 사용)
                scaler = RobustScaler()
                df_scaled[f'{col}_scaled'] = scaler.fit_transform(
                    df_scaled[col].values.reshape(-1, 1)
                )
                scaled_cols.append(col)
                
                print(f"✅ '{col}' 스케일링 완료 → '{col}_scaled'")
                
            except Exception as e:
                print(f"❌ '{col}' 스케일링 중 오류: {e}")
                continue
        
        # 6. 스케일링 성공 확인
        if not scaled_cols:
            raise ValueError("스케일링된 컬럼이 하나도 없습니다.")
        
        # 7. 원본 컬럼 제거
        if isDropCols:
            df_scaled = df_scaled.drop(scaled_cols, axis=1)
            print(f"\n🗑️  원본 컬럼 제거: {scaled_cols}")
        
        print(f"\n✅ 스케일링 완료: {len(scaled_cols)}개 컬럼")
        print(f"   최종 shape: {df_scaled.shape}")
        
        return df_scaled
    
    except Exception as e:
        print(f"❌ robustScaler 함수 실행 중 오류 발생: {e}")
        raise

# eof ----------------------------------------------------------- #

def corrGraph(df, compCol='Time', title='Correlation Matrix of Features', threshold=0.7):
    """
    데이터프레임의 상관관계를 분석하고 시각화하는 함수.

    Parameters
    ----------
    df : pandas.DataFrame
        분석할 데이터프레임.
    compCol : str, optional (default='Time')
        특정 변수와의 상관관계를 별도로 출력할 기준 컬럼명.
    title : str, optional (default='Correlation Matrix of Features')
        상관관계 히트맵의 제목.
    threshold : float, optional (default=0.7)
        높은 상관관계로 판단할 절댓값 기준.

    Functionality
    -------------
    1. 전체 상관관계 행렬을 계산하고 히트맵으로 시각화.
       - 결과 이미지를 '../images/CorrMatrixAllFeataures_yyyy_mmdd.png'로 저장.
    2. 상관계수의 평균, 최대, 최소, 표준편차를 출력.
    3. |correlation| > threshold 인 변수 쌍을 찾아 출력.
    4. compCol과 다른 변수들의 상관관계를 상위 10개까지 출력하고,
       '../results/CorrWith{compCol}_yyyy_mmdd.txt' 파일로 저장.
    5. 전체 상관계수 분포를 히스토그램으로 시각화.
       - 결과 이미지를 '../images/Distr_CorreCoeff_yyyy_mmdd.png'로 저장.

    Returns
    -------
    None
        화면 출력과 파일 저장, 그래프 시각화만 수행하며 값을 반환하지 않음.
    """
    # 2. 상관관계 행렬 계산
    correlation_matrix = df.corr()

    # 3. 상관관계 히트맵 시각화
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix,
                annot=False,          # 셀 안에 숫자 표시 여부
                cmap='coolwarm',      # 색상 테마
                center=0,             # 0을 기준으로 색상 분할
                vmin=-1, vmax=1,      # 상관계수 범위
                square=True,          # 셀을 정사각형으로 표시
                linewidths=0.5,       # 셀 구분선 두께
                cbar_kws={"shrink": 0.8})  # 컬러바 크기 조정
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    # 히트맵을 이미지 파일로 저장 (날짜 포함)
    plt.savefig(f'../images/CorrMatrixAllFeataures_{datetime.datetime.today().strftime("%Y_%m%d")}.png')
    plt.show()

    # 4. 상관관계 통계 요약 출력
    print("=" * 60)
    print("상관관계 통계 요약")
    print("=" * 60)

    # 대각선 제외한 상관계수 값 추출
    corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
    print(f"평균 상관계수: {corr_values.mean():.4f}")
    print(f"최대 상관계수: {corr_values.max():.4f}")
    print(f"최소 상관계수: {corr_values.min():.4f}")
    print(f"표준편차: {corr_values.std():.4f}")

    # 5. 높은 상관관계 쌍 찾기
    print("\n" + "=" * 60)
    print(f"높은 상관관계 변수 쌍 (|correlation| > {threshold})")
    print("=" * 60)

    # 상위 삼각행렬만 추출 (중복 제거)
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # 높은 상관관계 쌍을 리스트에 저장
    high_corr = []
    for column in upper_tri.columns:
        for index in upper_tri.index:
            corr_val = upper_tri.loc[index, column]
            if pd.notna(corr_val) and abs(corr_val) > threshold:
                high_corr.append({
                    'Variable 1': index,
                    'Variable 2': column,
                    'Correlation': corr_val
                })

    # DataFrame으로 변환 후 출력
    high_corr_df = pd.DataFrame(high_corr)
    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values('Correlation', key=lambda x: x.abs(), ascending=False)
        print(high_corr_df.to_string(index=False))
        print(f"\n총 {len(high_corr_df)}개의 높은 상관관계 쌍 발견")
    else:
        print("높은 상관관계 쌍이 발견되지 않았습니다.")

    # 6. 특정 변수(compCol)와의 상관관계 출력 및 저장
    if compCol in df.columns:
        header = "\n" + "=" * 60 + "\n"
        header += f"'{compCol}'와 상관관계가 높은 변수들 (상위 10개)\n"
        header += "=" * 60 + "\n"

        # compCol과 다른 변수들의 상관관계 절댓값 기준 정렬
        var_corr = correlation_matrix[compCol].abs().sort_values(ascending=False)
        result_text = var_corr.head(11)[1:].to_string()  # 자기 자신 제외

        # 화면 출력
        print(header)
        print(result_text)

        # 날짜 yyyy_mmdd 형식
        today = datetime.datetime.today().strftime("%Y_%m%d")

        # 결과 저장할 폴더 및 파일명 지정
        save_dir = "../results"
        os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성
        filename = f"{save_dir}/CorrWith{compCol}_{today}.txt"

        # 결과를 txt 파일로 저장
        with open(filename, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(result_text)

        print(f"\n결과가 파일로 저장되었습니다: {filename}")

    # 7. 상관관계 분포 히스토그램 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(corr_values, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Correlation Coefficients', fontsize=14)
    # 기준선 표시
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=1, label=f'High positive ({threshold})')
    plt.axvline(x=-threshold, color='green', linestyle='--', linewidth=1, label=f'High negative (-{threshold})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # 히스토그램을 이미지 파일로 저장 (날짜 포함)
    plt.savefig(f'../images/Distr_CorreCoeff_{today}.png')
    plt.show()
# eof ----------------------------------------------------------------------------------------------------------    


# Outlier가 어디에 있는지 추출하는 함수 - 신용카드사기용
def get_outlier_index(df, column, weight=1.5, class_filter=None):
    """
    특정 컬럼에서 IQR(Interquartile Range) 방식을 사용하여 이상치(outlier)의 인덱스를 반환하는 함수.
    기본적으로 전체 데이터에서 이상치를 탐지하며, 필요 시 특정 Class 값으로 필터링할 수 있습니다.

    Parameters
    ----------
    df : pandas.DataFrame
        분석할 데이터프레임. 'Class' 컬럼을 포함해야 합니다.
    column : str
        이상치를 탐지할 대상 컬럼명 (예: 'V14').
    weight : float, optional (default=1.5)
        IQR 범위를 확장하는 가중치. 일반적으로 1.5를 사용하며,
        값이 클수록 이상치 판정 범위가 넓어집니다.
    class_filter : int or None, optional (default=None)
        특정 Class 값으로 필터링할 때 사용.
        - None : 전체 데이터에서 이상치 탐지
        - 0    : 정상 거래(Class==0)만 대상으로 탐지
        - 1    : 사기 거래(Class==1)만 대상으로 탐지

    Returns
    -------
    pandas.Index
        이상치로 판정된 행(row)의 인덱스 집합.

    Notes
    -----
    - IQR 방식: Q1(25%)와 Q3(75%)를 기준으로 IQR = Q3 - Q1을 계산.
      이상치 기준은 [Q1 - weight*IQR, Q3 + weight*IQR] 범위를 벗어난 값.
    - class_filter를 지정하지 않으면 전체 데이터에서 이상치를 탐지합니다.

    Examples
    --------
    >>> # 사기 거래(Class==1)에서 V14 이상치 탐지
    >>> fraud_outliers = get_outlier_index(df, column='V14', weight=1.5, class_filter=1)

    >>> # 정상 거래(Class==0)에서 V14 이상치 탐지
    >>> normal_outliers = get_outlier_index(df, column='V14', weight=1.5, class_filter=0)

    >>> # 전체 데이터에서 V14 이상치 탐지
    >>> all_outliers = get_outlier_index(df, column='V14', weight=1.5)
    """
    # class_filter 적용
    if class_filter is not None:
        data = df[df['Class'] == class_filter][column]
    else:
        data = df[column]

    # 사분위수 계산
    q_25 = np.percentile(data.values, 25)
    q_75 = np.percentile(data.values, 75)
    iqr = q_75 - q_25
    iqr_weight = iqr * weight

    # 이상치 기준
    lowest_val = q_25 - iqr_weight
    highest_val = q_75 + iqr_weight

    # 이상치 인덱스 반환
    outlier_index = data[(data < lowest_val) | (data > highest_val)].index
    return outlier_index
# eof ----------------------------------------------------------------------------------------------------------    