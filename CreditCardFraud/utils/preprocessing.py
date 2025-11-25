import os
import pandas as pd
from sklearn.preprocessing import StandardScaler



# sof ----------------------------------------------------------- #
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
