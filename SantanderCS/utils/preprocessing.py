import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


# sof ----------------------------------------------------------- #
def load_data(train_path="../data/train.csv", test_path="../data/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


# eof ----------------------------------------------------------- #


def remove_zero_columns(df, threshold_rate=0.99, save_report=True):
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


def remove_zero_columns2(train_df, test_df, threshold_rate=0.99, save_report=True):
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


def split_features_target(train_df, target_col="TARGET"):
    """
    train_df 를 넣으면 'ID' 와 'TARGET'를 기본으로 삭제해서 X 를,
    'TARGET'를 기본값으로 y를 return 함
    사용법 : X_Features, y_labels = split_feature_target(train)
    """
    X = train_df.drop(["ID", target_col], axis=1)
    y = train_df[target_col]
    return X, y


# eof ----------------------------------------------------------- #


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

from sklearn.model_selection import train_test_split


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
