import os
import pandas as pd
from sklearn.preprocessing   import StandardScaler

# sof ----------------------------------------------------------- #
def load_data(train_path= '../data/train.csv', test_path= '../data/test.csv'):
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
            0값 비율 임계값 (0~1 사이)
        save_report : bool, default=True
            제거된 컬럼 정보를 CSV로 저장할지 여부
        
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
    threshold = row_count * threshold_rate
    
    print(f'\n{"="*102}')
    print(f"Zero Value Analysis (Threshold: {threshold_rate*100}% = {int(threshold):,} rows)")
    print("="*102)
    print(f"Total rows: {row_count:,}")
    print(f"Total columns: {len(df.columns):,}")
    print("="*102)
    
    ## Summary DataFrame 생성 
    summary_data = []
    
    for col in df.columns:
        zero_count = (df[col] == 0).sum()
        mode_freq = df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
        
        summary_data.append({
            'ColumnName': col,
            'na_Sum': df[col].isna().sum(),
            'nUnique': df[col].nunique(),
            'mode': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
            'modeFreq': mode_freq,
            'modeFreqRate': f"{(mode_freq / len(df) * 100):.2f}%",
            'zero_count': zero_count,
            'zero_count_rate': f"{(zero_count / len(df) * 100):.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Summary 정보 출력
    print(f'\n{"Summary 정보 (zero_count 내림차순)":^102}')
    print("="*102)
    print(summary_df.sort_values(by='zero_count', ascending=False).to_string(index=False))
    
    # Threshold 초과 컬럼 찾기
    remove_cols_df = summary_df[summary_df["zero_count"] > threshold].copy()
    remove_cols = remove_cols_df["ColumnName"].tolist()
    
    # 결과 저장
    if save_report and len(remove_cols) > 0:
        # doc 폴더 생성
        os.makedirs('../doc', exist_ok=True)
        
        # 파일 저장
        file_path = f'../doc/remove_cols_{threshold_rate}.csv'
        remove_cols_df.to_csv(file_path, index=False, encoding='utf-8')
        
        print(f'\n{"="*102}')
        print(f"제거 대상 컬럼 상세 정보")
        print("="*102)
        
        for i, col in enumerate(remove_cols, 1):
            col_info = remove_cols_df[remove_cols_df["ColumnName"] == col].iloc[0]
            print(f"{i:3d}. {col:20s} - zero_count: {col_info['zero_count']:>10,} "
                  f"({col_info['zero_count_rate']:>7s})")
        
        print(f'\n✓ {len(remove_cols)}개의 컬럼 정보를 {file_path}에 저장하였습니다.')
    
    # 컬럼 제거
    if len(remove_cols) > 0:
        clean_df = df.drop(remove_cols, axis=1)
        
        print(f'\n{"="*102}')
        print(f"결과 요약")
        print("="*102)
        print(f"제거된 컬럼 수: {len(remove_cols):,}")
        print(f"남은 컬럼 수: {clean_df.shape[1]:,}")
        print(f"원본 shape: {df.shape}")
        print(f"정제 후 shape: {clean_df.shape}")
        print(f"제거 비율: {len(remove_cols)/len(df.columns)*100:.2f}%")
        print("="*102)
    else:
        print(f'\n✓ Threshold({threshold_rate*100}%)를 초과하는 컬럼이 없습니다.')
        clean_df = df.copy()
    
    return clean_df


def compare_zero_thresholds(df, thresholds=[0.95, 0.97, 0.99]):
    """
    여러 threshold 값에 따른 제거 컬럼 수 비교
    
    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    thresholds : list, default=[0.95, 0.97, 0.99]
        비교할 threshold 값들
    
    Example:
    --------
    >>> compare_zero_thresholds(df, [0.90, 0.95, 0.99])
    """
    print(f'\n{"="*60}')
    print(f"Threshold 비교 분석")
    print("="*60)
    
    results = []
    row_count = df.shape[0]
    
    for threshold_rate in thresholds:
        threshold = row_count * threshold_rate
        zero_counts = [(col, (df[col] == 0).sum()) for col in df.columns]
        remove_count = sum(1 for col, count in zero_counts if count > threshold)
        
        results.append({
            'threshold_rate': f"{threshold_rate*100:.0f}%",
            'threshold': int(threshold),
            'remove_cols': remove_count,
            'remain_cols': len(df.columns) - remove_count,
            'remove_rate': f"{remove_count/len(df.columns)*100:.2f}%"
        })
    
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    print("="*60)
    
    return result_df
# -- EOF ----------------------------------------------------------------------------------------------  


def split_features_target(train_df, target_col='TARGET'):
    '''
    train_df 를 넣으면 'ID' 와 'TARGET'를 기본으로 삭제해서 X 를, 
    'TARGET'를 기본값으로 y를 return 함
    사용법 : X_Features, y_labels = split_feature_target(train)
    '''
    X = train_df.drop(['ID', target_col], axis=1)
    y = train_df[target_col]
    return X, y
# eof ----------------------------------------------------------- #

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
# eof ----------------------------------------------------------- #

def undersample(train_df, target_col='TARGET', n_majority=20000, random_state=42):
    majority = train_df[train_df[target_col] == 0].sample(n=n_majority, random_state=random_state)
    minority = train_df[train_df[target_col] == 1]
    balanced = pd.concat([majority, minority])
    return balanced
# eof ----------------------------------------------------------- #

# learn/test data sperate

from sklearn.model_selection import train_test_split

def data_split(X_features, y_target, size=0.2, rs=23):
    '''
    X_features와 y_target을 넣으면 기본 8:2로 분할하는 함수
    사용 예제:
        X_train, X_test, y_train, y_test = data_split(X_features, y_target)
    '''
    X_train = X_test = y_train = y_test = None

    if X_features is not None and y_target is not None and len(X_features) > 0 and len(y_target) > 0:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_features,
                y_target,
                test_size=size,
                random_state=rs,
                stratify=y_target
            )
        except ValueError as e:
            print(f"데이터 분할 중 오류 발생: {e}")
    else:
        print('데이터를 입력하세요')

    return X_train, X_test, y_train, y_test

# eof ----------------------------------------------------------- #    