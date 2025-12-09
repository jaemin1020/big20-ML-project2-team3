# result_report.py - Mercari Price Prediction 회귀 결과 분석 12.09 
# ============================================================================
# results 폴더의 JSON 파일을 읽어 회귀 모델 성능 비교표 및 시각화 ㅜ생성
# ============================================================================

import os
import json
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import numpy as np


# set_korean_font start ###########################
def set_korean_font():
    """
    운영체제에 맞는 한글 폰트 설정
    
    Notes:
    ------
    - Windows: 맑은 고딕
    - macOS: AppleGothic
    - Linux: NanumGothic
    """
    system = platform.system()
    
    if system == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='NanumGothic')
    
    plt.rc('axes', unicode_minus=False)
    
    print(f"✅ 한글 폰트 설정 완료: {system}")
# set_korean_font end ======================================


# load_results_to_df start ###########################
def load_results_to_df(
    folder_path='../results', 
    sort_by='RMSLE',  # 회귀: RMSLE 기준
    ascending=True,   # 회귀: 낮을수록 좋음
    filter_pattern=None,
    exclude_patterns=None
):
    """
    results 폴더의 JSON 파일을 읽어서 DataFrame으로 변환 (회귀 버전)
    
    Parameters:
    -----------
    folder_path : str
        결과 파일들이 저장된 폴더 경로 (기본값: '../results')
    sort_by : str, optional
        정렬 기준 컬럼명 (기본값: 'RMSLE' - Kaggle 평가지표)
    ascending : bool, optional
        오름차순 정렬 여부 (기본값: True - 회귀는 낮을수록 좋음)
    filter_pattern : str or list, optional
        파일명에 포함되어야 할 패턴 
        (예: 'reg', ['lgb', 'xgb'])
    exclude_patterns : str or list, optional
        파일명에서 제외할 패턴 
        (예: ['cache', 'submission'])
    
    Returns:
    --------
    pd.DataFrame
        model_name과 회귀 메트릭(R2, RMSE, MAE, RMSLE)을 컬럼으로 하는 DataFrame
    
    Examples:
    ---------
    >>> # 전체 회귀 결과 로드
    >>> df = load_results_to_df('../results')
    
    >>> # LGBM 결과만 로드
    >>> df = load_results_to_df('../results', filter_pattern='lgb')
    
    >>> # 여러 패턴 중 하나라도 포함
    >>> df = load_results_to_df('../results', filter_pattern=['lgb', 'xgb', 'stacking'])
    
    Notes:
    ------
    - JSON 구조: {"R2": 0.99, "RMSE": 3.4, "MAE": 0.17, "RMSLE": 0.012, ...}
    - 파일명에서 타임스탬프 제거 (예: _20251209_143022)
    - RMSLE 기준 오름차순 정렬 (낮을수록 좋은 모델)
    """
    
    results_list = []
    error_files = []
    no_metrics = []
    filtered_out = []
    
    # 폴더 존재 확인
    if not os.path.exists(folder_path):
        print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        return pd.DataFrame()
    
    # JSON 파일 찾기
    files = list(Path(folder_path).glob('*.json'))
    
    # 기본 제외 패턴
    default_exclude = [
        'cache', 'submission', 'residual_samples', 
        'trials', 'best_params', 'metrics_summary'
    ]
    if exclude_patterns:
        if isinstance(exclude_patterns, str):
            exclude_patterns = [exclude_patterns]
        default_exclude.extend(exclude_patterns)
    
    # 필터 패턴 리스트 변환
    if filter_pattern and isinstance(filter_pattern, str):
        filter_pattern = [filter_pattern]
    
    if not files:
        print(f"⚠️  폴더에 JSON 파일이 없습니다: {folder_path}")
        return pd.DataFrame()
    
    print(f"📂 총 {len(files)}개 JSON 파일 발견")
    if filter_pattern:
        print(f"🔍 필터 패턴: {filter_pattern}")
    if default_exclude:
        print(f"🚫 제외 패턴: {default_exclude}")
    print()
    
    for file_path in files:
        # 제외 패턴 체크
        if any(pattern in file_path.name for pattern in default_exclude):
            continue
        
        # 필터 패턴 체크
        if filter_pattern:
            if not any(pattern in file_path.name for pattern in filter_pattern):
                filtered_out.append(file_path.name)
                continue
        
        try:
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 회귀 메트릭이 있는지 확인
            has_metrics = False
            
            # 패턴 1: 직접 메트릭 ({"R2": 0.99, "RMSE": 3.4, ...})
            if isinstance(data, dict) and any(key in data for key in ['R2', 'RMSE', 'MAE', 'RMSLE', 'rmse', 'rmsle']):
                metrics = data
                has_metrics = True
            
            # 패턴 2: metrics 키 안에 있는 경우
            elif isinstance(data, dict) and 'metrics' in data:
                metrics = data['metrics']
                has_metrics = True
            
            # 패턴 3: result_dict 안에 있는 경우 (이전 버전 호환)
            elif isinstance(data, dict) and 'result_dict' in data:
                metrics = data['result_dict']
                has_metrics = True
            
            if has_metrics:
                # 파일명에서 모델명 추출
                model_name = file_path.stem
                
                # 타임스탬프 패턴 제거
                model_name = re.sub(r'_\d{8}_\d{6}$', '', model_name)  # _20251209_143022
                model_name = re.sub(r'_\d{8}$', '', model_name)         # _20251209
                model_name = re.sub(r'_reg$', '', model_name)           # _reg 제거
                model_name = re.sub(r'__+', '_', model_name)            # 이중 언더스코어
                model_name = model_name.rstrip('_')
                
                # 행 데이터 생성
                row = {'model_name': model_name, 'file_name': file_path.name}
                
                # 회귀 메트릭 추가 (대소문자 모두 지원)
                metric_keys = {
                    'R2': ['R2', 'r2', 'r_squared'],
                    'RMSE': ['RMSE', 'rmse', 'root_mean_squared_error'],
                    'MAE': ['MAE', 'mae', 'mean_absolute_error'],
                    'RMSLE': ['RMSLE', 'rmsle', 'RMSLE_Kaggle', 'rmsle_kaggle']
                }
                
                for standard_key, possible_keys in metric_keys.items():
                    for key in possible_keys:
                        if key in metrics:
                            row[standard_key] = metrics[key]
                            break
                    # 키를 찾지 못한 경우 None
                    if standard_key not in row:
                        row[standard_key] = None
                
                # 추가 정보 (있는 경우)
                if 'model_name' in data and data['model_name'] != model_name:
                    row['original_model_name'] = data['model_name']
                
                if '학습 시간' in data:
                    row['학습시간'] = data['학습 시간']
                elif '학습시간' in data:
                    row['학습시간'] = data['학습시간']
                
                results_list.append(row)
                print(f"✅ {model_name[:70]}")
            else:
                no_metrics.append(file_path.name)
                
        except json.JSONDecodeError as e:
            error_files.append((file_path.name, f"JSON 파싱 오류: {e}"))
            print(f"❌ JSON 파싱 오류: {file_path.name}")
        except Exception as e:
            error_files.append((file_path.name, str(e)))
            print(f"❌ 파일 읽기 오류: {file_path.name} - {e}")
    
    # DataFrame 생성
    if results_list:
        df = pd.DataFrame(results_list)
        
        # model_name을 첫 번째 컬럼으로
        cols = ['model_name', 'file_name'] + [col for col in df.columns if col not in ['model_name', 'file_name']]
        df = df[cols]
        
        # 정렬 (RMSLE 기준, 낮을수록 좋음)
        if sort_by and sort_by in df.columns:
            # NaN 제거 후 정렬
            df_sorted = df.dropna(subset=[sort_by]).sort_values(sort_by, ascending=ascending)
            df_nan = df[df[sort_by].isna()]
            df = pd.concat([df_sorted, df_nan]).reset_index(drop=True)
        
        print(f"\n{'='*80}")
        print(f"✅ 총 {len(df)}개 모델 결과 로드 완료")
        print(f"📊 컬럼: {list(df.columns)}")
        
        if filtered_out:
            print(f"🔍 필터로 제외된 파일: {len(filtered_out)}개")
        if error_files:
            print(f"⚠️  오류 파일: {len(error_files)}개")
        if no_metrics:
            print(f"⚠️  메트릭 없는 파일: {len(no_metrics)}개")
        print(f"{'='*80}\n")
        
        df # 일부러 print 걸지 않음(양식 깨지니까)
        
        return df
    else:
        print("\n⚠️  유효한 결과가 없습니다.")
        if error_files:
            print("\n오류 파일 목록:")
            for fname, err in error_files[:10]:
                print(f"  - {fname}: {err}")
        return pd.DataFrame()
# load_results_to_df end ======================================


# plot_model_comparison start ###########################
def plot_model_comparison(df, metrics=None, top_n=10, figsize=(16, 10), save_path="../images/results_plot.png"):
    """
    회귀 모델 비교 그래프 생성
    
    Parameters:
    -----------
    df : pd.DataFrame
        load_results_to_df()로 생성된 DataFrame
    metrics : list, optional
        비교할 지표 리스트 (기본값: ['RMSLE', 'RMSE', 'MAE', 'R2'])
    top_n : int, optional
        표시할 상위 모델 개수 (기본값: 10)
    figsize : tuple, optional
        그래프 크기 (기본값: (16, 10))
    save_path : str, optional
        그래프 저장 경로
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    
    Notes:
    ------
    - RMSLE, RMSE, MAE: 낮을수록 좋음 (빨강→초록)
    - R2: 높을수록 좋음 (초록→빨강)
    - RMSLE 기준 상위 N개 선택
    """
    
    set_korean_font()
    
    if df.empty:
        print("⚠️  DataFrame이 비어있습니다.")
        return None, None
    
    # 기본 metrics 설정
    if metrics is None:
        metrics = ['RMSLE', 'RMSE', 'MAE', 'R2']
    
    # 사용 가능한 metrics만 필터링
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print(f"⚠️  지정한 metrics가 DataFrame에 없습니다. 사용 가능: {list(df.columns)}")
        return None, None
    
    # 상위 N개 모델 선택 (RMSLE 기준)
    if 'RMSLE' in df.columns:
        df_top = df.dropna(subset=['RMSLE']).head(top_n).copy()
    else:
        df_top = df.head(top_n).copy()
    
    # NaN 체크
    for metric in available_metrics:
        if df_top[metric].isna().all():
            print(f"⚠️  '{metric}' 컬럼이 모두 NaN입니다.")
            available_metrics.remove(metric)
    
    if not available_metrics:
        print("⚠️  유효한 메트릭이 없습니다.")
        return None, None
    
    # 그래프 설정
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")
    
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # 각 metric 그래프
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # 유효한 데이터만
        valid_data = df_top[df_top[metric].notna()].copy()
        
        if valid_data.empty:
            ax.text(0.5, 0.5, f'{metric}: 데이터 없음', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            continue
        
        # 수평 막대 그래프
        bars = ax.barh(valid_data['model_name'], valid_data[metric], alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # 색상 설정 (RMSE/MAE/RMSLE: 낮을수록 초록, R2: 높을수록 초록)
        metric_min = valid_data[metric].min()
        metric_max = valid_data[metric].max()
        
        if metric_min == metric_max:
            colors = ['#90EE90'] * len(valid_data)
        else:
            if metric in ['RMSE', 'MAE', 'RMSLE']:
                # 낮을수록 좋음 (역순)
                normalized = 1 - (valid_data[metric] - metric_min) / (metric_max - metric_min)
            else:  # R2
                # 높을수록 좋음 (정순)
                normalized = (valid_data[metric] - metric_min) / (metric_max - metric_min)
            
            colors = plt.cm.RdYlGn(normalized)
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        # 제목 (낮을수록/높을수록 표시)
        if metric in ['RMSE', 'MAE', 'RMSLE']:
            title_suffix = '(낮을수록 좋음)'
        else:
            title_suffix = '(높을수록 좋음)'
        ax.set_title(f'{metric} 비교 {title_suffix}', fontsize=14, fontweight='bold', pad=10)
        
        # 값 표시
        for val, bar in zip(valid_data[metric], bars):
            if metric == 'R2':
                label = f'{val:.4f}'
            elif metric == 'RMSLE':
                label = f'{val:.6f}'
            else:
                label = f'{val:.2f}'
            
            ax.text(val, bar.get_y() + bar.get_height()/2, f' {label}', 
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=10)
    
    # 빈 subplot 숨기기
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Mercari Price Prediction - 모델 성능 비교 (Top {len(df_top)})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 그래프 저장: {save_path}")
        plt.show()
    
    return fig, axes
# plot_model_comparison end ======================================


# plot_metric_heatmap start ###########################
def plot_metric_heatmap(df, metrics=None, top_n=10, figsize=(10, 8), save_path="../images/result_hetmap.png"):
    """
    회귀 모델 메트릭 히트맵 생성
    
    Parameters:
    -----------
    df : pd.DataFrame
        load_results_to_df()로 생성된 DataFrame
    metrics : list, optional
        비교할 지표 리스트
    top_n : int, optional
        표시할 상위 모델 개수
    figsize : tuple, optional
        그래프 크기
    save_path : str, optional
        그래프 저장 경로
    
    Returns:
    --------
    fig : matplotlib figure object
    """
    
    set_korean_font()
    
    if df.empty:
        print("⚠️  DataFrame이 비어있습니다.")
        return None
    
    # 기본 metrics
    if metrics is None:
        metrics = ['RMSLE', 'RMSE', 'MAE', 'R2']
    
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print(f"⚠️  지정한 metrics가 DataFrame에 없습니다.")
        return None
    
    # 상위 N개 선택
    if 'RMSLE' in df.columns:
        df_top = df.dropna(subset=['RMSLE']).head(top_n).copy()
    else:
        df_top = df.head(top_n).copy()
    
    # 히트맵 데이터
    heatmap_data = df_top[['model_name'] + available_metrics].set_index('model_name')
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=figsize)
    
    # 히트맵 (주의: RMSE/MAE/RMSLE는 낮을수록 좋음)
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn_r',  # _r: 역순
                cbar_kws={'label': 'Score'}, linewidths=0.5, ax=ax,
                vmin=heatmap_data.min().min() * 0.95, 
                vmax=heatmap_data.max().max() * 1.05)
    
    ax.set_title(f'회귀 모델 성능 히트맵 (Top {len(df_top)})\n※ RMSLE/RMSE/MAE: 낮을수록 좋음(초록), R2: 높을수록 좋음', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Models', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 히트맵 저장: {save_path}")
    
    return fig
# plot_metric_heatmap end ======================================


# check_data_quality start ###########################
def check_data_quality(df):
    """
    DataFrame의 데이터 품질 체크 (회귀 버전)
    
    Parameters:
    -----------
    df : pd.DataFrame
        체크할 DataFrame
    
    Notes:
    ------
    - 회귀 메트릭: R2, RMSE, MAE, RMSLE
    - NaN, Inf, 이상치 확인
    """
    print("="*80)
    print("📊 데이터 품질 체크 (회귀)")
    print("="*80)
    
    metrics = ['R2', 'RMSE', 'MAE', 'RMSLE']
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        print(f"\n[{metric}]")
        print(f"  - 총 개수: {len(df)}")
        print(f"  - NaN 개수: {df[metric].isna().sum()}")
        print(f"  - Inf 개수: {df[metric].isin([float('inf'), float('-inf')]).sum()}")
        
        valid_data = df[metric].dropna()
        if len(valid_data) > 0:
            print(f"  - 최소값: {valid_data.min():.6f}")
            print(f"  - 최대값: {valid_data.max():.6f}")
            print(f"  - 평균값: {valid_data.mean():.6f}")
            print(f"  - 중앙값: {valid_data.median():.6f}")
            print(f"  - 표준편차: {valid_data.std():.6f}")
        
        # NaN이 있는 행
        if df[metric].isna().any():
            print(f"  ⚠️  NaN 값이 있는 모델:")
            nan_models = df[df[metric].isna()]['model_name'].tolist()
            for model in nan_models[:5]:
                print(f"      - {model}")
        
        # 이상치 체크 (IQR 방식)
        if len(valid_data) > 0:
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[metric] < Q1 - 1.5*IQR) | (df[metric] > Q3 + 1.5*IQR)]['model_name'].tolist()
            if outliers:
                print(f"  ⚠️  이상치 가능성 있는 모델 ({len(outliers)}개):")
                for model in outliers[:3]:
                    val = df[df['model_name']==model][metric].values[0]
                    print(f"      - {model}: {val:.6f}")
# check_data_quality end ======================================


# plot_rmsle_ranking start ###########################
def plot_rmsle_ranking(df, top_n=15, save_path="../images/result_rmsle_ranking.png"):
    """
    RMSLE 순위 막대 그래프 (Kaggle 제출용)
    
    Parameters:
    -----------
    df : pd.DataFrame
        load_results_to_df()로 생성된 DataFrame
    top_n : int
        표시할 상위 모델 개수
    save_path : str, optional
        저장 경로
    """
    set_korean_font()
    
    if 'RMSLE' not in df.columns:
        print("⚠️  RMSLE 컬럼이 없습니다.")
        return None
    
    df_valid = df.dropna(subset=['RMSLE']).head(top_n).copy()
    
    if df_valid.empty:
        print("⚠️  유효한 RMSLE 데이터가 없습니다.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 순위 추가
    df_valid['rank'] = range(1, len(df_valid) + 1)
    df_valid['label'] = df_valid['rank'].astype(str) + '. ' + df_valid['model_name']
    
    # 색상 (1위부터 점점 연해짐)
    colors = plt.cm.RdYlGn(np.linspace(0.9, 0.3, len(df_valid)))
    
    bars = ax.barh(df_valid['label'], df_valid['RMSLE'], color=colors, 
                   alpha=0.85, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('RMSLE (Root Mean Squared Log Error)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model Ranking', fontsize=13, fontweight='bold')
    ax.set_title(f'Kaggle Mercari - RMSLE 순위 (낮을수록 좋음)\nTop {len(df_valid)} Models', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # 값 표시
    for val, bar in zip(df_valid['RMSLE'], bars):
        ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.6f}', 
               ha='left', va='center', fontsize=10, fontweight='bold', color='darkblue')
    
    ax.grid(axis='x', alpha=0.4, linestyle='--')
    ax.invert_yaxis()  # 1위가 위로
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 RMSLE 순위 저장: {save_path}")
    
    return fig
# plot_rmsle_ranking end ======================================
