# result_report.py results 폴더를 읽어 평가표 작성하기 
import os
import json
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
import platform

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# 한글 폰트 설정 ##############################################################
def set_korean_font():
    """
    운영체제에 맞는 한글 폰트 설정
    """
    system = platform.system()
    
    if system == 'Windows':
        plt.rc('font', family='Malgun Gothic')  # 맑은 고딕
    elif system == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    else:  # Linux
        plt.rc('font', family='NanumGothic')
    
    # 마이너스 기호 깨짐 방지
    plt.rc('axes', unicode_minus=False)
    
    print(f"✅ 한글 폰트 설정 완료: {system}")
    

#  load_results_to_df ##############################################################
def load_results_to_df(
    folder_path='../results', 
    sort_by='F1', 
    ascending=False,
    filter_pattern=None,
    exclude_patterns=None
):
    """
    results 폴더의 .txt, .json 파일을 읽어서 DataFrame으로 변환
    
    Parameters:
    -----------
    folder_path : str
        결과 파일들이 저장된 폴더 경로
    sort_by : str, optional
        정렬 기준 컬럼명 (기본값: 'F1')
    ascending : bool, optional
        오름차순 정렬 여부 (기본값: False)
    filter_pattern : str or list, optional
        파일명에 포함되어야 할 패턴 (예: '_ho_', ['_ho_', 'best'])
    exclude_patterns : str or list, optional
        파일명에서 제외할 패턴 (예: ['EDA_', 'Corr'])
    
    Returns:
    --------
    pd.DataFrame
        model_name과 result_dict의 값들을 컬럼으로 하는 DataFrame
    
    Examples:
    ---------
    >>> # HyperOpt 결과만 로드
    >>> df = load_results_to_df('../results', filter_pattern='_ho_')
    
    >>> # 여러 패턴 중 하나라도 포함
    >>> df = load_results_to_df('../results', filter_pattern=['_ho_', '_HO_'])
    """
    
    results_list = []
    error_files = []
    no_result_dict = []
    filtered_out = []
    
    # 폴더가 존재하는지 확인
    if not os.path.exists(folder_path):
        print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        return pd.DataFrame()
    
    # .txt와 .json 파일 찾기
    files = []
    for ext in ['*.txt', '*.json']:
        files.extend(Path(folder_path).glob(ext))
    
    # 기본 제외 패턴
    default_exclude = ['__filelist', 'EDA_', 'CorrMatrix']
    if exclude_patterns:
        if isinstance(exclude_patterns, str):
            exclude_patterns = [exclude_patterns]
        default_exclude.extend(exclude_patterns)
    
    # 필터 패턴을 리스트로 변환
    if filter_pattern and isinstance(filter_pattern, str):
        filter_pattern = [filter_pattern]
    
    if not files:
        print(f"⚠️  폴더에 .txt 또는 .json 파일이 없습니다: {folder_path}")
        return pd.DataFrame()
    
    print(f"📂 총 {len(files)}개 파일 발견")
    if filter_pattern:
        print(f"🔍 필터 패턴: {filter_pattern}")
    if default_exclude:
        print(f"🚫 제외 패턴: {default_exclude}")
    print()
    
    for file_path in files:
        # 제외 패턴 체크
        if any(pattern in file_path.name for pattern in default_exclude):
            continue
        
        # 필터 패턴 체크 (AND 조건)
        if filter_pattern:
            if not any(pattern in file_path.name for pattern in filter_pattern):
                filtered_out.append(file_path.name)
                continue
        
        try:
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # result_dict가 있는지 확인
            if isinstance(data, dict) and 'result_dict' in data:
                # 파일명에서 모델명 추출 (다양한 날짜 패턴 제거)
                model_name = file_path.stem
                
                # 날짜/시간 패턴들 제거
                model_name = re.sub(r'_\d{8}_\d{6}$', '', model_name)  # _20251201_154210
                model_name = re.sub(r'_\d{8}$', '', model_name)        # _20251201
                model_name = re.sub(r'_\d{4}_\d{2}_\d{2}$', '', model_name)  # _2025_11_27
                model_name = re.sub(r'__+', '_', model_name)           # 이중 언더스코어
                model_name = model_name.rstrip('_')                     # 끝 언더스코어
                
                # result_dict 추출
                result_dict = data['result_dict']
                
                # model_name을 포함한 딕셔너리 생성
                row = {'model_name': model_name, 'file_name': file_path.name}
                row.update(result_dict)
                
                # 추가 정보 포함 (있는 경우)
                if '실행 시간' in data:
                    if isinstance(data['실행 시간'], set):
                        row['실행시간'] = list(data['실행 시간'])[0] if data['실행 시간'] else None
                    else:
                        row['실행시간'] = data['실행 시간']
                elif '실행시간' not in row and 'result_dict' in data and '실행시간' in data['result_dict']:
                    pass  # 이미 포함됨
                
                results_list.append(row)
                print(f"✅ {model_name[:70]}")
            else:
                no_result_dict.append(file_path.name)
                
        except json.JSONDecodeError as e:
            error_files.append((file_path.name, f"JSON 파싱 오류: {e}"))
            print(f"❌ JSON 파싱 오류: {file_path.name}")
        except Exception as e:
            error_files.append((file_path.name, str(e)))
            print(f"❌ 파일 읽기 오류: {file_path.name} - {e}")
    
    # DataFrame 생성
    if results_list:
        df = pd.DataFrame(results_list)
        
        # model_name을 첫 번째 컬럼으로 이동
        cols = ['model_name', 'file_name'] + [col for col in df.columns if col not in ['model_name', 'file_name']]
        df = df[cols]
        
        # 정렬
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        
        print(f"\n{'='*80}")
        print(f"✅ 총 {len(df)}개 모델 결과 로드 완료")
        print(f"📊 컬럼: {list(df.columns)}")
        
        if filtered_out:
            print(f"🔍 필터로 제외된 파일: {len(filtered_out)}개")
        if error_files:
            print(f"⚠️  오류 파일: {len(error_files)}개")
        if no_result_dict:
            print(f"⚠️  result_dict 없는 파일: {len(no_result_dict)}개")
        print(f"{'='*80}\n")
        
        return df
    else:
        print("\n⚠️  유효한 결과가 없습니다.")
        if error_files:
            print("\n오류 파일 목록:")
            for fname, err in error_files[:10]:  # 최대 10개만 표시
                print(f"  - {fname}: {err}")
        return pd.DataFrame()
# eof -------------------------------------------------------------------    

#  plot_model_comparison ##############################################################
def plot_model_comparison(df, metrics=None, top_n=20, figsize=(16, 12), save_path=None):
    """
    모델 비교 그래프 생성
    
    Parameters:
    -----------
    df : pd.DataFrame
        load_results_to_df()로 생성된 DataFrame
    metrics : list, optional
        비교할 지표 리스트 (기본값: ['AUC', '정밀도', '재현율', 'F1', 'F2'])
    top_n : int, optional
        표시할 상위 모델 개수 (기본값: 20)
    figsize : tuple, optional
        그래프 크기 (기본값: (16, 12))
    save_path : str, optional
        그래프 저장 경로 (기본값: None, 저장 안함)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # 한글 폰트 설정
    # set_korean_font()
    
    if df.empty:
        print("⚠️  DataFrame이 비어있습니다.")
        return None, None
    
    # 기본 metrics 설정
    if metrics is None:
        metrics = ['AUC', '정밀도', '재현율', 'F1', 'F2']
    
    # 사용 가능한 metrics만 필터링
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print(f"⚠️  지정한 metrics가 DataFrame에 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
        return None, None
    
    # 상위 N개 모델만 선택
    df_top = df.head(top_n).copy()
    
    # NaN, Inf 값 체크 및 제거
    for metric in available_metrics:
        if df_top[metric].isna().any():
            print(f"⚠️  '{metric}' 컬럼에 NaN 값이 있습니다. 해당 행을 제외합니다.")
            df_top = df_top[df_top[metric].notna()]
        
        if df_top[metric].isin([float('inf'), float('-inf')]).any():
            print(f"⚠️  '{metric}' 컬럼에 Inf 값이 있습니다. 해당 행을 제외합니다.")
            df_top = df_top[~df_top[metric].isin([float('inf'), float('-inf')])]
    
    if df_top.empty:
        print("⚠️  유효한 데이터가 없습니다.")
        return None, None
    
    # 그래프 스타일 설정
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 서브플롯 개수 계산
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # 각 metric에 대한 막대 그래프
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # 해당 metric의 유효한 데이터만 선택
        valid_data = df_top[df_top[metric].notna() & 
                            ~df_top[metric].isin([float('inf'), float('-inf')])].copy()
        
        if valid_data.empty:
            ax.text(0.5, 0.5, f'{metric}: 유효한 데이터 없음', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            continue
        
        # 수평 막대 그래프
        bars = ax.barh(valid_data['model_name'], valid_data[metric], alpha=0.8)
        
        # 색상 그라데이션 처리
        metric_min = valid_data[metric].min()
        metric_max = valid_data[metric].max()
        
        # AUC=0인 경우 특별 처리
        if metric == 'AUC':
            colors = []
            for val in valid_data[metric]:
                if val == 0.0:
                    colors.append('#CCCCCC')  # 회색 (예측 불가능)
                else:
                    # 0이 아닌 값들로만 정규화
                    non_zero_vals = valid_data[metric][valid_data[metric] > 0]
                    if len(non_zero_vals) > 0:
                        nz_min = non_zero_vals.min()
                        nz_max = non_zero_vals.max()
                        if nz_min == nz_max:
                            colors.append('#90EE90')
                        else:
                            normalized = (val - nz_min) / (nz_max - nz_min)
                            colors.append(plt.cm.RdYlGn(normalized))
                    else:
                        colors.append('#90EE90')
        else:
            # min과 max가 같은 경우 처리
            if metric_min == metric_max:
                colors = ['#90EE90'] * len(valid_data)  # 연두색으로 통일
            else:
                colors = plt.cm.RdYlGn((valid_data[metric] - metric_min) / (metric_max - metric_min))
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} 비교 (Top {len(valid_data)})', fontsize=14, fontweight='bold', pad=10)
        
        # 값 표시
        for i, (val, bar) in enumerate(zip(valid_data[metric], bars)):
            # AUC=0인 경우 특별 표시
            if metric == 'AUC' and val == 0.0:
                label = 'N/A'
            else:
                label = f'{val:.4f}'
            
            ax.text(val, bar.get_y() + bar.get_height()/2, label, 
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 그리드 추가
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # x축 범위 설정 (안전하게)
        if metric_min != metric_max:
            x_min = metric_min * 0.95 if metric_min > 0 else metric_min * 1.05
            x_max = metric_max * 1.05 if metric_max > 0 else metric_max * 0.95
            ax.set_xlim(x_min, x_max)
        
        # y축 레이블 크기 조정
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=10)
    
    # 사용하지 않는 서브플롯 숨기기
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 그래프 저장: {save_path}")
    
    return fig, axes
# eof -------------------------------------------------------------------    


#  plot_metric_heatmap ##############################################################
def plot_metric_heatmap(df, metrics=None, top_n=20, figsize=(12, 10), save_path=None):
    """
    모델별 metric 히트맵 생성
    
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
    """
    
    # 한글 폰트 설정
    # set_korean_font()
    
    if df.empty:
        print("⚠️  DataFrame이 비어있습니다.")
        return None
    
    # 기본 metrics 설정
    if metrics is None:
        metrics = ['AUC', '정밀도', '재현율', 'F1', 'F2']
    
    # 사용 가능한 metrics만 필터링
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print(f"⚠️  지정한 metrics가 DataFrame에 없습니다.")
        return None
    
    # 상위 N개 모델 선택
    df_top = df.head(top_n).copy()
    
    # 히트맵 데이터 준비
    heatmap_data = df_top[['model_name'] + available_metrics].set_index('model_name')
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=figsize)
    
    # 히트맵 그리기
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar_kws={'label': 'Score'}, linewidths=0.5, ax=ax,
                vmin=heatmap_data.min().min() * 0.95, 
                vmax=heatmap_data.max().max())
    
    ax.set_title(f'모델 성능 히트맵 (Top {len(df_top)})', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Models', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 히트맵 저장: {save_path}")
    
    return fig
# eof ------------------------------------------------------------------- 

   
# check_data_quality ##############################################################
def check_data_quality(df):
    """DataFrame의 데이터 품질 체크"""
    print("="*80)
    print("📊 데이터 품질 체크")
    print("="*80)
    
    metrics = ['AUC', '정밀도', '재현율', 'F1', 'F2']
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        print(f"\n[{metric}]")
        print(f"  - 총 개수: {len(df)}")
        print(f"  - NaN 개수: {df[metric].isna().sum()}")
        print(f"  - Inf 개수: {df[metric].isin([float('inf'), float('-inf')]).sum()}")
        print(f"  - 0.0 개수: {(df[metric] == 0.0).sum()}")
        print(f"  - 최소값: {df[metric].min()}")
        print(f"  - 최대값: {df[metric].max()}")
        print(f"  - 평균값: {df[metric].mean():.4f}")
        
        # NaN이 있는 행 표시
        if df[metric].isna().any():
            print(f"  ⚠️  NaN 값이 있는 모델:")
            nan_rows = df[df[metric].isna()]['model_name'].tolist()
            for model in nan_rows[:5]:  # 최대 5개만
                print(f"      - {model}")
        
        # 0.0인 행 표시 (AUC인 경우)
        if metric == 'AUC' and (df[metric] == 0.0).any():
            print(f"  ⚠️  AUC=0.0 (확률 예측 불가) 모델:")
            zero_rows = df[df[metric] == 0.0]['model_name'].tolist()
            for model in zero_rows[:5]:
                print(f"      - {model}")


# ==================== 사용 예시 ====================

if __name__ == "__main__":
    
    # 1️⃣ HyperOpt 결과만 로드 (_ho_ 포함)
    print("="*80)
    print("🔍 HyperOpt 최적화 결과 로드")
    print("="*80 + "\n")
    
    df_ho = load_results_to_df(
        folder_path='../results',
        filter_pattern='_ho_',  # _ho_ 포함된 파일만
        sort_by='F1',
        ascending=False
    )
    
    if not df_ho.empty:
        # 데이터 품질 체크
        check_data_quality(df_ho)
        
        # 데이터 요약 출력
        print("\n📊 상위 10개 모델:")
        display_cols = ['model_name', 'AUC', 'F1', 'F2', '정밀도', '재현율']
        available_cols = [col for col in display_cols if col in df_ho.columns]
        print(df_ho[available_cols].head(10).to_string(index=False))
        
        # 통계 요약
        print("\n📈 통계 요약:")
        stats_cols = ['AUC', 'F1', 'F2', '정밀도', '재현율']
        available_stats = [col for col in stats_cols if col in df_ho.columns]
        if available_stats:
            print(df_ho[available_stats].describe().round(4))
        
        # 2️⃣ 비교 그래프 생성
        print("\n" + "="*80)
        print("📊 그래프 생성 중...")
        print("="*80 + "\n")
        
        # 막대 그래프
        fig1, axes1 = plot_model_comparison(
            df_ho,
            metrics=['AUC', '정밀도', '재현율', 'F1', 'F2'],
            top_n=15,
            figsize=(16, 12),
            save_path='../results/hyperopt_comparison.png'
        )
        
        # 히트맵
        fig2 = plot_metric_heatmap(
            df_ho,
            metrics=['AUC', '정밀도', '재현율', 'F1', 'F2'],
            top_n=15,
            figsize=(12, 10),
            save_path='../results/hyperopt_heatmap.png'
        )
        
        plt.show()
        
        # 3️⃣ CSV 저장
        csv_path = '../results/hyperopt_results.csv'
        df_ho.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 결과 저장: {csv_path}")