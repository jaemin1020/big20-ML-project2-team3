import os
import datetime
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path


# plot_sampling_metrics ------------------------------------------------------------------------------
def plot_sampling_metrics(results, keys_to_plot=None, title="샘플링 평가 지표 비교", figsize = (10,6)):
    """
    샘플링 결과 비교 그래프 생성 함수
    
    Parameters
    ----------
    results : dict
        {"Model명": {"AUC":0.9, "정밀도":0.7 ...}, ...} 구조
    keys_to_plot : list, optional
        표시할 지표 리스트 (기본값: ['AUC','정밀도','재현율','F1'])
    title : str
        그래프 제목
    """
    rc('font', family='Malgun Gothic')  # Windows 기본 한글 폰트
    plt.rcParams['axes.unicode_minus'] = False
    
    if keys_to_plot is None:
        keys_to_plot = ['AUC', '정밀도', '재현율', 'F1']
    
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=figsize)

    # 차트 생성
    for i, key in enumerate(keys_to_plot):
        values = [results[model][key] for model in models]
        bars = plt.bar(x + i*width, values, width=width, label=key)
        
        # 각 막대 위 숫자 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, 
                     f"{height:.2f}", ha='center', fontsize=8)

    # x축, 제목, 범례 설정
    plt.xticks(x + width * (len(keys_to_plot)/2 - 0.5), models)
    plt.ylim(0, 1.05)
    plt.title(title + " (정확도 제외)")
    plt.ylabel("Score")
    plt.legend(title="평가 지표")
    
    save_path = Path(f'../images/{title}_{datetime.datetime.today().strftime("%Y_%m%d")}.png')
    version = 1
    while save_path.exists():
        save_path = Path(f'../images/{title}_{datetime.datetime.today().strftime("%Y_%m%d")}_{version:03d}.png')
        version += 1

    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📁 그래프 저장 완료: {save_path}")
    
    plt.show()
    
# eof -----------------------------------------------------------------------------------------------------------------------------------


# model_metrics_graph -------------------------------------------------------------------------------------------------------------------
def model_metrics_graph(results, title='모델별 성능 지표 비교', figsize=(20, 10)):
    """
    모델 성능 지표를 시각화하는 함수
    
    Parameters:
    -----------
    results : dict
        모델명을 key로, 성능지표 딕셔너리를 value로 가지는 딕셔너리
        예: {'model1': {'AUC': 0.95, '정밀도': 0.90, '재현율': 0.85, 'F1': 0.87}, ...}
    title : str, default='모델별 성능 지표 비교'
        그래프 전체 제목
    figsize : tuple, default=(20, 10)
        그래프 크기 (가로, 세로) 인치 단위
    
    Returns:
    --------
    None (matplotlib 그래프를 화면에 출력 및 저장)
    """

    try:
        # ========================================
        # 1. 데이터 준비 및 validation
        # ========================================
        if not isinstance(results, dict):
            raise ValueError("results는 dict 형태여야 합니다.")
        if len(results) == 0:
            raise ValueError("results가 비어 있습니다.")
        
        df_results = pd.DataFrame(results).T  # DataFrame 생성 (행: 모델명, 열: 성능지표)

        # 필요한 지표가 모두 있는지 확인
        required_metrics = ['AUC', '정밀도', '재현율', 'F1']
        for metric in required_metrics:
            if metric not in df_results.columns:
                raise ValueError(f"'{metric}' 지표가 results에 없습니다.")

        # 날짜 yyyy_mmdd 형식
        today = datetime.datetime.today().strftime("%Y_%m%d")

        # 파일명 생성 (title 공백 → '_')
        safe_title = title.replace(" ", "_")
        filename = f"../images/{safe_title}_{today}.png"

        # ========================================
        # 2. 그래프 스타일 설정
        # ========================================
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        # ========================================
        # 3. 개별 지표별 히스토그램 (2x2 서브플롯)
        # ========================================
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=20, fontweight='bold')

        metrics = required_metrics
        colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c']

        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            x_pos = np.arange(len(df_results.index))
            bars = ax.bar(x_pos, df_results[metric], color=color, alpha=0.7, edgecolor='black')

            ax.set_xlabel('모델', fontsize=12, fontweight='bold')
            ax.set_ylabel('점수', fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} 비교', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_results.index, rotation=45, ha='right', fontsize=9)
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            mean_val = df_results[metric].mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'평균: {mean_val:.4f}')
            ax.legend()

            max_idx = df_results[metric].idxmax()
            max_val = df_results[metric].max()
            max_pos = df_results.index.get_loc(max_idx)
            ax.text(max_pos, max_val + 0.02, f'{max_val:.4f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)  # 그래프 저장
        plt.show()

        # ========================================
        # 4. 통계 요약 출력
        # ========================================
        print("\n=== 지표별 통계 요약 ===")
        for metric in metrics:
            print(f"\n[{metric}]")
            print(f"  최고: {df_results[metric].max():.4f} ({df_results[metric].idxmax()})")
            print(f"  최저: {df_results[metric].min():.4f} ({df_results[metric].idxmin()})")
            print(f"  평균: {df_results[metric].mean():.4f}")
            print(f"  표준편차: {df_results[metric].std():.4f}")

        # ========================================
        # 5. 전체 지표를 한 그래프에 표시 (Grouped Bar Chart)
        # ========================================
        fig2, ax2 = plt.subplots(figsize=figsize)
        x = np.arange(len(df_results.index))
        width = 0.2

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (i - 1.5)
            ax2.bar(x + offset, df_results[metric], width, label=metric, color=color, alpha=0.7)

        ax2.set_xlabel('모델', fontsize=14, fontweight='bold')
        ax2.set_ylabel('점수', fontsize=14, fontweight='bold')
        ax2.set_title(title, fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_results.index, rotation=45, ha='right', fontsize=10)
        ax2.legend(fontsize=12)
        ax2.set_ylim([0, 1.0])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(filename.replace(".png", "_grouped.png"), dpi=300)  # 두 번째 그래프 저장
        plt.show()

        # ========================================
        # 6. 상위 모델 분석 (F1 Score 기준)
        # ========================================
        print("\n=== F1 Score 기준 상위 5개 모델 ===")
        top5_f1 = df_results.nlargest(5, 'F1')
        print(top5_f1[metrics])

        print(f"\n그래프가 저장되었습니다: {filename}")

    except Exception as e:
        print(f"[에러 발생] {e}")
        
# eof -----------------------------------------------------------------------------------------------------------------------------------    