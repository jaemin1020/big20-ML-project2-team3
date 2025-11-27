import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

def plot_sampling_metrics(results, keys_to_plot=None, title="샘플링 평가 지표 비교"):
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

    plt.figure(figsize=(10,6))

    # 차트 생성
    for i, key in enumerate(keys_to_plot):
        values = [results[model][key] for model in models]
        bars = plt.bar(x + i*width, values, width=width, label=key)
        
        # 각 막대 위 숫자 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, 
                     f"{height:.3f}", ha='center', fontsize=8)

    # x축, 제목, 범례 설정
    plt.xticks(x + width * (len(keys_to_plot)/2 - 0.5), models)
    plt.ylim(0, 1.05)
    plt.title(title + " (정확도 제외)")
    plt.ylabel("Score")
    plt.legend(title="평가 지표")
    
    plt.show()