import os
import sys

# 현재 작업 디렉토리 기준으로 상위 1단계 폴더를 루트로 설정
# current_dir  = os.getcwd()
# project_root = os.path.abspath(os.path.join(current_dir, '..'))
project_root = "c:/big20/git/big20-ML-project2-team3/CreditCardFraud"

# sys.path에 추가 (모듈 import용)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# print("프로젝트 루트로 설정된 경로:", project_root)
# print(sys.path)

import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score # hyperopt_tune 에서 사용
from sklearn.preprocessing import StandardScaler

from functools import partial
from typing import Dict, Any, Callable
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK
from hyperopt import space_eval

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import json
import pickle


def hyperopt_search(model_class, search_space, X_train, y_train,
                    max_evals=100, cv=5, scoring='roc_auc', random_state=23,
                    save_trials=True, trials_path='../models/trials/',
                    verbose=True, use_gpu=False):
    """
    HyperOpt를 사용한 하이퍼파라미터 탐색 함수 (학습 및 평가는 별도)
    
    Parameters:
    -----------
    model_class : class
        sklearn 모델 클래스 (예: LinearSVC, XGBClassifier)
    search_space : dict
        hyperopt search space 딕셔너리
    X_train : array-like
        학습 데이터
    y_train : array-like
        학습 레이블
    max_evals : int, default=100
        최적화 시도 횟수
    cv : int, default=5
        교차 검증 fold 수
    scoring : str, default='roc_auc'
        평가 지표 ('roc_auc', 'f1', 'recall' 등)
    random_state : int, default=23
        랜덤 시드
    save_trials : bool, default=True
        trials 객체 저장 여부 (중단 시 복구 가능)
    trials_path : str, default='../results/trials/'
        trials 저장 경로
    verbose : bool, default=True
        출력 여부
    use_gpu : bool, default=False (LGBM, XGB, Catboost = True)
        GPU 사용 여부
    
    Returns:
    --------
    dict : {
        'best_params': 최적 하이퍼파라미터 (변환 완료),
        'best_score': 최적 점수,
        'trials': hyperopt trials 객체,
        'elapsed_time': 탐색 시간,
        'model_name': 모델 이름
    }
    """
    
    # 모델 이름 자동 추출
    model_name = model_class.__name__
    
    if verbose:
        if use_gpu:
            print(f"GPU 모드 활성화 ({model_name})")
        print("=" * 50)
        print(f"  {model_name} 하이퍼파라미터 탐색 시작")
        print("=" * 50)
    
    start_time = time.time()  # 시작 시간 기록
    
    # Objective 함수 정의
    def objective(params):
        # 파라미터 타입 변환
        converted_params = {}
        for key, value in params.items():
            # quniform으로 정의된 정수형 파라미터 변환
            if key in ['n_estimators', 'max_depth', 'min_child_weight', 'max_iter', 'scale_pos_weight']:
                converted_params[key] = int(value)
            else:
                converted_params[key] = value
        
        # random_state 추가
        converted_params['random_state'] = random_state
        
        # GPU 설정 추가
        if use_gpu:
            if 'XGB' in model_name:
                converted_params['tree_method'] = 'gpu_hist'
                converted_params['gpu_id'] = 0
            elif 'LGBM' in model_name or 'LightGBM' in model_name:
                converted_params['device'] = 'gpu'
                converted_params['gpu_platform_id'] = 0
                converted_params['gpu_device_id'] = 0
            elif 'CatBoost' in model_name:
                converted_params['task_type'] = 'GPU'
                converted_params['devices'] = '0'
        
        # 모델 생성
        try:
            model = model_class(**converted_params)
        except Exception as e:
            print(f"모델 생성 오류: {e}")
            print(f"문제 파라미터: {converted_params}")
            return {'loss': 1.0, 'status': STATUS_OK}
        
        # 교차 검증
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            score = scores.mean()
        except Exception as e:
            print(f"교차 검증 오류: {e}")
            return {'loss': 1.0, 'status': STATUS_OK}
        
        return {'loss': -score, 'status': STATUS_OK}
    
    # 최적화 실행
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(random_state)
    )
    
    # 걸린 시간 계산
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\n탐색 완료 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
    
    # 최적 점수 추출
    best_score = -trials.best_trial['result']['loss']
    if verbose:
        print(f"최적 {scoring}: {best_score:.4f}")
    
    # best_params 변환 (choice 타입 처리) - 완전 개선 버전
    final_params = {}

    # ===== 0단계: integer_params 먼저 정의 =====
    integer_params = [
        # Common
        'n_estimators', 'max_depth', 'random_state',
        # XGBoost/LightGBM
        'min_child_weight', 'scale_pos_weight', 'num_leaves', 
        'min_child_samples', 'min_data_in_leaf', 'bagging_freq',
        # sklearn
        'max_iter', 'min_samples_split', 'min_samples_leaf',
        # CatBoost
        'iterations', 'depth'
    ]

    # ===== 1단계: search_space에서 choice 매핑 추출 시도 =====
    choice_mappings = {}
    for key, space_def in search_space.items():
        try:
            # hyperopt choice 객체에서 선택지 추출
            if hasattr(space_def, 'pos_args') and len(space_def.pos_args) > 1:
                choices = space_def.pos_args[1]
                if hasattr(choices, 'obj'):
                    extracted = choices.obj
                    # 리스트인지 확인
                    if isinstance(extracted, (list, tuple)):
                        choice_mappings[key] = extracted
                elif isinstance(choices, (list, tuple)):
                    choice_mappings[key] = choices
        except Exception as e:
            if verbose:
                print(f"⚠️ {key} 추출 실패: {e}")

    # ===== 2단계: 일반적인 모델별 choice 매핑 (fallback) =====
    default_choice_mappings = {
        # SGD (SGDClassifier에만 적용)
        'loss': ['hinge', 'log_loss', 'modified_huber'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        
        # SVM
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'dual': [False, True],
        
        # Tree models
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        
        # General
        'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],
        'activation': ['relu', 'tanh', 'logistic', 'identity'],
        'class_weight': [None, 'balanced'],
    }
    
    # SGD의 learning_rate만 choice (다른 모델은 float)
    if 'SGD' in model_name:
        default_choice_mappings['learning_rate'] = ['constant', 'optimal', 'invscaling', 'adaptive']

    # 추출 실패한 choice 파라미터는 기본 매핑 사용
    for key in best_params.keys():
        if key not in choice_mappings and key in default_choice_mappings:
            choice_mappings[key] = default_choice_mappings[key]

    # None 값 필터링 및 문자열 제거 (안전장치)
    choice_mappings = {
        k: v for k, v in choice_mappings.items() 
        if v is not None and isinstance(v, (list, tuple))
    }

    if verbose and choice_mappings:
        print("\n감지된 choice 파라미터:")
        for key, choices in choice_mappings.items():
            print(f"  - {key}: {choices} (타입: {type(choices).__name__}, 길이: {len(choices)})")

    # ===== 3단계: 파라미터 변환 =====
    for key, value in best_params.items():
        # choice 파라미터 처리
        if key in choice_mappings:
            mapping = choice_mappings[key]
            # 리스트/튜플인지 다시 확인
            if not isinstance(mapping, (list, tuple)):
                print(f"⚠️ {key}의 mapping이 리스트가 아닙니다: {type(mapping)} = {mapping}")
                final_params[key] = value
                continue
                
            try:
                idx = int(value)
                if idx < 0 or idx >= len(mapping):
                    print(f"⚠️ {key} 인덱스 범위 초과: idx={idx}, 선택지={mapping}")
                    final_params[key] = value
                else:
                    final_params[key] = mapping[idx]
            except (IndexError, ValueError, TypeError) as e:
                print(f"⚠️ {key} 변환 실패 (idx={value}): {e}")
                final_params[key] = value
        # quniform으로 정의된 정수형 파라미터
        elif key in integer_params:
            final_params[key] = int(value)
        # float 파라미터 (numpy → Python float 변환)
        else:
            if isinstance(value, np.floating):
                final_params[key] = float(value)
            elif isinstance(value, np.integer):
                final_params[key] = int(value)
            else:
                final_params[key] = value

    # random_state 추가
    final_params['random_state'] = random_state

    # GPU 설정 추가
    if use_gpu:
        if 'XGB' in model_name:
            final_params['tree_method'] = 'gpu_hist'
            final_params['gpu_id'] = 0
        elif 'LGBM' in model_name or 'LightGBM' in model_name:
            final_params['device'] = 'gpu'
            final_params['gpu_platform_id'] = 0
            final_params['gpu_device_id'] = 0
        elif 'CatBoost' in model_name:
            final_params['task_type'] = 'GPU'
            final_params['devices'] = '0'

    if verbose:
        print("\n최적 하이퍼파라미터:")
        for param_name, param_value in final_params.items():
            print(f"  - {param_name}: {param_value}")
    
    # trials 객체 저장 (중단 시 복구 가능)
    if save_trials:
        os.makedirs(trials_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trials_filename = f"{model_name}_trials_{timestamp}.pkl"
        trials_save_path = os.path.join(trials_path, trials_filename)
        
        with open(trials_save_path, 'wb') as f:
            pickle.dump(trials, f)
        
        if verbose:
            print(f"\n✓ Trials 객체 저장 완료: {trials_save_path}")
        
        # JSON 저장 시도 (실패해도 진행)
        try:
            def convert_numpy_types(obj):
                """numpy 타입을 재귀적으로 Python 기본 타입으로 변환"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            params_data = {
                'model_name': model_name,
                'best_params': convert_numpy_types(final_params),
                'best_score': float(best_score),
                'elapsed_time': float(elapsed_time),
                'max_evals': int(max_evals),
                'cv': int(cv),
                'scoring': scoring,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            params_filename = f"{model_name}_best_params_{timestamp}.json"
            params_save_path = os.path.join(trials_path, params_filename)
            
            with open(params_save_path, 'w', encoding='utf-8') as f:
                json.dump(params_data, f, ensure_ascii=False, indent=4)
            
            if verbose:
                print(f"✓ 최적 파라미터 저장 완료 (JSON): {params_save_path}")
                
        except Exception as e:
            if verbose:
                print(f"⚠️ JSON 저장 실패 (pickle로 대체): {e}")
            # pickle로 대체 저장
            try:
                result_filename = f"{model_name}_result_{timestamp}.pkl"
                result_save_path = os.path.join(trials_path, result_filename)
                with open(result_save_path, 'wb') as f:
                    pickle.dump({
                        'best_params': final_params,
                        'best_score': best_score,
                        'elapsed_time': elapsed_time
                    }, f)
                if verbose:
                    print(f"✓ pickle로 대체 저장: {result_save_path}")
            except Exception as pickle_err:
                if verbose:
                    print(f"⚠️ pickle 저장도 실패: {pickle_err}")
    
    # 반환
    return {
        'best_params': final_params,
        'best_score': best_score,
        'trials': trials,
        'elapsed_time': elapsed_time,
        'model_name': model_name
    }
# eof -------------------------------------------------------------------------------------


def train_and_evaluate(model_class, params, X_train, y_train, X_test, y_test,
                       save_model=True, save_model_path='../models/',
                       save_result_path='../results/', verbose=True):
    """
    찾은 하이퍼파라미터로 최종 모델 학습 및 평가 (X_Train 전체를 사용하고 cross_validation하는 방식임)
    
    Parameters:
    -----------
    model_class : class
        sklearn 모델 클래스
    params : dict
        하이퍼파라미터 딕셔너리
    X_train, y_train : array-like
        학습 데이터
    X_test, y_test : array-like
        테스트 데이터
    save_model : bool, default=True
        모델 저장 여부
    save_model_path : str, default='../models/'
        모델 저장 경로
    save_result_path : str, default='../results/'
        결과 저장 경로
    verbose : bool, default=True
        출력 여부
    
    Returns:
    --------
    dict : {
        'model': 학습된 모델,
        'metrics': 평가 지표,
        'confusion_matrix': 혼동 행렬,
        'result_dict': 결과 딕셔너리
    }
    
    Examples:
    ---------
    >>> # hyperopt_search로 찾은 파라미터 사용
    >>> result = train_and_evaluate(
    ...     model_class=LinearSVC,
    ...     params=search_result['best_params'],
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ...     save_model=True,
    ...     verbose=True
    ... )
    """
    
    model_name = model_class.__name__
    
    if verbose:
        print("=" * 50)
        print(f"  {model_name} 최종 학습 및 평가")
        print("=" * 50)
        print(f"사용 파라미터: {params}")
    
    start_time = time.time()
    
    # 모델 생성 및 학습
    try:
        model = model_class(**params)
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"❌ 모델 학습 오류: {e}")
        print(f"문제 파라미터: {params}")
        raise
    
    fit_time = time.time() - start_time
    if verbose:
        print(f"✓ 학습 완료 ({fit_time:.2f}초)")
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 확률 예측 (가능한 경우)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None
    except:
        y_proba = None
    
    # 평가 지표 계산
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # AUC 계산
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred)
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    
    if verbose:
        print("\n평가 결과:")
        for metric_name, metric_value in metrics.items():
            print(f"  - {metric_name}: {metric_value:.4f}")
        print(f"\n혼동 행렬:\n{cm}")
    
    # 모델 저장
    if save_model:
        os.makedirs(save_model_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_save_path = os.path.join(save_model_path, model_filename)
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)
        
        file_size = os.path.getsize(model_save_path) / (1024 * 1024)
        if verbose:
            print(f"\n✓ 모델 저장: {model_save_path} ({file_size:.2f} MB)")
    
    # 결과 딕셔너리
    result_dict = {
        'AUC': round(metrics.get('roc_auc', 0), 4),
        '정확도': round(metrics['accuracy'], 4),
        '정밀도': round(metrics['precision'], 4),
        '재현율': round(metrics['recall'], 4),
        'F1': round(metrics['f1'], 4)
    }
    
    # 결과 저장
    os.makedirs(save_result_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"{model_name}_result_{timestamp}.json"
    result_save_path = os.path.join(save_result_path, result_filename)
    
    data = {
        "result_dict": result_dict,
        "오차행렬": cm.tolist(),
        "학습 시간": fit_time,
        "하이퍼파라미터": params,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(result_save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    ########################################################################    
    # 불러오기 예시
    # with open(save_path, "r", encoding="utf-8") as f:
    #     loaded_data = json.load(f)
    # print(loaded_data["result_dict"])   # 원하는 부분만 꺼내기
    ########################################################################        
    
    
    if verbose:
        print(f"✓ 결과 저장: {result_save_path}")
    
    return {
        'model': model,
        'metrics': metrics,
        'confusion_matrix': cm,
        'result_dict': result_dict,
        'fit_time': fit_time
    }
# eof ----------------------------------------------------------------------------------------------------

def extract_best_params_from_trials(trials_path, choice_mappings, integer_params=['max_iter'], random_state=23):
    """trials 객체에서 최적 파라미터 추출"""
    import pickle
    import numpy as np
    
    # trials 로드
    with open(trials_path, 'rb') as f:
        trials = pickle.load(f)
    
    # 최적 trial 찾기
    best_idx = np.argmin([trial['result']['loss'] for trial in trials.trials])
    best_trial = trials.trials[best_idx]
    best_vals = best_trial['misc']['vals']
    
    # 파라미터 변환
    final_params = {}
    for key, value_list in best_vals.items():
        if len(value_list) == 0:
            continue
        
        value = value_list[0]
        
        if key in choice_mappings:
            final_params[key] = choice_mappings[key][int(value)]
        elif key in integer_params:
            final_params[key] = int(value)
        else:
            final_params[key] = float(value)
    
    final_params['random_state'] = random_state
    
    # 최적 점수
    best_score = -best_trial['result']['loss']
    
    return {
        'best_params': final_params,
        'best_score': best_score
    }
# eof ----------------------------------------------------------------------------------------------------