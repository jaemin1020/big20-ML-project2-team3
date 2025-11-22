# utils/model_utils.py 파일 생성
"""
모델 저장 및 불러오기 유틸리티

Functions:
    - save_model: 모델을 pickle 파일로 저장
    - load_model: pickle 파일에서 모델 불러오기
    - list_saved_models: 저장된 모델 목록 출력
    - delete_model: 저장된 모델 삭제
"""
import pickle
import os
from pathlib import Path
from datetime import datetime

# SOF ------------------------------------------------------------------------- #
def save_model(model, model_name, folder='../models', add_timestamp=False):
    """
    학습된 모델을 pickle 파일로 저장
    
    Parameters:
    -----------
    model : object
        저장할 학습된 모델 객체
    model_name : str
        모델 파일명 (확장자 제외)
    folder : str, default='models'
        저장할 폴더 경로
    add_timestamp : bool, default=False
        파일명에 타임스탬프 추가 여부
    
    Returns:
    --------
    str : 저장된 파일의 전체 경로
    
    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier()
    >>> model.fit(X_train, y_train)
    >>> save_model(model, 'random_forest')
    'models/random_forest.pkl'
    """
    # 폴더 생성 (없으면)
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    # 타임스탬프 추가 옵션
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_{timestamp}.pkl"
    else:
        filename = f"{model_name}.pkl"
    
    # 전체 경로
    filepath = os.path.join(folder, filename)
    
    try:
        # 모델 저장
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        # 파일 크기 확인
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✓ 모델 저장 완료: {filepath}")
        print(f"  파일 크기: {file_size:.2f} MB")
        
        return filepath
        
    except Exception as e:
        print(f"✗ 모델 저장 실패: {e}")
        return None
# EOF ------------------------------------------------------------------------- #

def load_model(model_name, folder='../models'):
    """
    저장된 모델을 pickle 파일에서 불러오기
    
    Parameters:
    -----------
    model_name : str
        불러올 모델 파일명 (확장자 포함/제외 모두 가능)
    folder : str, default='models'
        모델이 저장된 폴더 경로
    
    Returns:
    --------
    object : 불러온 모델 객체, 실패 시 None
    
    Example:
    --------
    >>> model = load_model('random_forest')
    >>> predictions = model.predict(X_test)
    """
    # .pkl 확장자 처리
    if not model_name.endswith('.pkl'):
        model_name = f"{model_name}.pkl"
    
    filepath = os.path.join(folder, model_name)
    
    # 파일 존재 확인
    if not os.path.exists(filepath):
        print(f"✗ 파일을 찾을 수 없습니다: {filepath}")
        print(f"\n사용 가능한 모델 목록:")
        list_saved_models(folder)
        return None
    
    try:
        # 모델 불러오기
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ 모델 로드 완료: {filepath}")
        print(f"  모델 타입: {type(model).__name__}")
        
        return model
        
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        return None
# EOF ------------------------------------------------------------------------- #

def list_saved_models(folder='../models'):
    """
    저장된 모델 목록 출력
    
    Parameters:
    -----------
    folder : str, default='models'
        모델이 저장된 폴더 경로
    
    Returns:
    --------
    list : 저장된 모델 파일명 리스트
    """
    if not os.path.exists(folder):
        print(f"✗ 폴더가 존재하지 않습니다: {folder}")
        return []
    
    pkl_files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"저장된 모델이 없습니다. (폴더: {folder})")
        return []
    
    print(f"\n{'='*60}")
    print(f"저장된 모델 목록 (폴더: {folder})")
    print(f"{'='*60}")
    
    for i, filename in enumerate(sorted(pkl_files), 1):
        filepath = os.path.join(folder, filename)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        
        print(f"{i}. {filename}")
        print(f"   크기: {file_size:.2f} MB")
        print(f"   수정일: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    return pkl_files
# EOF ------------------------------------------------------------------------- #

def delete_model(model_name, folder='../models'):
    """
    저장된 모델 삭제
    
    Parameters:
    -----------
    model_name : str
        삭제할 모델 파일명
    folder : str, default='models'
        모델이 저장된 폴더 경로
    
    Returns:
    --------
    bool : 삭제 성공 여부
    """
    if not model_name.endswith('.pkl'):
        model_name = f"{model_name}.pkl"
    
    filepath = os.path.join(folder, model_name)
    
    if not os.path.exists(filepath):
        print(f"✗ 파일을 찾을 수 없습니다: {filepath}")
        return False
    
    try:
        os.remove(filepath)
        print(f"✓ 모델 삭제 완료: {filepath}")
        return True
    except Exception as e:
        print(f"✗ 모델 삭제 실패: {e}")
        return False
# EOF ------------------------------------------------------------------------- #      