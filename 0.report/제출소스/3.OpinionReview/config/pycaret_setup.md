# 🧠 PyCaret 전용 환경 설정 가이드

PyCaret은 기존 ML 환경에 설치하면 `numpy`, `scikit-learn`, `lightgbm`, `matplotlib`
등 버전 충돌이 발생하기 때문에 **별도의 Conda 환경에서 사용**하는 것이 가장 안정적입니다.

---

## 🚀 1. 새 환경 생성

Python 권장 버전: **3.8 ~ 3.10**

```sh
conda create -n pycaret_env python=3.10 -y

📌 2. 환경 활성화
conda activate pycaret_env

📦 3. PyCaret 설치

권장 방식: pip

pip install pycaret[full]

📚 4. Jupyter Notebook 연결 (선택)
python -m ipykernel install --user --name pycaret_env --display-name "PyCaret Env"

🧪 5. 설치 테스트
from pycaret.classification import setup, compare_models
import pandas as pd

print("PyCaret 설치 성공!")
```
