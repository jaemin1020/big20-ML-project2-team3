# Python 팀 프로젝트 코딩 규칙

## 1. 기본 스타일 가이드 (PEP 8 기반)

### 1.1 들여쓰기 및 공백 : 

```python
# ✅ 좋은 예
def calculate_average(data):
    total = sum(data)
    count = len(data)
    return total / count

# ❌ 나쁜 예
def calculate_average(data):
  total=sum(data)
  count=len(data)
  return total/count
```

**규칙:**

- **들여쓰기**: 스페이스 4칸 (탭 사용 금지)
- **연산자 주변**: 공백 1칸 (`x = 1`, `a + b`)
- **콤마 뒤**: 공백 1칸 (`[1, 2, 3]`)
- **함수 정의 사이**: 빈 줄 2개
- **클래스 메서드 사이**: 빈 줄 1개

### 1.2 한 줄 길이

```python
# ✅ 좋은 예 (79자 이내)
result = calculate_statistics(
    data=user_data,
    method='average',
    include_outliers=False
)

# ❌ 나쁜 예 (너무 긴 한 줄)
result = calculate_statistics(data=user_data, method='average', include_outliers=False, normalize=True, remove_duplicates=True)
```

**규칙:** 한 줄 최대 **79자** (주석/docstring은 72자)

------

## 2. 명명 규칙 (Naming Conventions)

### 2.1 변수 및 함수명

```python
# ✅ 좋은 예
user_name = "홍길동"
total_sales = 1000
customer_count = 50

def calculate_total_price(items):
    pass

def get_user_by_id(user_id):
    pass
```

**규칙:**

- **snake_case** 사용 (소문자 + 언더스코어)
- **의미 있는 이름** 사용 (약어 지양)
- **동사 + 명사** 조합 권장 (`get_data`, `calculate_sum`)

### 2.2 클래스명

```python
# ✅ 좋은 예
class UserManager:
    pass

class DataProcessor:
    pass

class OrderService:
    pass
```

**규칙:** **PascalCase** (각 단어 첫 글자 대문자)

### 2.3 상수

```python
# ✅ 좋은 예
MAX_CONNECTIONS = 100
DATABASE_URL = "localhost:5432"
API_TIMEOUT = 30
```

**규칙:** **UPPER_SNAKE_CASE** (대문자 + 언더스코어)

### 2.4 Private 변수/메서드

```python
class User:
    def __init__(self):
        self._internal_id = 123  # protected
        self.__password = "secret"  # private
    
    def _validate_data(self):  # protected method
        pass
```

**규칙:**

- **Protected**: 앞에 `_` 한 개
- **Private**: 앞에 `__` 두 개

------

## 3. 주석 및 문서화

### 3.1 함수 Docstring

```python
def calculate_discount(price, discount_rate):
    """
    할인된 가격을 계산합니다.
    
    Args:
        price (float): 원래 가격
        discount_rate (float): 할인율 (0.0 ~ 1.0)
    
    Returns:
        float: 할인된 가격
    
    Example:
        >>> calculate_discount(10000, 0.2)
        8000.0
    """
    return price * (1 - discount_rate)
```

**규칙:**

- 모든 public 함수에 docstring 작성
- Google Style 또는 NumPy Style 통일
- 파라미터, 반환값, 예외 명시

### 3.2 인라인 주석

```python
# ✅ 좋은 예
x = x + 1  # 경계값 보정

# ❌ 나쁜 예
x = x + 1  # x에 1을 더함 (당연한 내용)
```

**규칙:**

- **왜(Why)**를 설명 (무엇(What)이 아님)
- 복잡한 로직에만 사용
- 주석 앞에 공백 2칸

------

## 4. Import 규칙

```python
# ✅ 좋은 예
# 1. 표준 라이브러리
import os
import sys
from datetime import datetime

# 2. 서드파티 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3. 로컬 모듈
from src.utils import helper
from src.models import User

# ❌ 나쁜 예
import os, sys  # 콤마로 한 줄에
from datetime import *  # wildcard import
```

**규칙:**

- **그룹별 순서**: 표준 라이브러리 → 서드파티 → 로컬
- 각 그룹 사이 **빈 줄 1개**
- `import *` 금지
- 한 줄에 하나의 import

------

## 5. 데이터 분석 프로젝트 특화 규칙

### 5.1 DataFrame 변수명

```python
# ✅ 좋은 예
df_sales = pd.read_csv('sales.csv')
df_users = pd.read_csv('users.csv')
df_merged = df_sales.merge(df_users)

# ❌ 나쁜 예
data1 = pd.read_csv('sales.csv')
temp = pd.read_csv('users.csv')
result = data1.merge(temp)
```

**규칙:** `df_` 접두사 + 의미있는 이름

### 5.2 메서드 체이닝

```python
# ✅ 좋은 예 (읽기 쉬움)
result = (
    df.query('age > 20')
    .groupby('city')['sales']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# ❌ 나쁜 예 (한 줄로 길게)
result = df.query('age > 20').groupby('city')['sales'].mean().sort_values(ascending=False).head(10)
```

**규칙:**

- 각 메서드를 **새 줄**에 작성
- 괄호로 감싸기
- 적절한 들여쓰기

### 5.3 시각화 코드

```python
# ✅ 좋은 예
fig, ax = plt.subplots(figsize=(10, 6))

df_sales.plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)

ax.set_title('월별 매출 현황', fontsize=14, fontweight='bold')
ax.set_xlabel('월', fontsize=12)
ax.set_ylabel('매출액 (만원)', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**규칙:**

- `fig, ax` 명시적 사용
- 제목, 축 레이블 항상 작성
- `tight_layout()` 사용

------

## 6. 에러 처리

```python
# ✅ 좋은 예
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
    df = pd.DataFrame()
except pd.errors.EmptyDataError:
    print("빈 파일입니다.")
    df = pd.DataFrame()
except Exception as e:
    print(f"예상치 못한 에러: {e}")
    raise

# ❌ 나쁜 예
try:
    df = pd.read_csv('data.csv')
except:  # 너무 광범위
    pass  # 에러 무시
```

**규칙:**

- **구체적인 예외** 처리
- `except:` 단독 사용 금지
- 에러 로깅 또는 적절한 처리

------

## 7. 함수 작성 원칙

### 7.1 함수 길이

```python
# ✅ 좋은 예 (한 가지 일만)
def calculate_total(items):
    """총합 계산"""
    return sum(item['price'] for item in items)

def apply_discount(total, rate):
    """할인 적용"""
    return total * (1 - rate)

# ❌ 나쁜 예 (여러 일을 함)
def process_order(items, discount_rate, tax_rate):
    total = sum(item['price'] for item in items)
    discounted = total * (1 - discount_rate)
    with_tax = discounted * (1 + tax_rate)
    # ... 더 많은 로직
    return with_tax
```

**규칙:**

- 함수는 **한 가지 일**만
- 최대 **50줄** 이내
- 복잡하면 **분리**

### 7.2 함수 인자

```python
# ✅ 좋은 예
def create_user(name, email, age=None, city=None):
    pass

# ❌ 나쁜 예 (인자 너무 많음)
def create_user(name, email, age, city, phone, address, zipcode, country, occupation):
    pass
```

**규칙:**

- 인자는 **5개 이하**
- 많으면 **딕셔너리** 또는 **클래스** 사용

------

## 8. 버전 관리 (Git) 규칙

### 8.1 커밋 메시지

```
# ✅ 좋은 예
feat: 사용자 로그인 기능 추가
fix: 데이터 중복 제거 버그 수정
docs: README에 설치 방법 추가
refactor: calculate_total 함수 최적화

# ❌ 나쁜 예
수정함
버그 고침
업데이트
```

**규칙:**

- **타입**: `feat`, `fix`, `docs`, `refactor`, `test`
- **내용**: 간결하고 명확하게
- **현재형** 사용

### 8.2 브랜치 전략

```
main (또는 master) - 배포 가능한 안정 버전
develop - 개발 중인 버전
feature/기능명 - 새 기능 개발
fix/버그명 - 버그 수정
```

------

## 9. 파일 및 폴더 구조

```
project/
├── data/
│   ├── raw/              # 원본 데이터
│   ├── processed/        # 전처리된 데이터
│   └── external/         # 외부 데이터
├── notebooks/            # Jupyter 노트북
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/                  # 소스 코드
│   ├── __init__.py
│   ├── data/            # 데이터 처리
│   ├── models/          # 모델 정의
│   ├── utils/           # 유틸리티 함수
│   └── visualization/   # 시각화 함수
├── tests/               # 테스트 코드
├── requirements.txt     # 의존성 패키지
├── .gitignore
└── README.md
```

------

## 10. 팀 협업 체크리스트

### 코드 작성 전

- [ ] 작업할 내용을 팀원과 공유
- [ ] feature 브랜치 생성
- [ ] 최신 develop 브랜치 pull

### 코드 작성 중

- [ ] 코딩 규칙 준수
- [ ] 주석 및 docstring 작성
- [ ] 의미있는 변수명 사용

### 코드 완성 후

- [ ] 코드 자체 리뷰
- [ ] 테스트 실행 (가능한 경우)
- [ ] 커밋 메시지 작성
- [ ] Pull Request 생성
- [ ] 코드 리뷰 요청

------

## 11. 추천 도구

### 11.1 코드 포맷팅

```bash
# Black (자동 포맷팅)
pip install black
black your_script.py

# isort (import 정렬)
pip install isort
isort your_script.py
```

### 11.2 코드 품질 검사

```bash
# Flake8 (PEP 8 검사)
pip install flake8
flake8 your_script.py

# Pylint (종합 검사)
pip install pylint
pylint your_script.py
```

### 11.3 IDE 설정

- **VS Code**: Python extension 설치
- **PyCharm**: 기본 PEP 8 검사 활성화
- **Jupyter**: jupyterlab-code-formatter 설치

------

## 12. 팀 규칙 커스터마이징

위 규칙을 기반으로 팀 상황에 맞게 조정하세요:

1. **필수 규칙**: 1, 2, 4, 8번
2. **권장 규칙**: 3, 5, 6, 7번
3. **선택 규칙**: 9, 10, 11번

팀 킥오프 미팅에서 규칙을 확정하고, 프로젝트 저장소의 `CONTRIBUTING.md`에 문서화하세요!