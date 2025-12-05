🛠 단계별 CUDA 환경 구축 (Windows 기준)

1. NVIDIA 드라이버 설치

* [NVIDIA 공식 드라이버 다운로드 페이지](https://www.nvidia.com/Download/index.aspx)에서 GPU 모델에 맞는 최신 드라이버 설치.
* 설치 후 재부팅.


3. Python 환경 준비

* 가상환경 생성 후 PyTorch 또는 TensorFlow 설치:

```conda
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
```

6. GPU 인식 확인

* Python에서 다음 코드 실행:

```python
import torch
print(torch.cuda.is_available())  # True면 정상
print(torch.cuda.get_device_name(0))  # GPU 이름 출력
```

7. AutoGluon 실행

* AutoGluon 설치:

pip install autogluon

* 모델 학습 시 GPU 사용:

```
predictor = TabularPredictor(label=TARGET_COL).fit(
    train_data=train_df,
    num_gpus=1  # GPU 1개 사용
)
```

✅ 핵심 체크리스트

* NVIDIA GPU 드라이버 최신 버전 설치 완료
* CUDA Toolkit과 cuDNN 버전 호환 확인
* Python에서 `torch.cuda.is_available()`가 True인지 확인

👉 이렇게 하면 AMD CPU + NVIDIA GPU 환경에서도 CUDA를 정상적으로 구축해 AutoGluon 모델을 GPU 가속으로 학습할 수 있습니다.
