conda env create -f mercari.yml

---

conda create -n mercari python=3.11
conda activate mercari
pip install scikit-learn numpy scipy pandas xgboost hyeropt catboost
pip install https://github.com/microsoft/LightGBM/releases/download/v3.3.5/lightgbm-3.3.5-py3-none-win_amd64.whl

conda install -n mercari ipykernel --update-deps --force-reinstall

pip install gensim sentence-transformers

pip install matplotlib seaborn

pip uninstall lightgbm -y

pip install lightgbm

---

conda env export > mercari.yml

---

conda activate mercari
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
