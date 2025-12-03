import os, glob, re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datetime import datetime
import warnings

# NLP 라이브러리
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

# 필요한 NLTK 데이터 다운로드
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

warnings.filterwarnings("ignore")


# 경로 설정
DATA_PATH = '../data'
OUTPUT_DIR = '../results'
IMAGE_DIR = '../images'

# =============================================================================
# 데이터 로드
# =============================================================================
def load_file_data():
    """지정된 경로의 모든 .data 파일을 DataFrame으로 로드"""
    # 데이터 로드
    all_files = glob.glob(os.path.join(DATA_PATH, "*.data"))
    if not all_files:
        raise FileNotFoundError(f"No .data files found in {DATA_PATH}")

    print(f"📂 {len(all_files)}개 파일 발견")

    data_list = []
    if not all_files or not isinstance(all_files, list):
        raise ValueError("path는 파일 경로의 list여야 합니다.")

    for file_ in all_files:
        filename = os.path.basename(file_).split('.')[0]
        try:
            with open(file_, 'r', encoding='latin1') as f:
                text_content = f.read()
            data_list.append({'filename': filename, 'opinion_text': text_content})
        except Exception as e:
            print(f"❌ Error reading {file_}: {e}")

    return pd.DataFrame(data_list)
# eof : End of Function --------------------------------------------------------- #

# =============================================================================
# 전처리 클래스
# =============================================================================
class TextPreprocessor:
    """텍스트 전처리를 위한 종합 클래스"""

    # __init__ ##############
    def __init__(
        self,
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        remove_numbers=True,
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        min_word_length=2,
        use_stemming=False,
        use_lemmatization=True,
    ):
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.stats = {
            "original_docs": 0,
            "empty_after_preprocessing": 0,
            "avg_words_before": 0,
            "avg_words_after": 0,
        }
    # eof -------------------------------------- #

    # _remove_html_tags ##############
    def _remove_html_tags(self, text):
        if not self.remove_html:
            return text
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")
    # eof -------------------------------------- #

    # _remove_urls ##############
    def _remove_urls(self, text):
        if not self.remove_urls:
            return text
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        text = re.sub(url_pattern, "", text)
        text = re.sub(r"www\.[a-zA-Z0-9]+\.[a-zA-Z]+", "", text)
        return text
    # eof -------------------------------------- #

    # _remove_emails ##############
    def _remove_emails(self, text):
        if not self.remove_emails:
            return text
        email_pattern = r"\S+@\S+"
        return re.sub(email_pattern, "", text)
    # eof -------------------------------------- #

    # _remove_numbers ##############
    def _remove_numbers(self, text):
        if not self.remove_numbers:
            return text
        return re.sub(r"\d+", "", text)
    # eof -------------------------------------- #

    # _remove_punctuation ##############
    def _remove_punctuation(self, text):
        if not self.remove_punctuation:
            return text
        return re.sub(r"[^\w\s]", " ", text)
    # eof -------------------------------------- #

    # _normalize_whitespace ##############
    def _normalize_whitespace(self, text):
        return re.sub(r"\s+", " ", text).strip()
    # eof -------------------------------------- #

    # _tokenize ##############
    def _tokenize(self, text):
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        return tokens
    # eof -------------------------------------- #

    # _filter_tokens ##############
    def _filter_tokens(self, tokens):
        filtered = []
        for token in tokens:
            if len(token) < self.min_word_length:
                continue
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            filtered.append(token)
        return filtered
    # eof -------------------------------------- #

    # _stem_or_lemmatize ##############
    def _stem_or_lemmatize(self, tokens):
        if self.use_stemming:
            return [self.stemmer.stem(token) for token in tokens]
        elif self.use_lemmatization:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    # eof -------------------------------------- #

    # preprocess_text ##############
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""

        original_word_count = len(text.split())
        text = self._remove_html_tags(text)
        text = self._remove_urls(text)
        text = self._remove_emails(text)
        text = self._remove_numbers(text)

        if self.lowercase:
            text = text.lower()

        text = self._remove_punctuation(text)
        text = self._normalize_whitespace(text)
        tokens = self._tokenize(text)
        tokens = self._filter_tokens(tokens)
        tokens = self._stem_or_lemmatize(tokens)

        processed_text = " ".join(tokens)
        return processed_text
    # eof -------------------------------------- #

    # preprocess_dataframe ##############
    def preprocess_dataframe(self, df, text_column):
        print("🔄 텍스트 전처리 시작...")
        print(
            f"   옵션: HTML제거={self.remove_html}, URL제거={self.remove_urls}, "
            f"숫자제거={self.remove_numbers}"
        )
        print(
            f"   옵션: 불용어제거={self.remove_stopwords}, "
            f"Lemmatization={self.use_lemmatization}, Stemming={self.use_stemming}"
        )

        self.stats["original_docs"] = len(df)

        df_processed = df.copy()
        df_processed["processed_text"] = df[text_column].apply(
            lambda x: self.preprocess_text(x)
        )

        empty_mask = df_processed["processed_text"].str.strip() == ""
        self.stats["empty_after_preprocessing"] = empty_mask.sum()
        df_processed = df_processed[~empty_mask].reset_index(drop=True)

        if len(df_processed) > 0:
            df_processed["word_count"] = df_processed["processed_text"].apply(
                lambda x: len(x.split())
            )
            self.stats["avg_words_after"] = df_processed["word_count"].mean()

        print(f"✅ 전처리 완료:")
        print(f"   - 원본 문서 수: {self.stats['original_docs']}")
        print(f"   - 제거된 빈 문서: {self.stats['empty_after_preprocessing']}")
        print(f"   - 최종 문서 수: {len(df_processed)}")
        if len(df_processed) > 0:
            print(f"   - 평균 단어 수: {self.stats['avg_words_after']:.1f}")

        return df_processed
    # eof -------------------------------------- #

    # get_custom_stopwords ##############
    def get_custom_stopwords(self, additional_words=None):
        if additional_words:
            self.stop_words.update(additional_words)
            print(f"✅ {len(additional_words)}개 사용자 정의 불용어 추가됨")
    # eof -------------------------------------- #

# eof -----------------------------------------------------------------------#

# data load and default PreProcessing #########################################
def get_default_data():
    """
    텍스트 데이터 로드 및 기본 전처리를 수행하는 함수.

    이 함수는 지정된 경로에서 텍스트 파일들을 로드한 후,
    TextPreprocessor 클래스를 사용하여 HTML, URL, 이메일, 숫자, 특수문자 제거,
    소문자 변환, 불용어 제거, 표제어 추출 등의 기본 전처리를 수행합니다.

    Returns:
        pd.DataFrame: 전처리된 텍스트를 포함한 DataFrame (컬럼: 'filename', 'processed_text')
    """

    # 데이터 로드
    document_df = load_file_data()

    # 전처리 설정 및 실행
    preprocessor = TextPreprocessor(
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        remove_numbers=True,
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        min_word_length=2,
        use_stemming=False,
        use_lemmatization=True,
    )

    # 도메인 특화 불용어 추가 (필요시)
    # preprocessor.get_custom_stopwords(['custom', 'words', 'here'])

    # 전처리 실행
    return preprocessor.preprocess_dataframe(document_df, "opinion_text")
# eof --------------------------------------------------------------------------------#