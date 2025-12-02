import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pycaret.clustering import setup, create_model, assign_model
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


# =============================================================================
# 전처리 클래스
# =============================================================================
class TextPreprocessor:
    """텍스트 전처리를 위한 종합 클래스"""

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
        """
        Parameters:
        -----------
        remove_html : bool, HTML 태그 제거 여부
        remove_urls : bool, URL 제거 여부
        remove_emails : bool, 이메일 주소 제거 여부
        remove_numbers : bool, 숫자 제거 여부
        lowercase : bool, 소문자 변환 여부
        remove_punctuation : bool, 특수문자 제거 여부
        remove_stopwords : bool, 불용어 제거 여부
        min_word_length : int, 최소 단어 길이
        use_stemming : bool, 어간 추출 사용 여부
        use_lemmatization : bool, 표제어 추출 사용 여부
        """
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

        # 도구 초기화
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # 통계 추적
        self.stats = {
            "original_docs": 0,
            "empty_after_preprocessing": 0,
            "avg_words_before": 0,
            "avg_words_after": 0,
        }

    def _remove_html_tags(self, text):
        """HTML 태그 제거"""
        if not self.remove_html:
            return text
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")

    def _remove_urls(self, text):
        """URL 제거"""
        if not self.remove_urls:
            return text
        # http://, https://, www. 패턴 제거
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        text = re.sub(url_pattern, "", text)
        text = re.sub(r"www\.[a-zA-Z0-9]+\.[a-zA-Z]+", "", text)
        return text

    def _remove_emails(self, text):
        """이메일 주소 제거"""
        if not self.remove_emails:
            return text
        email_pattern = r"\S+@\S+"
        return re.sub(email_pattern, "", text)

    def _remove_numbers(self, text):
        """숫자 제거"""
        if not self.remove_numbers:
            return text
        return re.sub(r"\d+", "", text)

    def _remove_punctuation(self, text):
        """특수문자 제거 (단어만 남김)"""
        if not self.remove_punctuation:
            return text
        return re.sub(r"[^\w\s]", " ", text)

    def _normalize_whitespace(self, text):
        """공백 정규화"""
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize(self, text):
        """토큰화"""
        try:
            tokens = word_tokenize(text)
        except:
            # word_tokenize 실패 시 간단한 split 사용
            tokens = text.split()
        return tokens

    def _filter_tokens(self, tokens):
        """토큰 필터링 (불용어, 짧은 단어 제거)"""
        filtered = []
        for token in tokens:
            # 최소 길이 체크
            if len(token) < self.min_word_length:
                continue
            # 불용어 체크
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            filtered.append(token)
        return filtered

    def _stem_or_lemmatize(self, tokens):
        """어간 추출 또는 표제어 추출"""
        if self.use_stemming:
            return [self.stemmer.stem(token) for token in tokens]
        elif self.use_lemmatization:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens

    def preprocess_text(self, text):
        """단일 텍스트 전처리"""
        if not isinstance(text, str):
            return ""

        # 원본 단어 수 (통계용)
        original_word_count = len(text.split())

        # 1. HTML 태그 제거
        text = self._remove_html_tags(text)

        # 2. URL 제거
        text = self._remove_urls(text)

        # 3. 이메일 제거
        text = self._remove_emails(text)

        # 4. 숫자 제거
        text = self._remove_numbers(text)

        # 5. 소문자 변환
        if self.lowercase:
            text = text.lower()

        # 6. 특수문자 제거
        text = self._remove_punctuation(text)

        # 7. 공백 정규화
        text = self._normalize_whitespace(text)

        # 8. 토큰화
        tokens = self._tokenize(text)

        # 9. 토큰 필터링
        tokens = self._filter_tokens(tokens)

        # 10. 어간/표제어 추출
        tokens = self._stem_or_lemmatize(tokens)

        # 최종 텍스트
        processed_text = " ".join(tokens)

        return processed_text

    def preprocess_dataframe(self, df, text_column):
        """DataFrame 전처리"""
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

        # 전처리 적용
        df_processed = df.copy()
        df_processed["processed_text"] = df[text_column].apply(
            lambda x: self.preprocess_text(x)
        )

        # 빈 문서 확인
        empty_mask = df_processed["processed_text"].str.strip() == ""
        self.stats["empty_after_preprocessing"] = empty_mask.sum()

        # 빈 문서 제거
        df_processed = df_processed[~empty_mask].reset_index(drop=True)

        # 통계 계산
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

    def get_custom_stopwords(self, additional_words=None):
        """사용자 정의 불용어 추가"""
        if additional_words:
            self.stop_words.update(additional_words)
            print(f"✅ {len(additional_words)}개 사용자 정의 불용어 추가됨")


# =============================================================================
# 데이터 로드 함수
# =============================================================================
def load_file_data(path):
    """지정된 경로의 모든 .data 파일을 DataFrame으로 로드"""
    data_list = []
    if not path or not isinstance(path, list):
        raise ValueError("path는 파일 경로의 list여야 합니다.")

    for file_ in path:
        filename = os.path.basename(file_).split(".")[0]
        try:
            with open(file_, "r", encoding="latin1") as f:
                text_content = f.read()
            data_list.append({"filename": filename, "opinion_text": text_content})
        except Exception as e:
            print(f"❌ Error reading {file_}: {e}")

    return pd.DataFrame(data_list)


# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == "__main__":
    # 경로 설정
    DATA_PATH = "../data"
    OUTPUT_DIR = "../results"
    IMAGE_DIR = "../images"

    # 데이터 로드
    all_files = glob.glob(os.path.join(DATA_PATH, "*.data"))
    if not all_files:
        raise FileNotFoundError(f"No .data files found in {DATA_PATH}")

    print(f"\n{'='*60}")
    print(f"📂 {len(all_files)}개 파일 발견")
    document_df = load_file_data(all_files)
    print(f"✅ {len(document_df)}개 문서 로드 완료")

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
    document_df = preprocessor.preprocess_dataframe(document_df, "opinion_text")

    # TF-IDF 벡터화
    print(f"\n{'='*60}")
    print("🔤 TF-IDF 벡터화 중...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,  # 최소 2개 문서에 등장
        max_df=0.8,  # 80% 이상 문서에 등장하는 단어 제외
    )
    X = vectorizer.fit_transform(document_df["processed_text"])
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    print(f"✅ TF-IDF 행렬 생성 완료: {tfidf_df.shape}")

    # 타임스탬프 및 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 전처리된 데이터 저장
    processed_path = os.path.join(OUTPUT_DIR, f"preprocessed_data_{timestamp}.csv")
    document_df.to_csv(processed_path, index=False, encoding="utf-8")
    print(f"💾 전처리 데이터 저장: {processed_path}")

    # TF-IDF 저장
    tfidf_path = os.path.join(OUTPUT_DIR, f"tfidf_matrix_{timestamp}.csv")
    tfidf_df.to_csv(tfidf_path, index=False)
    print(f"💾 TF-IDF 행렬 저장: {tfidf_path}")

    # 클러스터링
    print(f"\n{'='*60}")
    print("🤖 클러스터링 모델 생성 중...")
    s = setup(data=tfidf_df, session_id=42, verbose=False)

    kmeans = create_model("kmeans", verbose=False)
    dbscan = create_model("dbscan", verbose=False)
    birch = create_model("birch", verbose=False)

    # Silhouette Score 계산
    def get_silhouette_score(model, tfidf_df):
        clustered = assign_model(model)
        labels = clustered["Cluster"]
        unique_labels = set(labels)

        if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
            return None

        valid_mask = labels != -1
        if valid_mask.sum() > 0:
            return silhouette_score(tfidf_df[valid_mask], labels[valid_mask])
        return None

    print("\n📊 Silhouette Score 계산 중...")
    kmeans_score = get_silhouette_score(kmeans, tfidf_df)
    dbscan_score = get_silhouette_score(dbscan, tfidf_df)
    birch_score = get_silhouette_score(birch, tfidf_df)

    scores = {"KMeans": kmeans_score, "DBSCAN": dbscan_score, "Birch": birch_score}

    # 결과 저장
    results_path = os.path.join(OUTPUT_DIR, f"results_{timestamp}.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("텍스트 클러스터링 결과 보고서\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"원본 문서 수: {preprocessor.stats['original_docs']}\n")
        f.write(f"전처리 후 문서 수: {len(document_df)}\n")
        f.write(f"TF-IDF 특성 수: {tfidf_df.shape[1]}\n\n")

        f.write("=== 전처리 설정 ===\n")
        f.write(f"HTML 제거: {preprocessor.remove_html}\n")
        f.write(f"URL 제거: {preprocessor.remove_urls}\n")
        f.write(f"숫자 제거: {preprocessor.remove_numbers}\n")
        f.write(f"불용어 제거: {preprocessor.remove_stopwords}\n")
        f.write(f"Lemmatization: {preprocessor.use_lemmatization}\n\n")

        f.write("=== Silhouette Score 비교 ===\n")
        for model, score in scores.items():
            score_str = f"{score:.4f}" if score is not None else "N/A"
            f.write(f"{model}: {score_str}\n")

    print(f"\n{'='*60}")
    print(f"✨ 분석 완료!")
    print(f"📁 결과 파일: {results_path}")
    print("=" * 60)
