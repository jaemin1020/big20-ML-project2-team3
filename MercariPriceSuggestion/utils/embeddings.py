# utils/embeddings.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from gensim.models import FastText


# ---------------------------------------------------------------
# Base embedding interface
# ---------------------------------------------------------------
class BaseEmbedding:
    is_sparse = False

    def fit_transform(self, texts):
        raise NotImplementedError

    def transform(self, texts):
        raise NotImplementedError


# ---------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------
class TfidfEmbedding(BaseEmbedding):
    def __init__(self, max_features=30000, stop_words="english"):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
        )
        self.is_sparse = True

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)


# ---------------------------------------------------------------
# Word2Vec (gensim)
# ---------------------------------------------------------------
class Word2VecEmbedding(BaseEmbedding):
    def __init__(self, vector_size=300, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.is_sparse = False

    def fit_transform(self, texts):
        tokenized = [t.split() for t in texts]
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )
        return np.vstack([self._sent_vec(t) for t in tokenized])

    def transform(self, texts):
        tokenized = [t.split() for t in texts]
        return np.vstack([self._sent_vec(t) for t in tokenized])

    def _sent_vec(self, tokens):
        vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
        if len(vectors) == 0:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)


# ---------------------------------------------------------------
# Sentence-BERT
# ---------------------------------------------------------------
class BertEmbedding(BaseEmbedding):
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.model = SentenceTransformer(model_name, device=device)
        self.is_sparse = False

    def fit_transform(self, texts):
        return np.array(self.model.encode(texts, show_progress_bar=True))

    def transform(self, texts):
        return np.array(self.model.encode(texts, show_progress_bar=True))
# ---------------------------------------------------------------
# FastText
# ---------------------------------------------------------------
class FastTextEmbedding(BaseEmbedding):
    def __init__(self, vector_size=300, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.is_sparse = False  # Dense Embedding

    def fit_transform(self, texts):
        tokenized = [t.split() for t in texts]

        self.model = FastText(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )

        return np.vstack([self._sent_vec(tokens) for tokens in tokenized])

    def transform(self, texts):
        tokenized = [t.split() for t in texts]
        return np.vstack([self._sent_vec(tokens) for tokens in tokenized])

    def _sent_vec(self, tokens):
        vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
        if len(vectors) == 0:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)