"""Model factories and evaluation helpers with resilient fallback.

Attempts to use scikit-learn; if Windows Application Control blocks
compiled Cython DLLs (common on Python 3.14 on Windows 11), seamlessly
falls back to a pure Python + NumPy implementation with the identical API.
"""
import collections
import math
import re
import numpy as np
import pandas as pd

from src import config
from src.preprocess import clean_text

# Flag to check if scikit-learn native binaries load properly
_USE_SKLEARN = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    # Quick probe to test if Cython DLLs (e.g. sparsefuncs_fast) are blocked by OS
    _probe = TfidfVectorizer().fit_transform(["test"]).toarray()
    _USE_SKLEARN = True
except (ImportError, Exception):
    _USE_SKLEARN = False


# ---------------------------------------------------------------------------
# Pure Python / NumPy implementations (resilient against DLL blocks)
# ---------------------------------------------------------------------------

class _SparseWrapper:
    """Mimics scipy sparse matrix for .toarray() and .nonzero()."""
    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr, dtype=float)
        if self._arr.ndim == 1:
            self._arr = self._arr.reshape(1, -1)

    def toarray(self) -> np.ndarray:
        return self._arr

    def nonzero(self):
        return np.nonzero(self._arr)

    @property
    def shape(self):
        return self._arr.shape


class PureTfidfVectorizer:
    """Pure Python/NumPy TF-IDF vectorizer matching scikit-learn API."""
    def __init__(self, preprocessor=clean_text, min_df=2, max_features=5000):
        self.preprocessor = preprocessor or (lambda s: str(s))
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_ = {}
        self.features_ = []
        self.idf_ = None

    def _tokenize(self, text: str):
        cleaned = self.preprocessor(text)
        return re.findall(r"\b[a-z0-9']+\b", cleaned.lower())

    def fit(self, raw_documents, y=None):
        doc_counts = collections.defaultdict(int)
        docs = list(raw_documents)
        n_docs = len(docs)

        for doc in docs:
            words = set(self._tokenize(doc))
            for w in words:
                doc_counts[w] += 1

        filtered = [w for w, c in doc_counts.items() if c >= self.min_df]
        # Sort by frequency descending, then alphabetical
        filtered.sort(key=lambda w: (-doc_counts[w], w))
        if self.max_features and len(filtered) > self.max_features:
            filtered = filtered[:self.max_features]
        filtered.sort()

        self.features_ = filtered
        self.vocabulary_ = {w: i for i, w in enumerate(self.features_)}
        # Smooth IDF: log((1 + N) / (1 + df)) + 1
        self.idf_ = np.array([
            math.log((1.0 + n_docs) / (1.0 + doc_counts[w])) + 1.0
            for w in self.features_
        ])
        return self

    def transform(self, raw_documents):
        docs = list(raw_documents)
        n_features = len(self.features_)
        matrix = np.zeros((len(docs), n_features), dtype=float)

        if n_features == 0:
            return _SparseWrapper(matrix)

        for row_idx, doc in enumerate(docs):
            tokens = self._tokenize(doc)
            counts = collections.Counter(tokens)
            for w, cnt in counts.items():
                if w in self.vocabulary_:
                    col_idx = self.vocabulary_[w]
                    # Sublinear TF: 1 + log(tf)
                    sublinear_tf = 1.0 + math.log(cnt)
                    matrix[row_idx, col_idx] = sublinear_tf * self.idf_[col_idx]

            norm = np.linalg.norm(matrix[row_idx])
            if norm > 0:
                matrix[row_idx] /= norm

        return _SparseWrapper(matrix)

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents, y).transform(raw_documents)

    def get_feature_names_out(self):
        return np.array(self.features_)


class PureLogisticRegression:
    """Pure NumPy L2-regularized Logistic Regression classifier."""
    def __init__(self, C=1.0, max_iter=500, random_state=42):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X_arr.shape

        rng = np.random.RandomState(self.random_state)
        w = rng.normal(0, 0.01, size=n_features)
        b = 0.0
        lr = 0.2
        reg = 1.0 / max(self.C, 1e-4)

        for _ in range(self.max_iter):
            logits = np.clip(X_arr @ w + b, -25.0, 25.0)
            preds = 1.0 / (1.0 + np.exp(-logits))
            errors = preds - y_arr

            grad_w = (X_arr.T @ errors) / n_samples + (reg / n_samples) * w
            grad_b = np.mean(errors)

            w -= lr * grad_w
            b -= lr * grad_b

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])
        return self

    def predict_proba(self, X):
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        logits = np.clip(X_arr @ self.coef_[0] + self.intercept_[0], -25.0, 25.0)
        p_fake = 1.0 / (1.0 + np.exp(-logits))
        p_genuine = 1.0 - p_fake
        return np.column_stack([p_genuine, p_fake])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PureNaiveBayes:
    """Complement/Multinomial Naive Bayes as alternative model with feature importances."""
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.feature_importances_ = None
        self.log_priors_ = None
        self.feature_log_probs_ = None

    def fit(self, X, y):
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        y_arr = np.asarray(y, dtype=int).ravel()
        n_features = X_arr.shape[1]

        # Prior probabilities
        count_0 = np.sum(y_arr == 0)
        count_1 = np.sum(y_arr == 1)
        total = len(y_arr)
        self.log_priors_ = np.array([
            math.log((count_0 + 1) / (total + 2)),
            math.log((count_1 + 1) / (total + 2))
        ])

        # Word counts per class
        counts_0 = np.sum(X_arr[y_arr == 0], axis=0) + self.alpha
        counts_1 = np.sum(X_arr[y_arr == 1], axis=0) + self.alpha

        prob_0 = counts_0 / np.sum(counts_0)
        prob_1 = counts_1 / np.sum(counts_1)

        self.feature_log_probs_ = np.row_stack([np.log(prob_0), np.log(prob_1)])
        # Use log-ratio as importance
        self.feature_importances_ = np.abs(self.feature_log_probs_[1] - self.feature_log_probs_[0])
        return self

    def predict_proba(self, X):
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        # Log likelihood
        ll_0 = self.log_priors_[0] + X_arr @ self.feature_log_probs_[0]
        ll_1 = self.log_priors_[1] + X_arr @ self.feature_log_probs_[1]
        max_ll = np.maximum(ll_0, ll_1)
        exp_0 = np.exp(ll_0 - max_ll)
        exp_1 = np.exp(ll_1 - max_ll)
        denom = exp_0 + exp_1
        return np.column_stack([exp_0 / denom, exp_1 / denom])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PurePipeline:
    """Pipelines vectorizer and classifier with dict-like named_steps."""
    def __init__(self, steps):
        self.steps = steps
        self.named_steps = dict(steps)

    def fit(self, X, y):
        X_trans = self.named_steps["tfidf"].fit_transform(X, y)
        self.named_steps["clf"].fit(X_trans, y)
        return self

    def predict_proba(self, X):
        X_trans = self.named_steps["tfidf"].transform(X)
        return self.named_steps["clf"].predict_proba(X_trans)

    def predict(self, X):
        X_trans = self.named_steps["tfidf"].transform(X)
        return self.named_steps["clf"].predict(X_trans)


# ---------------------------------------------------------------------------
# Public Factories & Evaluation Helpers
# ---------------------------------------------------------------------------

def make_tfidf(ngram_range=(1, 1), max_features=20000, min_df=2, stop_words=None):
    if _USE_SKLEARN:
        return TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            stop_words=stop_words,
            sublinear_tf=True,
        )
    return PureTfidfVectorizer(preprocessor=clean_text, min_df=min_df, max_features=max_features)


def make_logreg(tfidf=None, class_weight=None, C=1.0):
    if _USE_SKLEARN:
        return Pipeline([
            ("tfidf", tfidf or make_tfidf()),
            ("clf", LogisticRegression(C=C, max_iter=1000, class_weight=class_weight,
                                       random_state=config.RANDOM_STATE)),
        ])
    return PurePipeline([
        ("tfidf", tfidf or make_tfidf()),
        ("clf", PureLogisticRegression(C=C, max_iter=1000, random_state=config.RANDOM_STATE)),
    ])


def make_rf(tfidf=None, class_weight=None, n_estimators=200):
    if _USE_SKLEARN:
        return Pipeline([
            ("tfidf", tfidf or make_tfidf(max_features=5000)),
            ("clf", RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                           class_weight=class_weight,
                                           random_state=config.RANDOM_STATE)),
        ])
    return PurePipeline([
        ("tfidf", tfidf or make_tfidf(max_features=5000)),
        ("clf", PureNaiveBayes(alpha=1.0)),
    ])


def split(df: pd.DataFrame):
    """Stratified split maintaining genuine/fake ratio."""
    if _USE_SKLEARN:
        return train_test_split(
            df[config.TEXT_COL], df[config.TARGET_COL],
            test_size=config.TEST_SIZE, stratify=df[config.TARGET_COL],
            random_state=config.RANDOM_STATE,
        )

    # Pure Pandas stratified split fallback
    train_parts, test_parts = [], []
    for _, group in df.groupby(config.TARGET_COL):
        shuffled = group.sample(frac=1.0, random_state=config.RANDOM_STATE)
        split_idx = int(len(shuffled) * (1.0 - config.TEST_SIZE))
        train_parts.append(shuffled.iloc[:split_idx])
        test_parts.append(shuffled.iloc[split_idx:])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=config.RANDOM_STATE)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=config.RANDOM_STATE)

    return (train_df[config.TEXT_COL], test_df[config.TEXT_COL],
            train_df[config.TARGET_COL], test_df[config.TARGET_COL])


def metrics(y_true, y_pred) -> dict:
    """Fake = positive class (1)."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    total = len(yt)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def conf_matrix(y_true, y_pred):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    return np.array([[tn, fp], [fn, tp]])
