"""Explainability helpers.

For a linear model (Logistic Regression on TF-IDF) the prediction is
    logit = intercept + sum_i(coef_i * tfidf_i)
so each word's contribution (coef_i * tfidf_i) is an exact explanation of
THAT model. For Random Forest we fall back to global feature importances
(less precise: it says which words matter overall, not in this review).
"""
# pyrefly: ignore [missing-import]
import numpy as np


def top_contributions(pipeline, text: str, k: int = 6):
    """Return (fake_words, genuine_words), each a list of (word, score)."""
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    vec = tfidf.transform([text])
    names = np.array(tfidf.get_feature_names_out())
    idx = vec.nonzero()[1]
    if len(idx) == 0:
        return [], []

    if hasattr(clf, "coef_"):
        scores = vec.toarray()[0, idx] * clf.coef_[0, idx]
    else:  # tree ensemble: global importance, signed by nothing -> all "influential"
        scores = clf.feature_importances_[idx]

    order = np.argsort(scores)
    words = names[idx][order]
    vals = scores[order]

    if hasattr(clf, "coef_"):
        fake = [(w, float(v)) for w, v in zip(words[::-1], vals[::-1]) if v > 0][:k]
        genuine = [(w, float(v)) for w, v in zip(words, vals) if v < 0][:k]
    else:
        fake = [(w, float(v)) for w, v in zip(words[::-1], vals[::-1])][:k]
        genuine = []
    return fake, genuine


def global_top_features(pipeline, k: int = 20):
    """Most fake-leaning and most genuine-leaning words (Logistic Regression only)."""
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise ValueError("global_top_features needs a linear model.")
    names = np.array(tfidf.get_feature_names_out())
    coef = clf.coef_[0]
    order = np.argsort(coef)
    return (list(zip(names[order[::-1][:k]], coef[order[::-1][:k]])),
            list(zip(names[order[:k]], coef[order[:k]])))
