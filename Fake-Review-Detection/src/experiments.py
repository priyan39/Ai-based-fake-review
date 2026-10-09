"""Applied-scientist experiments. Results go to results/experiments.csv.

    python -m src.experiments

Exp 1: TF-IDF configuration (unigram vs unigram+bigram, stopwords, max_features)
Exp 2: Logistic Regression vs Random Forest (same split)  -> see train.py
Exp 3: Class imbalance handling (none / class_weight / downsample)
"""
import pandas as pd

from src import config
from src.data import load_raw, prepare
from src.models import make_logreg, make_tfidf, metrics, split


def run_exp1(X_train, X_test, y_train, y_test):
    configs = {
        "unigram": dict(ngram_range=(1, 1)),
        "unigram+bigram": dict(ngram_range=(1, 2)),
        "unigram, stopwords removed": dict(ngram_range=(1, 1), stop_words="english"),
        "unigram+bigram, max_features=5000": dict(ngram_range=(1, 2), max_features=5000),
    }
    rows = []
    for name, kw in configs.items():
        pipe = make_logreg(tfidf=make_tfidf(**kw))
        pipe.fit(X_train, y_train)
        rows.append({"experiment": "Exp1 TF-IDF", "setting": name,
                     **metrics(y_test, pipe.predict(X_test))})
    return rows


def run_exp3(X_train, X_test, y_train, y_test):
    rows = []
    counts = y_train.value_counts()
    print("Train class counts:", counts.to_dict())

    pipe = make_logreg(); pipe.fit(X_train, y_train)
    rows.append({"experiment": "Exp3 Imbalance", "setting": "no handling",
                 **metrics(y_test, pipe.predict(X_test))})

    pipe = make_logreg(class_weight="balanced"); pipe.fit(X_train, y_train)
    rows.append({"experiment": "Exp3 Imbalance", "setting": "class_weight=balanced",
                 **metrics(y_test, pipe.predict(X_test))})

    # Downsample majority class in TRAIN only (test stays untouched)
    n = counts.min()
    idx = pd.concat([
        y_train[y_train == c].sample(n, random_state=config.RANDOM_STATE) for c in counts.index
    ]).index
    pipe = make_logreg(); pipe.fit(X_train.loc[idx], y_train.loc[idx])
    rows.append({"experiment": "Exp3 Imbalance", "setting": "downsample majority",
                 **metrics(y_test, pipe.predict(X_test))})
    return rows


def main():
    df = prepare(load_raw())
    X_train, X_test, y_train, y_test = split(df)
    rows = run_exp1(X_train, X_test, y_train, y_test) + run_exp3(X_train, X_test, y_train, y_test)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(config.RESULTS_DIR / "experiments.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
