"""Quick sanity tests on a tiny made-up sample (NOT real data, no accuracy claims)."""
import pandas as pd

from src import config
from src.data import prepare
from src.models import make_logreg, split
from src.preprocess import clean_text


def test_clean_text():
    assert clean_text("  GREAT!!!  http://x.com <b>Buy</b> ") == "great buy"
    assert clean_text(None) == ""


def test_prepare_and_fit():
    good = [f"works well after {i} days of use, battery fine" for i in range(30)]
    bad = [f"amazing perfect product highly recommend {i} five stars" for i in range(30)]
    raw = pd.DataFrame({"text_": good + bad + [None], "label": ["OR"] * 30 + ["CG"] * 30 + ["OR"]})
    df = prepare(raw)
    assert len(df) == 60 and set(df[config.TARGET_COL]) == {0, 1}
    X_tr, X_te, y_tr, y_te = split(df)
    pipe = make_logreg().fit(X_tr, y_tr)
    assert pipe.predict_proba(X_te).shape[1] == 2
