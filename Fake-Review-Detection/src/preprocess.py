"""Light text cleaning.

We deliberately keep this SIMPLE. TF-IDF already down-weights very common
words, and stemming/stopword removal can throw away useful style signals
(e.g. "very", "not"). Experiment with them if you wish (see notebook 02).
"""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_NON_ALNUM = re.compile(r"[^a-z0-9\s']")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """lowercase -> strip urls/html -> keep letters/digits/apostrophes -> squeeze spaces."""
    if text is None:
        return ""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return text
