"""Backend: load the saved model and score a review."""
# pyrefly: ignore [missing-import]
import joblib

from src import config
from src.explain import top_contributions

_model = None


def load_model():
    global _model
    if _model is None:
        if not config.BEST_MODEL_PATH.exists():
            raise FileNotFoundError("models/best_model.pkl not found. Run `python -m src.train` first.")
        _model = joblib.load(config.BEST_MODEL_PATH)
    return _model


def predict(text: str) -> dict:
    model = load_model()
    p_fake = float(model.predict_proba([text])[0, 1])
    is_fake = p_fake >= 0.5
    fake_words, genuine_words = top_contributions(model, text)
    return {
        "label": config.LABEL_NAMES[int(is_fake)],
        "is_fake": is_fake,
        "confidence": p_fake if is_fake else 1 - p_fake,
        "p_fake": p_fake,
        "fake_words": fake_words,
        "genuine_words": genuine_words,
    }
