"""Central settings so every script/notebook uses the same paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "reviews.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DB_PATH = ROOT / "database" / "predictions.db"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Kaggle "Fake Reviews Dataset" (Salminen et al.) uses these raw column names.
RAW_TEXT_COL = "text_"
RAW_LABEL_COL = "label"      # CG = computer-generated (fake), OR = original (genuine)

# Names we use internally after loading
TEXT_COL = "text"
TARGET_COL = "is_fake"       # 1 = fake, 0 = genuine
LABEL_NAMES = {0: "Genuine Review", 1: "Fake Review"}

for d in (MODELS_DIR, RESULTS_DIR, DB_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)
