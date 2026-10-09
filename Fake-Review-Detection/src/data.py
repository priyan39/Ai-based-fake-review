"""Loading and basic cleaning of the dataset."""
import pandas as pd

from src import config


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as downloaded (no changes)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it (see data/README.md) and save it as data/reviews.csv"
        )
    return pd.read_csv(path)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, build the 0/1 target, drop missing + duplicate reviews."""
    df = df.rename(columns={config.RAW_TEXT_COL: config.TEXT_COL}).copy()
    if config.TEXT_COL not in df.columns or config.RAW_LABEL_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{config.RAW_TEXT_COL}' and '{config.RAW_LABEL_COL}'. "
            f"Found: {list(df.columns)}"
        )

    labels = df[config.RAW_LABEL_COL].astype(str).str.strip().str.upper()
    mapping = {"CG": 1, "OR": 0}
    unknown = set(labels.unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Unexpected label values: {unknown}. Expected CG / OR.")
    df[config.TARGET_COL] = labels.map(mapping)

    df = df.dropna(subset=[config.TEXT_COL])
    df[config.TEXT_COL] = df[config.TEXT_COL].astype(str)
    df = df[df[config.TEXT_COL].str.strip() != ""]
    # Exact duplicate texts can leak between train and test, so keep one copy.
    df = df.drop_duplicates(subset=[config.TEXT_COL]).reset_index(drop=True)
    return df
