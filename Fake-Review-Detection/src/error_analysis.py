"""Error analysis on the saved test predictions.

    python -m src.error_analysis
"""
import pandas as pd

from src import config


def load_predictions() -> pd.DataFrame:
    path = config.RESULTS_DIR / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.train` first.")
    df = pd.read_csv(path)
    df["n_words"] = df["text"].astype(str).str.split().str.len()
    df["outcome"] = "TN (correct genuine)"
    df.loc[(df.actual == 1) & (df.pred == 1), "outcome"] = "TP (correct fake)"
    df.loc[(df.actual == 0) & (df.pred == 1), "outcome"] = "FP (genuine flagged fake)"
    df.loc[(df.actual == 1) & (df.pred == 0), "outcome"] = "FN (fake missed)"
    return df


def summarize(df: pd.DataFrame):
    print(df["outcome"].value_counts(), "\n")
    print("Average review length (words) by outcome:")
    print(df.groupby("outcome")["n_words"].mean().round(1), "\n")

    df["length_bucket"] = pd.cut(df["n_words"], [0, 10, 30, 60, 10_000],
                                 labels=["<=10", "11-30", "31-60", ">60"])
    df["error"] = (df.actual != df.pred).astype(int)
    print("Error rate by review length:")
    print(df.groupby("length_bucket", observed=True)["error"].agg(["mean", "count"]).round(3))


def examples(df: pd.DataFrame, outcome_prefix: str, n=10, most_confident=True):
    sub = df[df["outcome"].str.startswith(outcome_prefix)].copy()
    sub["confidence"] = sub["prob_fake"].where(sub.pred == 1, 1 - sub["prob_fake"])
    return sub.sort_values("confidence", ascending=not most_confident).head(n)[
        ["text", "actual", "pred", "prob_fake", "n_words"]]


def main():
    df = load_predictions()
    summarize(df)
    for tag in ("FP", "FN"):
        ex = examples(df, tag, n=15)
        ex.to_csv(config.RESULTS_DIR / f"errors_{tag.lower()}.csv", index=False)
        print(f"\nMost confident {tag} examples saved to results/errors_{tag.lower()}.csv")


if __name__ == "__main__":
    main()
