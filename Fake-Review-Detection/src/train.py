"""Train Logistic Regression and Random Forest, compare them, save the best.

Run from the project root:
    python -m src.train
"""
import json

# pyrefly: ignore [missing-import]
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import load_raw, prepare
from src.models import conf_matrix, make_logreg, make_rf, metrics, split


def main():
    df = prepare(load_raw())
    print(f"Reviews after cleaning: {len(df)}")
    print(df[config.TARGET_COL].value_counts().rename({0: "genuine", 1: "fake"}))

    X_train, X_test, y_train, y_test = split(df)

    candidates = {
        "Logistic Regression": make_logreg(),
        "Random Forest": make_rf(),
    }

    rows, fitted, preds = [], {}, {}
    for name, pipe in candidates.items():
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        fitted[name], preds[name] = pipe, y_pred
        rows.append({"model": name, **metrics(y_test, y_pred)})

        try:
            cm = conf_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ax.imshow(cm, cmap="Blues")
            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, f"{val}", ha="center", va="center", color="black")
            ax.set_xticks([0, 1], labels=["Genuine", "Fake"])
            ax.set_yticks([0, 1], labels=["Genuine", "Fake"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.title(f"{name}")
            plt.tight_layout()
            plt.savefig(config.RESULTS_DIR / f"confusion_{name.replace(' ', '_').lower()}.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"Warning: could not save confusion matrix plot for {name}: {e}")

    results = pd.DataFrame(rows).round(4)
    results.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    print("\n", results.to_string(index=False))

    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        results.set_index("model")[["accuracy", "precision", "recall", "f1"]].T.plot(kind="bar", ax=ax)
        plt.ylim(0, 1.05)
        plt.title("Model comparison (test set)")
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / "model_comparison.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: could not save comparison bar chart: {e}")

    # Select best by F1 (balances precision and recall)
    best_name = results.sort_values("f1", ascending=False).iloc[0]["model"]
    joblib.dump(fitted[best_name], config.BEST_MODEL_PATH)
    print(f"\nBest model by F1: {best_name} -> saved to {config.BEST_MODEL_PATH}")

    # Save test predictions for error analysis
    out = pd.DataFrame({
        "text": X_test.values,
        "actual": y_test.values,
        "pred": preds[best_name],
        "prob_fake": fitted[best_name].predict_proba(X_test)[:, 1],
    })
    out.to_csv(config.RESULTS_DIR / "test_predictions.csv", index=False)
    (config.RESULTS_DIR / "best_model.json").write_text(json.dumps({"best_model": best_name}))


if __name__ == "__main__":
    main()
