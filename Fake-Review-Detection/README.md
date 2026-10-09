# AI-Based Fake Review Detection Using NLP and Machine Learning

Classifies a product review as **Genuine** or **Fake** using TF-IDF features with Logistic Regression and
Random Forest, plus experiments, error analysis, explainability, a SQLite prediction log and a Streamlit app.

## Architecture

| Layer | Where | What |
|---|---|---|
| Data | `data/reviews.csv` (you download it) | Kaggle Fake Reviews Dataset |
| Backend | `src/` | loading, cleaning, models, training, experiments, error analysis, explainability, predictor |
| Database | `database/predictions.db` (auto-created) | SQLite log of predictions made in the app (`src/db.py`) |
| Frontend | `app.py` | Streamlit UI |
| Analysis | `notebooks/` | EDA, preprocessing, model comparison |

## Folder structure

```
Fake-Review-Detection/
├── app.py                  Streamlit frontend
├── requirements.txt
├── data/                   put reviews.csv here (see data/README.md)
├── database/               SQLite file is created here
├── models/                 best_model.pkl is saved here by training
├── results/                metrics, plots, error tables (created by scripts)
├── notebooks/              01_data_analysis, 02_preprocessing, 03_model_comparison
├── src/                    config, data, preprocess, models, train, experiments,
│                           error_analysis, explain, predictor, db
└── tests/                  quick sanity tests
```

## Setup (Windows PowerShell)

```powershell
cd Fake-Review-Detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
(macOS/Linux: `source venv/bin/activate`)

## Run order

1. Download the dataset and save it as `data/reviews.csv` (instructions in `data/README.md`).
2. `jupyter notebook` and open `notebooks/01_data_analysis.ipynb` to inspect the data.
3. `python -m src.train` : trains both models, writes `results/model_comparison.csv`, plots, and `models/best_model.pkl`.
4. `python -m src.experiments` : TF-IDF configs and class-imbalance experiments -> `results/experiments.csv`.
5. `python -m src.error_analysis` : false positive / false negative tables.
6. `streamlit run app.py` : opens the web app.
7. `python -m pytest tests` : sanity tests (optional).

## Notes

- Fake = positive class (1). Label `CG` = computer-generated = fake, `OR` = original = genuine.
- All metrics come from YOUR run; none are pre-filled. Write your report from `results/`.
- The dataset's fake reviews are machine-generated, so real-world performance on other sites will likely be lower.
- Explainability: Logistic Regression coefficient x TF-IDF value per word (exact for that model); optional SHAP cell in notebook 03.
