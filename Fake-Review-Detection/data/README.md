# Dataset

Recommended: **Fake Reviews Dataset** (Kaggle, uploaded by `mexwell`), based on the paper
*"Creating and detecting fake reviews of online products"* (Salminen et al., 2022).

1. Open https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset (free Kaggle account needed).
2. Click **Download** and unzip.
3. Rename the CSV to `reviews.csv` and put it in this folder: `Fake-Review-Detection/data/reviews.csv`.

Expected columns:

| column   | meaning                                             |
|----------|-----------------------------------------------------|
| category | product category                                    |
| rating   | star rating (1-5)                                   |
| label    | `CG` = computer-generated (**fake**), `OR` = original (**genuine**) |
| text_    | the review text                                     |

Check the real row count and class balance yourself with notebook `01_data_analysis.ipynb`;
don't trust numbers quoted anywhere else.

Important caveat: the "fake" reviews here are machine-generated, not human-written spam found on a real
site, so scores on this dataset will likely look better than real-world performance.
