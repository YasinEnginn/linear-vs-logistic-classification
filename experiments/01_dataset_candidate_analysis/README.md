# Dataset Candidate Analysis

This experiment compares three structurally different binary-classification datasets before the final dataset is selected.

| Candidate | Structure | Source | License | Local file expected by the script |
|---|---|---|---|---|
| Raisin | Balanced numerical measurements | [UCI](https://archive.ics.uci.edu/dataset/850/raisin), [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset) | UCI: CC BY 4.0; Kaggle: CC0 | `data/raw/raisin/Raisin_Dataset.xlsx` |
| Mushroom | Categorical attributes | [UCI](https://archive.ics.uci.edu/dataset/73/mushroom), [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification) | CC BY 4.0 / database source terms | `datasets/raw/mushroom/mushrooms.csv` |
| Credit Card Fraud | Large, anonymized, extremely imbalanced numerical data | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), [ULB](https://www.ulb.be/di/map/adalpozz/data/creditcard.Rdata) | Kaggle dataset terms / source citation required | `datasets/raw/credit_card_fraud/creditcard.csv` |

Only the selected Raisin raw file is versioned. Downloaded candidate files remain local and are ignored by Git because the source links are preserved and the Credit Card Fraud file exceeds GitHub's regular file-size limit.

To regenerate all figures after placing the files at the paths above, activate the project environment and run from the repository root:

```powershell
python experiments/01_dataset_candidate_analysis/candidate_visual_analysis.py
```

The script writes both PNG and PDF versions to `outputs/`. PNG files are convenient for Markdown and presentations; PDF files preserve vector quality for the final report where supported.
