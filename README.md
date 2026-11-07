# Linear and Logistic Regression for Binary Classification

This project will implement linear regression and logistic regression from scratch and compare them fairly on the same binary-classification problem. The selected problem is the classification of raisin varieties from image-derived physical measurements.

## Current status

- [x] Define the project requirements and comparison objective
- [x] Research three structurally different binary-classification candidates
- [x] Verify sources, licenses, sample counts, target classes, and leakage risks
- [x] Perform an initial visual assessment of candidate nonlinearity
- [x] Select and document the final dataset
- [x] Preserve the selected raw dataset without modification
- [x] Complete the basic data inspection (EDA Task 3.1)
- [ ] Produce the full exploratory visualization set
- [ ] Define preprocessing and stratified train/validation/test splits
- [ ] Implement linear regression from scratch
- [ ] Implement logistic regression from scratch
- [ ] Tune both models under the same validation protocol
- [ ] Compare raw and polynomial/interaction feature spaces
- [ ] Evaluate the final models and write the report

## Selected dataset

| Item | Value |
|---|---|
| Dataset | Raisin Dataset |
| Primary source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/850/raisin) |
| License | CC BY 4.0 |
| Shape | 900 rows × 8 columns |
| Input features | 7 numerical measurements |
| Target | `Class` |
| Classes | `Kecimen` and `Besni` (450 samples each) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Obvious target leakage | None identified |

The small selected raw file is versioned so that the main analysis is reproducible. Larger candidate datasets and downloaded archives are intentionally excluded because their authoritative sources are documented. See [the final dataset-selection record](docs/final_dataset_selection.md) for provenance, licensing, the data dictionary, integrity hash, and the complete decision rationale.

## Repository structure

```text
data/
  raw/raisin/                       Unchanged selected dataset
docs/
  final_dataset_selection.md        Selection record and data dictionary
experiments/
  01_dataset_candidate_analysis/    Candidate-analysis script and figures
notebooks/
  basic_data_inspection.ipynb       EDA Task 3.1
requirements.txt                    Python dependencies
```

## Setup

The project was developed with Python 3.13 on Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open `notebooks/basic_data_inspection.ipynb` in VS Code and select the `.venv` Python kernel. Run the cells from top to bottom. The notebook loads the unchanged file from `data/raw/raisin/Raisin_Dataset.xlsx`.

## Candidate analysis

The three candidates intentionally represent different data structures:

- Raisin: compact, balanced, numerical, and interpretable;
- Mushroom: categorical and encoding-intensive;
- Credit Card Fraud: numerical, large, anonymized, and extremely imbalanced.

The report-ready PNG and PDF figures are stored under `experiments/01_dataset_candidate_analysis/outputs/`. The [candidate-analysis notes](experiments/01_dataset_candidate_analysis/README.md) list the data sources and local paths required to regenerate every figure.

## Data policy

- Never edit files under `data/raw/` in place.
- Write future cleaning, encoding, scaling, and feature-engineering results under `data/processed/`.
- Do not commit locally downloaded candidate datasets or archives.
- Record every transformation in code so that processed data can be regenerated.
