# Final Dataset Selection

## Decision

The **Raisin Dataset** was selected as the final dataset for the linear-versus-logistic regression classification project.

## Selection rationale

The Raisin Dataset is the strongest fit for the project requirements because it:

- contains 900 samples, comfortably exceeding the 100-sample minimum;
- has exactly two target classes;
- is perfectly balanced, with 450 Kecimen and 450 Besni samples;
- contains seven interpretable numerical input features;
- has no missing values or duplicate rows;
- contains no obvious identifier, target copy, or post-outcome feature;
- is small enough for repeated from-scratch optimization experiments;
- supports scaling, scatter plots, polynomial features, and interaction features naturally;
- shows substantial class overlap and curved feature relationships in the initial visual analysis.

The initial graphs indicate that the classes are not perfectly separable using a single straight boundary. This is preliminary evidence of a non-trivial classification structure. The claim will later be tested by comparing the same models on raw features and shared degree-two polynomial and interaction features.

## Why the other candidates were not selected

### Mushroom Classification

The Mushroom Dataset is a valid binary-classification dataset, but all 22 inputs are categorical. It would require extensive one-hot encoding and is not a natural match for the project's numerical scatter plots and polynomial-feature comparison.

### Credit Card Fraud Detection

The Credit Card Fraud Dataset is large and realistic, but it is extremely imbalanced: only 492 of 284,807 transactions are fraudulent. In addition, 28 inputs are anonymized PCA components. This limits semantic interpretation and prevents a complete feature-level leakage audit. The imbalance would also dominate the comparison between the two regression-based classifiers.

## Source and provenance

- **Dataset name:** Raisin Dataset
- **Primary repository:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/850/raisin)
- **Kaggle mirror:** [Raisin Dataset on Kaggle](https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset)
- **Creators:** İlkay Çınar, Murat Koklu, and Sakir Tasdemir
- **Creator affiliation:** Selçuk University, Türkiye
- **Repository publisher:** UCI Machine Learning Repository
- **Original study year:** 2020
- **UCI DOI:** <https://doi.org/10.24432/C5660T>
- **Download date:** November 7, 2026
- **UCI license:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Kaggle license:** CC0: Public Domain
- **Academic suitability:** Suitable for academic use. The dataset and original study must be cited in the report.

Recommended dataset citation:

> Çınar, İ., Koklu, M., & Tasdemir, S. (2020). Raisin [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5660T

## Local raw-data storage

- **Final project copy:** `data/raw/raisin/Raisin_Dataset.xlsx`
- **Downloaded source copy:** `datasets/raw/raisin/Raisin_Dataset/Raisin_Dataset.xlsx`
- **File size:** 84,629 bytes
- **SHA-256:** `B68BF64472DA5E3E8A07C47D78A5EFCB3A8DAFAF92770139FF1CAA1C9447CA77`

The source and final-project copies have matching SHA-256 hashes. Therefore, the file under `data/raw/` is an unchanged byte-for-byte copy of the downloaded raw dataset.

The raw file must never be edited in place. Any cleaning, target encoding, scaling, or feature engineering output must be written later under `data/processed/`.

## Dataset verification

| Check | Result |
|---|---|
| Number of samples | 900 |
| Input features | 7 |
| Target columns | 1 |
| Dataset shape | 900 rows × 8 columns |
| Target classes | Kecimen and Besni |
| Kecimen samples | 450 |
| Besni samples | 450 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Obvious target leakage | None identified |

## Target definition

- **Target column:** `Class`
- **Negative class (`0`):** `Kecimen`
- **Positive class (`1`):** `Besni`

Neither variety is inherently a positive or negative outcome. The mapping is a documented modeling convention and must remain consistent throughout preprocessing, training, evaluation, figures, and reporting.

## Data dictionary

| Column | Role | Type | Unit | Meaning |
|---|---|---|---|---|
| `Area` | Input | Integer | pixels | Number of pixels inside the raisin boundary. |
| `MajorAxisLength` | Input | Continuous | pixels | Length of the longest axis of the raisin. |
| `MinorAxisLength` | Input | Continuous | pixels | Length of the shortest axis of the raisin. |
| `Eccentricity` | Input | Continuous | unitless | Degree to which the raisin shape is elongated. |
| `ConvexArea` | Input | Integer | pixels | Area of the smallest convex boundary surrounding the raisin. |
| `Extent` | Input | Continuous | unitless ratio | Ratio of raisin area to the area of its bounding box. |
| `Perimeter` | Input | Continuous | pixels | Length of the raisin boundary. |
| `Class` | Target | Categorical | — | Raisin variety: Kecimen or Besni. |

## Leakage assessment

All seven inputs are morphological measurements extracted from raisin images. They are available before the class prediction is made. None directly contains the words Kecimen or Besni, a previously predicted class, or information generated after classification.

Some features may be highly predictive of the class. Strong predictive power alone is not target leakage when the measurement is legitimately available at prediction time.

## Initial visual evidence

The following generated figures support the initial dataset assessment:

- `experiments/01_dataset_candidate_analysis/outputs/raisin_class_distribution.pdf`
- `experiments/01_dataset_candidate_analysis/outputs/raisin_pairplot.pdf`
- `experiments/01_dataset_candidate_analysis/outputs/raisin_eccentricity_perimeter.pdf`
- `experiments/01_dataset_candidate_analysis/outputs/candidate_comparison_table.pdf`

The pair plot and focused Eccentricity–Perimeter scatter plot show substantial overlap between Kecimen and Besni. The feature relationships also include curved patterns and outliers. These observations suggest that a simple linear boundary will not classify every sample correctly.

## Next step

Create a reproducible preprocessing pipeline that:

1. loads the unchanged file from `data/raw/raisin/Raisin_Dataset.xlsx`;
2. encodes Kecimen as `0` and Besni as `1`;
3. performs a stratified train/validation/test split;
4. calculates scaling parameters using the training set only;
5. preserves the test set for final evaluation.
