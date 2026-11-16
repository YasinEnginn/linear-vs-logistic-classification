# Final Dataset Selection

## Decision

The **Raisin Dataset** is used for the project.

It satisfies the assignment requirements and keeps the modeling work easy to
understand:

- 900 observations;
- exactly two balanced classes;
- seven numerical and interpretable features;
- no missing values or duplicate rows;
- no obvious target leakage;
- enough class overlap to make the comparison meaningful.

## Candidate comparison

Three different data structures were considered:

| Dataset | Main characteristic | Decision |
|---|---|---|
| Raisin | Small, balanced, numerical | Selected |
| Mushroom | Fully categorical | Not selected because it requires extensive encoding |
| Credit Card Fraud | Large and extremely imbalanced | Not selected because imbalance would dominate the comparison |

The aim is to study the two regression-based classifiers, not to make data
encoding or imbalance handling the main problem. Raisin is therefore the most
suitable choice.

## Source

- Dataset: Raisin Dataset
- Primary source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/850/raisin)
- DOI: <https://doi.org/10.24432/C5660T>
- Creators: İlkay Çınar, Murat Koklu, and Sakir Tasdemir
- Publisher: UCI Machine Learning Repository
- Download date: November 7, 2026
- License: Creative Commons Attribution 4.0 International
- Local file: `data/raw/raisin/Raisin_Dataset.xlsx`
- SHA-256: `B68BF64472DA5E3E8A07C47D78A5EFCB3A8DAFAF92770139FF1CAA1C9447CA77`

The license permits academic use with attribution. The dataset and its DOI
must be cited in the final report.

## Target

- Target column: `Class`
- Negative class (`0`): `Kecimen`
- Positive class (`1`): `Besni`

The numeric mapping is only a modeling convention. It does not rank one raisin
variety above the other.

## Data dictionary

| Column | Role | Meaning |
|---|---|---|
| `Area` | Input | Number of pixels inside the raisin boundary |
| `MajorAxisLength` | Input | Length of the longest axis |
| `MinorAxisLength` | Input | Length of the shortest axis |
| `Eccentricity` | Input | Shape elongation |
| `ConvexArea` | Input | Area of the smallest convex boundary |
| `Extent` | Input | Raisin area divided by bounding-box area |
| `Perimeter` | Input | Length of the raisin boundary |
| `Class` | Target | `Kecimen` or `Besni` |

## Verification

| Check | Result |
|---|---|
| Rows | 900 |
| Input features | 7 |
| Target classes | 2 |
| Kecimen / Besni | 450 / 450 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Obvious leakage | None found |

All input columns are physical measurements available before classification.
The target column is excluded from the feature matrix. No identifier,
timestamp, target copy, or post-classification result is used as an input.

## Modeling relevance

The EDA plots show class overlap and some curved feature relationships. This is
enough to compare:

1. scaled original features;
2. shared degree-two polynomial features.

Both models will receive the same feature matrices so the comparison remains
fair.
