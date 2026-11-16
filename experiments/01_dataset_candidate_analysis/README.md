# Dataset Candidate Analysis

This folder contains the initial comparison used to select one dataset. The
comparison is intentionally simple: source, license, sample count, target
classes, data structure, leakage risk, and initial plots.

| Candidate | Samples | Target | Structure | Source | License | Academic use |
|---|---:|---|---|---|---|---|
| Raisin | 900 | Kecimen / Besni | Balanced numerical measurements | [UCI](https://archive.ics.uci.edu/dataset/850/raisin) | CC BY 4.0 | Permitted with attribution |
| Mushroom | 8,124 | Edible / Poisonous | Categorical attributes | [UCI](https://archive.ics.uci.edu/dataset/73/mushroom) | CC BY 4.0 | Permitted with attribution |
| Credit Card Fraud | 284,807 | Normal / Fraud | Large, extremely imbalanced numerical data | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), [ULB](https://www.ulb.be/di/map/adalpozz/data/creditcard.Rdata) | Open Database License / Database Contents License | Permitted under the license terms; source citation required |

All three datasets exceed 100 observations and have exactly two target classes.
The detailed leakage and structure checks are represented in the generated
comparison table and figures.

The license statements refer to the linked primary catalog pages. Academic
use is suitable when their attribution and citation conditions are followed.

Raisin was selected because it lets the project focus on scaling, polynomial
features, gradient descent, and the difference between linear and logistic
cost functions.

To regenerate the candidate figures from the repository root:

```powershell
python experiments/01_dataset_candidate_analysis/candidate_visual_analysis.py
```

The script expects the source files at the paths documented inside the script
and writes PNG and PDF figures to `outputs/`.
