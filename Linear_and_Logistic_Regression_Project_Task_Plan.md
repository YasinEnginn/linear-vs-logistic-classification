# Project Task Plan

## Learning objective

The purpose of this project is to verify that we can correctly apply the
concepts learned in Course 1, Weeks 1–3. The project should remain small,
readable, and mathematically correct.

Course references: `C1_W1.pdf`, `C1_W2.pdf`, and `C1_W3.pdf`.

We need to demonstrate:

- a linear model and squared-error cost;
- a logistic model, sigmoid, and binary cross-entropy cost;
- batch gradient descent;
- feature scaling;
- degree-two feature engineering;
- L2 regularization;
- a fair comparison on the same binary-classification data.

Ready-made regression or classification models are not allowed.

## 1. Dataset

- [x] Compare at least three candidates.
- [x] Check source, license, sample count, two-class target, and leakage.
- [x] Select the Raisin Dataset.
- [x] Keep the raw file unchanged under `data/raw/`.
- [x] Record the target and feature meanings.

## 2. Data inspection

- [x] Report shape, columns, and data types.
- [x] Count classes, missing values, and duplicates.
- [x] Calculate descriptive statistics.
- [x] Plot class counts, histograms, boxplots, correlations, and selected
  scatterplots.
- [x] Inspect outliers without deleting them automatically.

## 3. Preprocessing

- [x] Encode `Kecimen` as `0` and `Besni` as `1`.
- [x] Create a stratified 60/20/20 train/validation/test split.
- [x] Calculate means and standard deviations from training data only.
- [x] Apply the same scaling values to validation and test data.
- [x] Prepare two feature sets:
  - A: seven scaled original features;
  - B: shared degree-two square and interaction features.

## 4. Linear regression classifier

Use:

\[
f_{w,b}(X)=Xw+b
\]

\[
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})^2
+\frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2
\]

- [x] Implement score, cost, gradient, gradient descent, and thresholded
  prediction with NumPy.
- [x] Store cost history.
- [x] Test training and prediction.

The linear output is a score, not a probability.

## 5. Logistic regression classifier

Use:

\[
g(z)=\frac{1}{1+e^{-z}}, \qquad f_{w,b}(X)=g(Xw+b)
\]

\[
J(w,b)=-\frac{1}{m}\sum_{i=1}^{m}
\left[y^{(i)}\log(f^{(i)})+(1-y^{(i)})\log(1-f^{(i)})\right]
+\frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2
\]

- [x] Implement sigmoid, probability, cost, gradient, gradient descent, and
  class prediction with NumPy.
- [x] Store cost history.
- [x] Test sigmoid, gradient, training, and probability range.

For both models, regularize `w` but not `b`.

## 6. Evaluation

- [x] Calculate `TP`, `TN`, `FP`, and `FN`.
- [x] Calculate accuracy, precision, recall, specificity, and F1-score.
- [x] Handle zero denominators safely.
- [x] Inspect a small number of validation thresholds.
- [x] Choose one validation metric and one final threshold.

## 7. Small validation experiment

Try only a few understandable values for:

- learning rate;
- iteration count;
- `lambda_`;
- threshold, if it is changed from `0.5`.

- [x] Use validation data for every choice.
- [x] Check that cost decreases.
- [x] Do not create a large or complicated search system.

## 8. Final comparison

Run both models on both feature sets:

| Features | Linear model | Logistic model |
|---|---:|---:|
| Scaled original features | A1 | A2 |
| Degree-two features | B1 | B2 |

- [x] Use identical observations and features for both models.
- [x] Evaluate the chosen settings once on the test set.
- [x] Report metrics, confusion matrices, and cost curves.
- [ ] Explain what was learned from the comparison in the final report.

## 9. Final checks

- [x] Run notebooks from top to bottom.
- [x] Run `python -m pytest -q`.
- [x] Confirm that no ready-made model class is imported.
- [x] Record dependency versions in `requirements.txt`.
- [ ] State known limitations in the final report.
- [ ] Keep the final report focused on whether the learned concepts were
  applied correctly.
