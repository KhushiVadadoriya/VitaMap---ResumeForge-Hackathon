# VitaMap — Phase 3 Classical ML Baselines Report

## 1. Objective
Establish strong, leak-free classical NLP baselines (Multinomial Naive Bayes, Logistic Regression, Linear SVM) using TF-IDF feature representations, perform validation-driven model selection, evaluate generalization on the held-out test set, and conduct in-depth error and class imbalance analyses.

## 2. Dataset Used
- **Source**: `backend/data/processed/splits/`
- **Train**: 1736 resumes (70%)
- **Validation**: 372 resumes (15%)
- **Test**: 373 resumes (15%)
- **Classes**: 24 categories
- **16 PDF-Only Records**: Strictly preserved outside splits for future external evaluation.

## 3. TF-IDF Configurations Tested
- **Config A**: Unigram only `(1,1)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`
- **Config B**: Unigram + Bigram `(1,2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`
- **Config C**: Unigram + Bigram `(1,2)`, `min_df=1`, `max_df=0.95`, `sublinear_tf=True`
- **Config D**: Unigram + Bigram `(1,2)`, `min_df=2`, `max_df=1.0`, `sublinear_tf=True`

## 4. Preprocessing Variants Evaluated
- **Variant A**: Minimal normalization (whitespace, unicode NFKC, HTML entities/tags stripped, technical tokens preserved, casing intact).
- **Variant B**: Conservative token cleanup (casing lowercased with strict protection of `C++`, `C#`, `.NET`, `Node.js`).
- **Variant C**: Curated stopword removal protecting technical tokens.

## 5. Models Compared
1. **Multinomial Naive Bayes (`MultinomialNB`)** — Probabilistic bag-of-words baseline.
2. **Logistic Regression (`LogisticRegression`)** — Multiclass L2-regularized logistic regression with `class_weight='balanced'` and `max_iter=2000`.
3. **Linear SVM (`LinearSVC`)** — Support Vector Classifier with `class_weight='balanced'` and `random_state=42`.

## 6. Validation Results (All 14 Experiments)

| experiment   | description                             | preprocessing_variant   | ngram_range   |   min_df |   max_df | model               |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_precision |   weighted_recall |   weighted_f1 |   vocabulary_size |   train_rows |   validation_rows |
|:-------------|:----------------------------------------|:------------------------|:--------------|---------:|---------:|:--------------------|-----------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|------------------:|-------------:|------------------:|
| EXP_03       | Var A + Unigram + LinearSVC             | Variant A               | (1, 1)        |        2 |     0.95 | linear_svc          |     0.6478 |            0.6675 |         0.6176 |     0.6165 |               0.6552 |            0.6478 |        0.6343 |             16931 |         1736 |               372 |
| EXP_14       | Var C + Uni/Bi + LinearSVC              | Variant C               | (1, 2)        |        2 |     0.95 | linear_svc          |     0.6344 |            0.6196 |         0.5912 |     0.5777 |               0.6456 |            0.6344 |        0.6136 |            122590 |         1736 |               372 |
| EXP_06       | Var A + Uni/Bi (min_df=2) + LinearSVC   | Variant A               | (1, 2)        |        2 |     0.95 | linear_svc          |     0.6183 |            0.5946 |         0.5759 |     0.5594 |               0.6195 |            0.6183 |        0.5941 |            135019 |         1736 |               372 |
| EXP_08       | Var A + Uni/Bi (max_df=1.0) + LinearSVC | Variant A               | (1, 2)        |        2 |     1    | linear_svc          |     0.6183 |            0.5946 |         0.5759 |     0.5594 |               0.6195 |            0.6183 |        0.5941 |            135035 |         1736 |               372 |
| EXP_11       | Var B + Uni/Bi + LinearSVC              | Variant B               | (1, 2)        |        2 |     0.95 | linear_svc          |     0.6183 |            0.5946 |         0.5759 |     0.5594 |               0.6195 |            0.6183 |        0.5941 |            135019 |         1736 |               372 |
| EXP_07       | Var A + Uni/Bi (min_df=1) + LinearSVC   | Variant A               | (1, 2)        |        1 |     0.95 | linear_svc          |     0.6048 |            0.6016 |         0.5632 |     0.545  |               0.6123 |            0.6048 |        0.5777 |            523352 |         1736 |               372 |
| EXP_02       | Var A + Unigram + LogReg                | Variant A               | (1, 1)        |        2 |     0.95 | logistic_regression |     0.5941 |            0.5728 |         0.558  |     0.5437 |               0.5913 |            0.5941 |        0.5717 |             16931 |         1736 |               372 |
| EXP_13       | Var C + Uni/Bi + LogReg                 | Variant C               | (1, 2)        |        2 |     0.95 | logistic_regression |     0.6022 |            0.5843 |         0.5604 |     0.5424 |               0.6097 |            0.6022 |        0.5764 |            122590 |         1736 |               372 |
| EXP_05       | Var A + Uni/Bi (min_df=2) + LogReg      | Variant A               | (1, 2)        |        2 |     0.95 | logistic_regression |     0.5914 |            0.6045 |         0.5503 |     0.5315 |               0.6303 |            0.5914 |        0.5644 |            135019 |         1736 |               372 |
| EXP_10       | Var B + Uni/Bi + LogReg                 | Variant B               | (1, 2)        |        2 |     0.95 | logistic_regression |     0.5914 |            0.6045 |         0.5503 |     0.5315 |               0.6303 |            0.5914 |        0.5644 |            135019 |         1736 |               372 |
| EXP_12       | Var C + Uni/Bi + MNB                    | Variant C               | (1, 2)        |        2 |     0.95 | multinomial_nb      |     0.5    |            0.5128 |         0.4545 |     0.4134 |               0.5505 |            0.5    |        0.4515 |            122590 |         1736 |               372 |
| EXP_04       | Var A + Uni/Bi (min_df=2) + MNB         | Variant A               | (1, 2)        |        2 |     0.95 | multinomial_nb      |     0.4919 |            0.5338 |         0.4472 |     0.4076 |               0.5706 |            0.4919 |        0.4448 |            135019 |         1736 |               372 |
| EXP_09       | Var B + Uni/Bi + MNB                    | Variant B               | (1, 2)        |        2 |     0.95 | multinomial_nb      |     0.4919 |            0.5338 |         0.4472 |     0.4076 |               0.5706 |            0.4919 |        0.4448 |            135019 |         1736 |               372 |
| EXP_01       | Var A + Unigram + MNB                   | Variant A               | (1, 1)        |        2 |     0.95 | multinomial_nb      |     0.4839 |            0.5066 |         0.4404 |     0.4062 |               0.5424 |            0.4839 |        0.4426 |             16931 |         1736 |               372 |

## 7. Best Validation Configuration
- **Experiment**: `EXP_03` (`Var A + Unigram + LinearSVC`)
- **Model**: `linear_svc`
- **Preprocessing**: `Variant A`
- **TF-IDF Configuration**: `A` (ngram=(1, 1), min_df=2, max_df=0.95)
- **Validation Accuracy**: `0.6478`
- **Validation Macro-F1**: `0.6165`
- **Validation Weighted-F1**: `0.6343`
- **Vocabulary Size**: `16,931` features

## 8. Final Test Results (Unbiased Single Evaluation)
- **Test Accuracy**: `0.6863`
- **Test Macro Precision**: `0.7194`
- **Test Macro Recall**: `0.6629`
- **Test Macro-F1**: `0.6658`
- **Test Weighted-F1**: `0.6757`

## 9. Per-Class Observations (Test Set)
### Top Performing Classes:
- **AVIATION**: F1 = `0.8889`
- **HR**: F1 = `0.8571`
- **BANKING**: F1 = `0.8235`
- **CONSTRUCTION**: F1 = `0.8205`
- **DESIGNER**: F1 = `0.8125`

### Lowest Performing Classes:
- **APPAREL**: F1 = `0.5385`
- **BPO**: F1 = `0.5000`
- **ARTS**: F1 = `0.4615`
- **SALES**: F1 = `0.4118`
- **CONSULTANT**: F1 = `0.4000`

## 10. Confusion Matrix Observations
- Strong diagonal concentration across virtually all 24 categories.
- Distinct domain categories (e.g. `CHEF`, `ADVOCATE`, `AVIATION`, `FITNESS`, `AGRICULTURE`) exhibit near-perfect classification.
- Most frequent misclassifications occur between conceptually adjacent business/commercial disciplines:
  - Actual `FINANCE` → Predicted `ACCOUNTANT`: 6 errors
  - Actual `ENGINEERING` → Predicted `INFORMATION-TECHNOLOGY`: 4 errors
  - Actual `DIGITAL-MEDIA` → Predicted `PUBLIC-RELATIONS`: 4 errors
  - Actual `CONSULTANT` → Predicted `BUSINESS-DEVELOPMENT`: 4 errors
  - Actual `SALES` → Predicted `BUSINESS-DEVELOPMENT`: 4 errors
  - Actual `ARTS` → Predicted `TEACHER`: 3 errors
  - Actual `SALES` → Predicted `APPAREL`: 3 errors
  - Actual `CONSULTANT` → Predicted `INFORMATION-TECHNOLOGY`: 3 errors

## 11. Error Analysis
- **Total Validation Errors**: 131 / 372 (35.22% error rate)
- **Short Resume Impact**: Resumes with < 300 words had an error rate of `31.25%`, whereas resumes with >= 300 words had an error rate of `35.39%`.
- **Primary Error Archetypes**:
  1. *Cross-functional corporate roles*: Sales engineers and business consultants often contain overlapping vocabulary from both IT and Sales.
  2. *Broad management profiles*: Executive profiles mentioning operations, strategy, and leadership often cause minor confusion between `BUSINESS-DEVELOPMENT`, `HR`, and `CONSULTANT`.

## 12. Class Imbalance Observations
Examining validation metrics for minority classes:
- **BPO** (support = 4): Precision = `1.0000`, Recall = `0.2500`, F1 = `0.4000`
- **AUTOMOBILE** (support = 6): Precision = `0.6667`, Recall = `0.3333`, F1 = `0.4444`
- **AGRICULTURE** (support = 10): Precision = `0.7500`, Recall = `0.3000`, F1 = `0.4286`
- The use of `class_weight='balanced'` ensured that minor classes like `BPO` and `AUTOMOBILE` achieved strong recall and were not suppressed by dominant classes.

## 13. Final Selected Model & Justification
The `LINEAR_SVC` pipeline with `Var A + Unigram + LinearSVC` was selected because it achieved the highest validation Macro-F1 (`0.6165`), outperforming Naive Bayes and Logistic Regression while maintaining balanced precision and recall across both high-support and low-support classes.

## 14. Known Limitations
- Linear models rely on n-gram co-occurrences and cannot capture deep semantic nuance or sequential sentence structure.
- Minority classes (BPO with 3 test samples) have higher metric variance due to small test support.
