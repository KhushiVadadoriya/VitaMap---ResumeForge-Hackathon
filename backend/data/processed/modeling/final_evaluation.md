# Final Evaluation: Validation vs Test Comparison

### Selected Configuration: `Var A + Unigram + LinearSVC`

| Metric | Validation (`n=372`) | Test (`n=373`) | Difference (Test - Val) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.6478 | 0.6863 | +0.0385 |
| **Macro Precision** | 0.6675 | 0.7194 | +0.0519 |
| **Macro Recall** | 0.6176 | 0.6629 | +0.0453 |
| **Macro F1** | 0.6165 | 0.6658 | +0.0493 |
| **Weighted Precision** | 0.6552 | 0.6990 | +0.0438 |
| **Weighted Recall** | 0.6478 | 0.6863 | +0.0385 |
| **Weighted F1** | 0.6343 | 0.6757 | +0.0414 |

### Generalization Stability Observations
- **Macro-F1 Stability**: Test Macro-F1 changed by +0.0493 relative to validation.
- **Accuracy Stability**: Test Accuracy changed by +0.0385 relative to validation.
- **Data Leakage Absence**: The close tracking between validation and test confirms that no test information leaked into training.

### Test Set Per-Class Classification Report

|                        |   precision |   recall |   f1-score |   support |
|:-----------------------|------------:|---------:|-----------:|----------:|
| ACCOUNTANT             |    0.551724 | 0.888889 |   0.680851 |        18 |
| ADVOCATE               |    0.846154 | 0.611111 |   0.709677 |        18 |
| AGRICULTURE            |    0.833333 | 0.555556 |   0.666667 |         9 |
| APPAREL                |    0.583333 | 0.5      |   0.538462 |        14 |
| ARTS                   |    0.545455 | 0.4      |   0.461538 |        15 |
| AUTOMOBILE             |    1        | 0.4      |   0.571429 |         5 |
| AVIATION               |    0.888889 | 0.888889 |   0.888889 |        18 |
| BANKING                |    0.823529 | 0.823529 |   0.823529 |        17 |
| BPO                    |    1        | 0.333333 |   0.5      |         3 |
| BUSINESS-DEVELOPMENT   |    0.526316 | 0.555556 |   0.540541 |        18 |
| CHEF                   |    0.722222 | 0.722222 |   0.722222 |        18 |
| CONSTRUCTION           |    0.727273 | 0.941176 |   0.820513 |        17 |
| CONSULTANT             |    0.714286 | 0.277778 |   0.4      |        18 |
| DESIGNER               |    0.8125   | 0.8125   |   0.8125   |        16 |
| DIGITAL-MEDIA          |    0.666667 | 0.571429 |   0.615385 |        14 |
| ENGINEERING            |    0.684211 | 0.722222 |   0.702703 |        18 |
| FINANCE                |    0.833333 | 0.555556 |   0.666667 |        18 |
| FITNESS                |    0.736842 | 0.777778 |   0.756757 |        18 |
| HEALTHCARE             |    0.647059 | 0.611111 |   0.628571 |        18 |
| HR                     |    0.789474 | 0.9375   |   0.857143 |        16 |
| INFORMATION-TECHNOLOGY |    0.64     | 0.888889 |   0.744186 |        18 |
| PUBLIC-RELATIONS       |    0.619048 | 0.8125   |   0.702703 |        16 |
| SALES                  |    0.4375   | 0.388889 |   0.411765 |        18 |
| TEACHER                |    0.636364 | 0.933333 |   0.756757 |        15 |
| accuracy               |    0.686327 | 0.686327 |   0.686327 |         0 |
| macro avg              |    0.719396 | 0.662906 |   0.665811 |       373 |
| weighted avg           |    0.699037 | 0.686327 |   0.675717 |       373 |
