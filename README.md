# bioml

**Binary classification toolkit for biomedical datasets.**

Built and maintained by [Scott A. Bowler](https://github.com/sabowler) — Ndhlovu Lab, Weill Cornell Medicine.

Packages four production-tested classifiers — SVM, XGBoost, KNN, and Neural Network — into a single, configurable Python library with shared preprocessing, ROC/AUC evaluation, and feature importance outputs.

---

## Quickstart

```python
import bioml

# Run XGBoost with defaults
results = bioml.run_xgb("my_data.csv", outcome_col="Case-control status")

# Run SVM, dropping ID columns
results = bioml.run_svm(
    "my_data.csv",
    outcome_col="PE",
    drop_cols=["Patient ID", "Case ID", "Event"],
)

# Run with a YAML config file
results = bioml.run_xgb("my_data.csv", config="configs/xgb.yaml")

# All runners return a results dict
print(f"Test AUC: {results['test_auc']:.3f} [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
print(f"Selected features: {results['selected_features']}")
print(f"Outputs saved to: {results['run_dir']}")
```

---

## Installation

```bash
git clone https://github.com/sabowler/bioml.git
cd bioml
pip install -e .

# Optional: Neural network support
pip install -e ".[nn]"

# Development (includes pytest)
pip install -e ".[dev]"
```

---

## Supported Classifiers

| Function | Model | Feature Selection | Tuning |
|---|---|---|---|
| `bioml.run_xgb()` | XGBoost | Correlation filter | Cross-validated grid search |
| `bioml.run_svm()` | SVM | RFECV | GridSearchCV |
| `bioml.run_knn()` | K-Nearest Neighbors | SelectKBest | GridSearchCV pipeline |
| `bioml.run_nn()` | Neural Network (TF/Keras) | Correlation filter | Config-driven |

---

## Input Data Format

- CSV (`.csv`) or Excel (`.xlsx`)
- One row per sample
- One column as the binary outcome (0/1)
- All other numeric columns treated as features
- Missing values (`NaN`) are filled with 0

---

## Output Structure

Each run creates a timestamped directory under `results/`:

```
results/
└── xgb_2025-03-07_14-30-00/
    ├── run.log                        # Full run log
    ├── xgb_model.pkl                  # Serialized model
    ├── xgb_cv_results.xlsx            # Hyperparameter search results
    ├── XGBoost.ROC.png                # ROC curve (train + test with 95% CI)
    ├── XGBoost.FeatureImportance.png  # Top features bar chart
    ├── XGBoost.FeatureImportance.xlsx # Full feature importance table
    └── surrogate_markers.csv          # Features removed due to correlation
```

---

## Configuration

All classifiers accept a YAML config file via the `config=` argument. Example configs are in `configs/`.

```yaml
# configs/xgb.yaml
outcome_col: "Case-control status"
drop_cols: ["Event", "Patient ID", "Case ID"]
test_size: 0.33
random_state: 42
corr_threshold: 0.75
output_dir: "results"
cv_folds: 10
param_grid:
  n_estimators: [50, 100, 150]
  eta: [0.01, 0.1, 1]
  min_child_weight: [1, 3, 5]
  gamma: [0, 1, 5]
  subsample: [0.5, 0.75, 1.0]
  lambda: [0, 3, 5]
  alpha: [0, 3, 5]
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Publications

This toolkit was developed in support of the following work:

- Bowler SA et al. *A machine learning approach utilizing DNA methylation as an accurate classifier of COVID-19 disease severity.* Scientific Reports (2022)
- Premeaux TA, Bowler SA et al. *Machine learning models based on fluid immunoproteins that predict non-AIDS adverse events in people with HIV.* iScience (2024)

Full publication list: [Google Scholar](https://scholar.google.com/citations?user=qM-DZhMAAAAJ)

---

## License

MIT
