"""
bioml - Binary classification toolkit for biomedical datasets.

Supports SVM, XGBoost, KNN, and Neural Network classifiers with
shared preprocessing, ROC/AUC evaluation, and feature importance outputs.

Example usage:
    import bioml

    results = bioml.run_xgb("data.csv", outcome_col="Case-control status")
    results = bioml.run_svm("data.csv", outcome_col="PE")
    results = bioml.run_knn("data.csv", outcome_col="Case-control status")
    results = bioml.run_nn("data.csv", outcome_col="Case-control status")

    # Or with a YAML config file:
    results = bioml.run_xgb("data.csv", config="configs/xgb.yaml")
"""

from bioml.classifiers.svm import run_svm
from bioml.classifiers.xgb import run_xgb
from bioml.classifiers.knn import run_knn
from bioml.classifiers.nn import run_nn

__version__ = "0.1.0"
__author__ = "Scott A. Bowler"
__all__ = ["run_svm", "run_xgb", "run_knn", "run_nn"]
