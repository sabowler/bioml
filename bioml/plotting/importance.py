"""
Shared feature importance plotting and export.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_feature_importance(
    features: list,
    importances: list,
    model_name: str,
    run_dir: Path,
    top_n: int = 20,
):
    """
    Save feature importance as both a bar chart PNG and an Excel file.

    Parameters
    ----------
    features : list of feature names
    importances : list of importance scores (coefficients, gain, etc.)
    model_name : str used in filenames and title
    run_dir : Path to output directory
    top_n : number of top features to show in plot
    """
    fi = (
        pd.DataFrame({"Feature": features, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    # Save full table
    excel_path = run_dir / f"{model_name}.FeatureImportance.xlsx"
    fi.to_excel(excel_path, index=False)
    logger.info(f"Feature importance saved to {excel_path}")

    # Plot top N
    plot_data = fi.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)), dpi=150)
    ax.barh(plot_data["Feature"][::-1], plot_data["Importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {min(top_n, len(fi))} Features — {model_name}")
    fig.tight_layout()
    png_path = run_dir / f"{model_name}.FeatureImportance.png"
    fig.savefig(png_path)
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {png_path}")

    return fi
