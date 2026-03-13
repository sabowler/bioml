"""Shared feature importance plotting and export."""
import logging
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_feature_importance(
    features: list,
    importances: list,
    title: str = "Feature Importance",
    output_path: str = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Plot and export feature importances.

    Parameters
    ----------
    features : list of str
        Feature names.
    importances : list of float
        Corresponding importance scores (coefficients, gain, etc.).
    title : str
        Plot title.
    output_path : str, optional
        If provided, saves PNG and an Excel file (output_path.xlsx).
    top_n : int
        Maximum number of features to display (default 20).

    Returns
    -------
    pd.DataFrame
        Sorted feature importance table with columns ['Feature', 'Importance'].
    """
    fi_df = (
        pd.DataFrame({"Feature": features, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    plot_df = fi_df.head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.35)), dpi=150)
    ax.barh(plot_df["Feature"][::-1], plot_df["Importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        excel_path = output_path.replace(".png", ".xlsx")
        fi_df.to_excel(excel_path, index=False)
        logger.info(f"Feature importance saved to {output_path} and {excel_path}")
    else:
        plt.show()
    plt.close(fig)

    return fi_df
