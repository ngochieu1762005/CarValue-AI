from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


class CarPriceEvaluator:
    def __init__(self, y_true, y_pred, X_test: Optional[pd.DataFrame] = None):
        self.y_true = np.asarray(y_true, dtype=float).ravel()
        self.y_pred = np.asarray(y_pred, dtype=float).ravel()

        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"Length mismatch: y_true has {len(self.y_true)}, "
                f"y_pred has {len(self.y_pred)}."
            )
        if X_test is not None and len(X_test) != len(self.y_true):
            raise ValueError(
                f"Length mismatch: X_test has {len(X_test)} rows, "
                f"y_true has {len(self.y_true)}."
            )

        self.X_test = X_test
        self.residuals = self.y_true - self.y_pred

    @staticmethod
    def _mape(y_true, y_pred) -> float:
        mask = y_true != 0
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def calculate_base_metrics(self) -> dict:
        return {
            "MAE": round(float(mean_absolute_error(self.y_true, self.y_pred)), 2),
            "Median AE": round(float(median_absolute_error(self.y_true, self.y_pred)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(self.y_true, self.y_pred))), 2),
            "MAPE (%)": round(self._mape(self.y_true, self.y_pred), 2),
            "Bias (mean residual)": round(float(self.residuals.mean()), 2),
            "R2": round(float(r2_score(self.y_true, self.y_pred)), 4),
        }

    def evaluate_by_segment(
        self, segment_col: str, n_bins: int = 5, max_categories: int = 15
    ) -> pd.DataFrame:
        if self.X_test is None or segment_col not in self.X_test.columns:
            raise ValueError(f"{segment_col} not found in X_test dataframe.")

        segment = pd.Series(self.X_test[segment_col].to_numpy())

        if pd.api.types.is_numeric_dtype(segment) and segment.nunique() > max_categories:
            segment = pd.qcut(segment, q=n_bins, duplicates="drop")

        df = pd.DataFrame(
            {
                "Segment": segment,
                "abs_err": np.abs(self.residuals),
                "sq_err": self.residuals ** 2,
                "residual": self.residuals,
                "pct_err": np.where(
                    self.y_true != 0,
                    np.abs(self.residuals) / np.where(self.y_true == 0, 1, self.y_true) * 100,
                    np.nan,
                ),
            }
        )

        out = df.groupby("Segment", observed=True).agg(
            Count=("abs_err", "size"),
            MAE=("abs_err", "mean"),
            RMSE=("sq_err", lambda s: np.sqrt(s.mean())),
            Bias=("residual", "mean"),
            MAPE=("pct_err", "mean"),
        )
        return out.round(2).reset_index().sort_values("MAE", ascending=False)

    def _finish_plot(self, fig, save_path, show):
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_actual_vs_predicted(self, model_name="Model", save_path=None, show=True):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.y_true, self.y_pred, alpha=0.5, color="blue")

        lo = min(self.y_true.min(), self.y_pred.min())
        hi = max(self.y_true.max(), self.y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")

        ax.set_title(f"Actual vs. Predicted Prices - {model_name}")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        self._finish_plot(fig, save_path, show)
        return fig

    def plot_residuals(self, model_name="Model", save_path=None, show=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(self.y_pred, self.residuals, alpha=0.5, color="green")
        ax1.axhline(y=0, color="r", linestyle="--", lw=2)
        ax1.set_title(f"Residuals vs. Predicted - {model_name}")
        ax1.set_xlabel("Predicted Price")
        ax1.set_ylabel("Residuals (Actual - Predicted)")
        ax1.grid(True, linestyle="--", alpha=0.7)

        sns.histplot(self.residuals, kde=True, ax=ax2, color="green")
        ax2.axvline(0, color="r", linestyle="--", lw=2)
        ax2.set_title("Residual Distribution")
        ax2.set_xlabel("Residuals (Actual - Predicted)")

        fig.tight_layout()
        self._finish_plot(fig, save_path, show)
        return fig