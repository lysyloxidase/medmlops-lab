"""XGBoost baseline for honest tabular-model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass(frozen=True)
class GBDTBaselineConfig:
    """Deterministic CPU XGBoost configuration."""

    backend: str = "xgboost"
    n_estimators: int = 80
    max_depth: int = 3
    learning_rate: float = 0.08
    min_child_weight: float = 5.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_lambda: float = 1.0
    seed: int = 42


class GBDTBaseline:
    """XGBoost baseline the PyTorch hero must beat or match."""

    def __init__(self, config: GBDTBaselineConfig | None = None) -> None:
        self.config = config or GBDTBaselineConfig()
        self.effective_backend = self.config.backend
        self.model = self._build_model()

    def _build_model(self) -> Any:
        if self.config.backend == "xgboost":
            try:
                import xgboost as xgb

                return xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    min_child_weight=self.config.min_child_weight,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    reg_lambda=self.config.reg_lambda,
                    random_state=self.config.seed,
                    seed=self.config.seed,
                    n_jobs=1,
                )
            except Exception:
                self.effective_backend = "sklearn_hist_gradient_boosting"
        else:
            self.effective_backend = "sklearn_hist_gradient_boosting"

        return HistGradientBoostingClassifier(
            max_iter=self.config.n_estimators,
            max_leaf_nodes=2**self.config.max_depth,
            learning_rate=self.config.learning_rate,
            l2_regularization=self.config.reg_lambda,
            random_state=self.config.seed,
        )

    def fit(
        self,
        x: Any,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> GBDTBaseline:
        """Fit the baseline on a preprocessed feature matrix."""

        matrix = self._matrix_for_backend(x)
        if self.effective_backend == "xgboost":
            self.model.fit(matrix, y, sample_weight=sample_weight, verbose=False)
        else:
            self.model.fit(matrix, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, x: Any) -> NDArray[np.float64]:
        """Return positive-class probabilities."""

        probabilities = self.model.predict_proba(self._matrix_for_backend(x))
        return np.asarray(probabilities[:, 1], dtype=np.float64)

    def _matrix_for_backend(self, x: Any) -> Any:
        using_fallback = self.effective_backend == "sklearn_hist_gradient_boosting"
        if using_fallback and sparse.issparse(x):
            return x.toarray()
        return x
