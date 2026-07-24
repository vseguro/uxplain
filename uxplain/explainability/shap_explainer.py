"""
SHAP explainer for uncertainty metrics.
"""

from __future__ import annotations

import numpy as np
import shap

from ..protocols import ConformalPredictorProtocol
from ..uncertainty.metrics import (
    UncertaintyMetric,
    make_uncertainty_function,
)
from .fast_shap import tree_shap_values


class ShapUncertaintyExplainer:
    """
    SHAP-based explainer for uncertainty metrics.
    """

    def __init__(
        self,
        cp: ConformalPredictorProtocol,
        confidence: float = 0.9,
        algorithm: str = "auto",
        feature_names: list[str] | None = None,
        metric: UncertaintyMetric = "width",
        fast_path: bool = True,
    ):
        """
        Initialize SHAP explainer.

        Parameters
        ----------
        cp
            Fitted conformal predictor.

        confidence : float

        algorithm : str

        feature_names : list of str, optional
            Feature names for SHAP explanation output.

        metric : {"width", "lower", "upper", "midpoint"}
            Which scalar function of ``(lower, upper)`` to explain.

        fast_path : bool
            Use exact TreeSHAP on the component models when the metric is an
            affine function of them and those models are tree ensembles (see
            :mod:`uxplain.explainability.fast_shap`). Falls back to the generic
            explainer whenever the shortcut does not apply. Set to ``False`` to
            always take the generic path, e.g. to compare the two.
        """

        self.cp = cp
        self.confidence = confidence
        self.algorithm = algorithm
        self.feature_names = feature_names
        self.metric = metric
        self.fast_path = fast_path

        #: True when the last ``explain()`` call used exact TreeSHAP.
        self.used_fast_path: bool | None = None

        self._shap_explainer: shap.Explainer | None = None
        self._background: np.ndarray | None = None
        self._target_function = None

    def fit(
        self,
        X_background: np.ndarray,
        algorithm: str | None = None,
    ) -> None:
        """
        Build SHAP explainer from background data.
        """

        self._target_function = make_uncertainty_function(
            self.cp,
            confidence=self.confidence,
            metric=self.metric,
        )
        self._background = np.asarray(X_background)

        self._shap_explainer = shap.Explainer(
            self._target_function,
            X_background,
            algorithm=algorithm or self.algorithm,
        )

    def explain(
        self,
        X: np.ndarray,
    ) -> shap.Explanation:
        """
        Compute SHAP values for X.

        Takes the exact TreeSHAP shortcut when it applies, otherwise runs the
        generic SHAP explainer over the composed uncertainty function.
        """

        if self._shap_explainer is None:
            raise RuntimeError(
                "Explainer not fitted. Call fit() first."
            )

        explanation = None
        if self.fast_path:
            fast = tree_shap_values(
                self.cp,
                self.metric,
                np.asarray(X),
                self._background,
                self._target_function,
            )
            if fast is not None:
                values, base_values = fast
                explanation = shap.Explanation(
                    values=values,
                    base_values=base_values,
                    data=np.asarray(X),
                )
        self.used_fast_path = explanation is not None

        if explanation is None:
            explanation = self._shap_explainer(X)

        if self.feature_names is not None:
            explanation.feature_names = self.feature_names

        explanation.metric = self.metric

        return explanation