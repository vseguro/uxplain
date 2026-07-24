"""Tests for the exact TreeSHAP shortcut used on affine conformal summaries."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from uxplain import (
    CQRConformalPredictor,
    CrepesConformalPredictor,
    ShapUncertaintyExplainer,
)
from uxplain.explainability.fast_shap import tree_shap_values

REGRESSION_METRICS = ["width", "lower", "upper", "midpoint"]


def _fit_cqr(data):
    cp = CQRConformalPredictor(
        lower_model=GradientBoostingRegressor(
            loss="quantile", alpha=0.05, n_estimators=30, random_state=0
        ),
        upper_model=GradientBoostingRegressor(
            loss="quantile", alpha=0.95, n_estimators=30, random_state=0
        ),
    )
    cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
    return cp


def _fit_crepes(data, method):
    cp = CrepesConformalPredictor(
        RandomForestRegressor(n_estimators=20, random_state=0), method=method
    )
    cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
    return cp


def _explain(cp, data, metric, fast_path):
    explainer = ShapUncertaintyExplainer(cp, metric=metric, fast_path=fast_path)
    explainer.fit(data["X_calib"])
    explanation = explainer.explain(data["X_test"][:20])
    return explainer, explanation


class TestCQRFastPath:
    @pytest.mark.parametrize("metric", REGRESSION_METRICS)
    def test_fast_path_is_taken(self, data, metric):
        cp = _fit_cqr(data)
        explainer, _ = _explain(cp, data, metric, fast_path=True)
        assert explainer.used_fast_path is True

    @pytest.mark.parametrize("metric", REGRESSION_METRICS)
    def test_agrees_with_generic_path(self, data, metric):
        cp = _fit_cqr(data)
        _, fast = _explain(cp, data, metric, fast_path=True)
        _, generic = _explain(cp, data, metric, fast_path=False)
        # TreeSHAP accumulates in float32, hence the loose-ish tolerance.
        assert np.abs(np.asarray(fast.values)
                      - np.asarray(generic.values)).max() < 1e-4

    @pytest.mark.parametrize("metric", REGRESSION_METRICS)
    def test_additivity(self, data, metric):
        """base_value + sum(phi) must reproduce the metric itself."""
        cp = _fit_cqr(data)
        explainer, explanation = _explain(cp, data, metric, fast_path=True)
        target = np.asarray(
            explainer._target_function(data["X_test"][:20])
        ).ravel()
        reconstructed = (np.asarray(explanation.base_values).ravel()
                         + np.asarray(explanation.values).sum(axis=1))
        assert np.abs(reconstructed - target).max() < 1e-4

    def test_disabled_by_flag(self, data):
        cp = _fit_cqr(data)
        explainer, _ = _explain(cp, data, "width", fast_path=False)
        assert explainer.used_fast_path is False


class TestCrepesFastPath:
    """The shortcut must accept only the metrics that stay affine."""

    def test_standard_accepts_all_metrics(self, data):
        cp = _fit_crepes(data, "standard")
        for metric in REGRESSION_METRICS:
            explainer, _ = _explain(cp, data, metric, fast_path=True)
            assert explainer.used_fast_path is True, metric

    def test_standard_width_is_constant(self, data):
        """A standard interval has a fixed width, so nothing explains it."""
        cp = _fit_crepes(data, "standard")
        _, explanation = _explain(cp, data, "width", fast_path=True)
        assert np.allclose(explanation.values, 0.0)

    @pytest.mark.parametrize("metric", ["width", "lower", "upper"])
    def test_normalized_rejects_scaled_metrics(self, data, metric):
        """Normalized intervals scale with a difficulty estimator, so the
        endpoints and the width are no longer affine in the base model."""
        cp = _fit_crepes(data, "normalized")
        explainer, _ = _explain(cp, data, metric, fast_path=True)
        assert explainer.used_fast_path is False

    def test_normalized_accepts_midpoint(self, data):
        """The midpoint of a symmetric interval is the base prediction."""
        cp = _fit_crepes(data, "normalized")
        explainer, _ = _explain(cp, data, "midpoint", fast_path=True)
        assert explainer.used_fast_path is True


class TestFastPathGuards:
    def test_non_tree_model_falls_back(self, data):
        """Ridge is not a tree ensemble, so TreeSHAP cannot be used."""
        cp = CQRConformalPredictor(lower_model=Ridge(), upper_model=Ridge())
        cp.fit(data["X_train"], data["y_train"],
               data["X_calib"], data["y_calib"])
        explainer, _ = _explain(cp, data, "width", fast_path=True)
        assert explainer.used_fast_path is False

    def test_classification_metric_returns_none(self, data):
        """Classification summaries have no affine decomposition."""
        cp = _fit_cqr(data)
        result = tree_shap_values(
            cp, "set_size", data["X_test"][:5], data["X_calib"], lambda X: X[:, 0]
        )
        assert result is None
