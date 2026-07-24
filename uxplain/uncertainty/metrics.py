"""
Uncertainty metrics for conformal prediction.

These functions define scalar measures derived from a conformal predictor's
output that can serve as the *target* of an explainer.

Regression metrics reduce ``(lower, upper)`` to a single scalar:

- ``"width"``    : ``upper - lower``           (how wide the interval is)
- ``"lower"``    : ``lower``                   (the pessimistic prediction)
- ``"upper"``    : ``upper``                   (the optimistic prediction)
- ``"midpoint"`` : ``(lower + upper) / 2``     (the central prediction)

Classification metrics reduce ``(predict_set, predict_p, predict_proba)``
to a single scalar per sample:

- ``"set_size"``    : number of classes in the prediction set
                     (analogue of ``"width"``: ↑ = more uncertain)
- ``"credibility"`` : highest p-value across classes
                     (how compatible the most-likely class is with calibration)
- ``"confidence"``  : ``1 − second-highest p-value``
                     (how confidently we reject the runner-up class)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

RegressionMetric = Literal["width", "lower", "upper", "midpoint"]
ClassificationMetric = Literal["set_size", "credibility", "confidence"]
UncertaintyMetric = Literal[
    "width", "lower", "upper", "midpoint",
    "set_size", "credibility", "confidence",
]


METRIC_LABELS: dict[str, str] = {
    "width": "Interval width",
    "lower": "Lower bound",
    "upper": "Upper bound",
    "midpoint": "Interval midpoint",
    "set_size": "Set size",
    "credibility": "Credibility (max p-value)",
    "confidence": "Confidence (1 - 2nd p-value)",
}


_REGRESSION_REDUCERS: dict[
    str, Callable[[np.ndarray, np.ndarray], np.ndarray]
] = {
    "width": lambda lower, upper: upper - lower,
    "lower": lambda lower, _upper: lower,
    "upper": lambda _lower, upper: upper,
    "midpoint": lambda lower, upper: (lower + upper) / 2.0,
}


def _set_size(cp, X: np.ndarray, confidence: float) -> np.ndarray:
    return cp.predict_set(X, confidence=confidence).sum(axis=1).astype(float)


def _credibility(cp, X: np.ndarray, _confidence: float) -> np.ndarray:
    return cp.predict_p(X).max(axis=1)


def _conformal_confidence(cp, X: np.ndarray, _confidence: float) -> np.ndarray:
    p = np.sort(cp.predict_p(X), axis=1)
    # 1 - second-highest p-value (well-defined for n_classes >= 2)
    return 1.0 - p[:, -2]


_CLASSIFICATION_REDUCERS: dict[
    str, Callable[[object, np.ndarray, float], np.ndarray]
] = {
    "set_size": _set_size,
    "credibility": _credibility,
    "confidence": _conformal_confidence,
}


def is_classifier_predictor(cp) -> bool:
    """Return True if ``cp`` exposes the conformal-classifier interface."""

    return hasattr(cp, "predict_set") and hasattr(cp, "predict_p")


def interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Compute interval width."""

    return upper - lower


def metric_label(metric: UncertaintyMetric) -> str:
    """Return a human-readable label for ``metric`` (used by plots)."""

    if metric not in METRIC_LABELS:
        raise ValueError(
            f"Unknown uncertainty metric '{metric}'. "
            f"Choose from {list(METRIC_LABELS)}."
        )
    return METRIC_LABELS[metric]


def make_uncertainty_function(
    conformal_predictor,
    confidence: float = 0.9,
    metric: UncertaintyMetric = "width",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a callable ``X -> uncertainty_metric(X)`` for an explainer.

    Dispatches between regression and classification reducers based on
    the conformal predictor's interface (presence of ``predict_set``).

    Parameters
    ----------
    conformal_predictor
        Fitted conformal predictor (regressor or classifier).
    confidence : float
        Nominal coverage level passed to the conformal predictor.
    metric : str
        - Regression: ``"width"``, ``"lower"``, ``"upper"``, ``"midpoint"``
        - Classification: ``"set_size"``, ``"credibility"``,
          ``"confidence"``

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Function ``f(X) -> np.ndarray`` of shape ``(n_samples,)``.
    """

    if is_classifier_predictor(conformal_predictor):
        if metric not in _CLASSIFICATION_REDUCERS:
            raise ValueError(
                f"Metric '{metric}' is not valid for classification. "
                f"Choose from {list(_CLASSIFICATION_REDUCERS)}."
            )
        reducer = _CLASSIFICATION_REDUCERS[metric]

        def uncertainty_function(X: np.ndarray) -> np.ndarray:
            return reducer(conformal_predictor, X, confidence)

        return uncertainty_function

    if metric not in _REGRESSION_REDUCERS:
        raise ValueError(
            f"Metric '{metric}' is not valid for regression. "
            f"Choose from {list(_REGRESSION_REDUCERS)}."
        )

    reducer_reg = _REGRESSION_REDUCERS[metric]

    def uncertainty_function(X: np.ndarray) -> np.ndarray:
        lower, upper = conformal_predictor.predict(
            X,
            confidence=confidence,
        )
        return reducer_reg(lower, upper)

    return uncertainty_function


def make_interval_width_function(
    conformal_predictor,
    confidence: float = 0.9,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Backwards-compatible alias for ``make_uncertainty_function(..., metric="width")``.
    """

    return make_uncertainty_function(
        conformal_predictor,
        confidence=confidence,
        metric="width",
    )