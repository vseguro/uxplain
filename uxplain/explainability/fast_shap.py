"""
Exact TreeSHAP shortcut for conformal summaries that are linear in the
component models.

The generic path hands ``shap.Explainer`` a plain callable — the composition of
the base model(s) with the conformal calibration. Because SHAP cannot see that a
tree ensemble sits inside that callable, it falls back to Exact or permutation
explainers, whose cost grows with the number of features and the size of the
background set.

Several conformal summaries are, however, *affine* in the component models'
predictions. Writing the summary as

.. math::

    u(x) = \\sum_k w_k\\, m_k(x) + c,

with fixed weights :math:`w_k` and a constant :math:`c`, and using the linearity
of Shapley values in the payoff function, gives

.. math::

    \\phi_j\\big(u, x\\big) = \\sum_k w_k\\, \\phi_j\\big(m_k, x\\big).

So the attributions can be obtained by running exact TreeSHAP on each component
model and combining the results, which is orders of magnitude cheaper than
KernelSHAP over the composed function. The constant :math:`c` (the calibration
offset) shifts only the base value, never the attributions.

Eligible combinations:

===================================  ===================================
Predictor / metric                   Decomposition
===================================  ===================================
CQR, ``"width"``                     ``upper_model - lower_model``
CQR, ``"midpoint"``                  ``(lower_model + upper_model) / 2``
CQR, ``"lower"`` / ``"upper"``       the corresponding quantile model
crepes standard, ``"lower"``,        the base model (the calibration
``"upper"``, ``"midpoint"``          offset is constant)
crepes standard, ``"width"``         constant — all attributions are zero
===================================  ===================================

``"normalized"`` and Mondrian variants scale the interval by a difficulty
estimator or bin the calibration set, so the summary stops being affine in a
tree model and the shortcut does not apply.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tree_shap_values", "describe_fast_path"]

# Relative tolerance for the residual check that verifies the affine model.
_LINEARITY_RTOL = 1e-6


def _components(cp, metric: str):
    """Return ``[(model, weight), ...]`` for an affine summary, or ``None``.

    ``None`` means this predictor/metric pair has no affine decomposition and
    the caller must fall back to the generic explainer.
    """
    lower_model = getattr(cp, "lower_model", None)
    upper_model = getattr(cp, "upper_model", None)

    # --- CQR: interval endpoints are two independent quantile models ---
    if lower_model is not None and upper_model is not None:
        return {
            "width": [(upper_model, 1.0), (lower_model, -1.0)],
            "midpoint": [(lower_model, 0.5), (upper_model, 0.5)],
            "lower": [(lower_model, 1.0)],
            "upper": [(upper_model, 1.0)],
        }.get(metric)

    # --- crepes-style: a single base model with a calibration offset ---
    #
    # Which metrics stay affine depends on the method: "standard" shifts both
    # endpoints by the same constant (so all four metrics qualify), whereas
    # "normalized" scales the offset by a difficulty estimator, leaving only the
    # midpoint affine — and only when the interval is symmetric. Rather than
    # enumerate those cases, we propose the decomposition and let the residual
    # check in :func:`tree_shap_values` reject whatever does not hold.
    model = getattr(cp, "model", None)
    if model is None:
        return None
    return {
        # width drops the base model entirely; it qualifies only when the
        # calibration offset is the same for every sample.
        "width": [],
        "midpoint": [(model, 1.0)],
        "lower": [(model, 1.0)],
        "upper": [(model, 1.0)],
    }.get(metric)


def _tree_explainer(model, background):
    """Build a ``shap.TreeExplainer``, or return ``None`` if unsupported."""
    import shap

    try:
        return shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
        )
    except Exception:
        # Not a tree ensemble SHAP recognises (linear model, SVM, pipeline, ...).
        return None


def tree_shap_values(cp, metric, X, background, target_fn):
    """
    Compute SHAP values for ``metric`` via exact TreeSHAP on component models.

    Parameters
    ----------
    cp
        Fitted conformal predictor.
    metric : str
        Scalar summary being explained.
    X : np.ndarray
        Samples to explain.
    background : np.ndarray
        Reference data defining the baseline expectation.
    target_fn : callable
        The true ``X -> metric`` function, used to verify the decomposition and
        to recover the calibration offset.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or None
        ``(values, base_values)`` with shapes ``(n_samples, n_features)`` and
        ``(n_samples,)``. ``None`` when the shortcut does not apply, in which
        case the caller should use the generic explainer.
    """
    components = _components(cp, metric)
    if components is None:
        return None

    X = np.asarray(X, dtype=float)
    background = np.asarray(background, dtype=float)

    # An empty component list means the summary is claimed to be constant (e.g.
    # the width of a standard conformal interval). It flows through the same
    # path: the affine part is zero, every attribution is zero, and the residual
    # check below confirms the value really is constant.
    explainers = []
    for model, weight in components:
        expl = _tree_explainer(model, background)
        if expl is None:
            return None
        explainers.append((expl, weight))

    values = np.zeros_like(X)
    base = 0.0
    linear_part = np.zeros(len(X))
    for expl, weight in explainers:
        phi = np.asarray(expl.shap_values(X, check_additivity=False), dtype=float)
        if phi.ndim != 2 or phi.shape != X.shape:
            return None
        values += weight * phi
        base += weight * float(np.mean(np.atleast_1d(expl.expected_value)))
        linear_part += weight * np.asarray(
            expl.model.predict(X), dtype=float
        ).ravel()

    # The calibration offset is whatever the true summary adds on top of the
    # affine part. It must be constant across samples; if it is not, the
    # decomposition is wrong for this predictor and we refuse to use it.
    actual = np.asarray(target_fn(X), dtype=float).ravel()
    offset = actual - linear_part
    scale = max(np.abs(actual).max(), 1.0)
    if offset.std() > _LINEARITY_RTOL * scale:
        return None

    base_values = np.full(len(X), base + float(offset.mean()))
    return values, base_values


def describe_fast_path(cp, metric: str) -> str:
    """One-line description of whether the shortcut applies, for logs and docs."""
    components = _components(cp, metric)
    if components is None:
        return f"metric '{metric}': no affine decomposition, using generic SHAP"
    if not components:
        return (
            f"metric '{metric}': candidate constant summary — attributions are "
            "zero if the calibration offset does not vary across samples"
        )
    return (
        f"metric '{metric}': candidate affine decomposition over "
        f"{len(components)} component model(s), eligible for exact TreeSHAP"
    )
