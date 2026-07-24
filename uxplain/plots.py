"""
Figure builders for SHAP, PDP and LIME uncertainty explanations.

These functions assemble complete figures (single panels or grids) out of the
primitives in :mod:`uxplain.plotting`, which draw everything with plain
``matplotlib``. Use the primitives directly when you want to place a plot in an
``Axes`` you already control; use the ``generate_*_plots`` builders here when you
just want the standard figure for a given explanation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .plotting import (
    STYLE,
    ice_curves,
    lime_local,
    pdp_curve,
    pdp_importance,
    pdp_interaction,
    pdp_with_ice,
    shap_bar,
    shap_beeswarm,
    shap_waterfall,
)
from .uncertainty.metrics import metric_label

VALID_SHAP_KINDS = ("beeswarm", "bar", "waterfall", "summary")
VALID_LIME_KINDS = ("local",)
VALID_PDP_KINDS = ("pdp", "ice", "pdp_ice", "pdp_2d", "importance")


def _validate_kinds(kinds, valid, method):
    if isinstance(kinds, str):
        kinds = [kinds]
    invalid = [k for k in kinds if k not in valid]
    if invalid:
        raise ValueError(
            f"Unknown plot kind(s) {invalid} for {method}. "
            f"Choose from {valid}."
        )
    return list(kinds)


def _close_new_figures(pre_existing: set[int]) -> None:
    """Close every figure created since the ``pre_existing`` snapshot."""
    for fnum in set(plt.get_fignums()) - pre_existing:
        plt.close(fnum)


def _grid(n_panels: int, figsize, per_panel=(5.5, 4.2), max_cols: int = 3):
    """Create a subplot grid sized for ``n_panels``."""
    ncols = min(max_cols, max(n_panels, 1))
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (per_panel[0] * ncols, per_panel[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    return fig, axes[:n_panels]


def generate_shap_plots(
    explanation,
    X=None,
    kinds: list[str] | None = None,
    feature_names: list[str] | None = None,
    waterfall_index: int | None = None,
    figsize: tuple[float, float] = (9, 5),
    show: bool = True,
    max_display: int | None = None,
):
    """
    Build the standard SHAP figures for an uncertainty explanation.

    Parameters
    ----------
    explanation
        SHAP explanation with ``.values`` and, ideally, ``.data`` and
        ``.base_values``.
    X : np.ndarray, optional
        Feature values used to colour the beeswarm. Falls back to
        ``explanation.data``.
    kinds : list of str, optional
        Any of ``"bar"``, ``"beeswarm"``, ``"waterfall"``, ``"summary"``
        (an alias of ``"beeswarm"``). Defaults to all three main kinds.
    waterfall_index : int, optional
        Sample to decompose. When ``None``, the instance with the largest
        total absolute contribution is chosen.
    max_display : int, optional
        Cap on how many features each plot shows.

    Returns
    -------
    dict
        Maps each kind to its ``matplotlib`` figure.
    """

    if kinds is None:
        kinds = ["beeswarm", "bar", "waterfall"]
    kinds = _validate_kinds(kinds, VALID_SHAP_KINDS, "SHAP")

    pre_figs = set(plt.get_fignums())

    if feature_names is not None:
        explanation.feature_names = feature_names

    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]

    auto_selected = waterfall_index is None
    if auto_selected:
        waterfall_index = int(np.abs(values).sum(axis=1).argmax())

    figures = {}

    if "bar" in kinds:
        ax = shap_bar(explanation, feature_names=feature_names,
                      max_display=max_display, figsize=figsize)
        figures["bar"] = ax.figure

    # "summary" is kept as an alias of "beeswarm"; when both are requested each
    # gets its own figure so callers can style them independently.
    for key in ("beeswarm", "summary"):
        if key in kinds:
            ax = shap_beeswarm(explanation, X=X, feature_names=feature_names,
                               max_display=max_display, figsize=figsize)
            figures[key] = ax.figure

    if "waterfall" in kinds:
        title = None
        if auto_selected:
            metric_txt = metric_label(getattr(explanation, "metric", "width"))
            title = (f"Highest-contribution instance (sample {waterfall_index})"
                     f" — {metric_txt}")
        ax = shap_waterfall(explanation, index=waterfall_index,
                            feature_names=feature_names,
                            max_display=max_display or 10,
                            figsize=figsize, title=title)
        figures["waterfall"] = ax.figure

    if show:
        plt.show()
        _close_new_figures(pre_figs)

    return figures


generate_default_plots = generate_shap_plots


def generate_lime_plots(
    explanation,
    kinds: list[str] | None = None,
    sample_index: int = 0,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    max_display: int | None = None,
) -> dict:
    """
    Build LIME figures from a ``LIMEExplanation``.

    Parameters
    ----------
    explanation : LIMEExplanation
        Output of ``LimeUncertaintyExplainer.explain()``.
    kinds : list of str, optional
        Only ``"local"`` is available. Defaults to ``["local"]``.
    sample_index : int
        Row of ``local_coefficients`` to plot.
    """

    if kinds is None:
        kinds = ["local"]
    kinds = _validate_kinds(kinds, VALID_LIME_KINDS, "LIME")

    pre_figs = set(plt.get_fignums())

    if sample_index is None:
        sample_index = 0

    metric_name = metric_label(getattr(explanation, "metric", "width"))
    names = explanation.feature_names
    figures = {}

    if figsize is None:
        figsize = (9, max(4.0, 0.55 * len(names)))

    if "local" in kinds:
        coefs = np.asarray(explanation.local_coefficients)[sample_index]
        ax = lime_local(coefs, names, figsize=figsize,
                        max_display=max_display, metric_name=metric_name,
                        sample_index=sample_index)
        figures["local"] = ax.figure

    if show:
        plt.show()
        _close_new_figures(pre_figs)

    return figures


_PDP_KIND_REQUIREMENTS = {
    "pdp": ("average", "both"),
    "ice": ("individual", "both"),
    "pdp_ice": ("both",),
}


def _validate_pdp_kinds(kinds: list[str], explanation_kind: str) -> None:
    for plot_kind in kinds:
        allowed = _PDP_KIND_REQUIREMENTS.get(plot_kind)
        if allowed is None:
            continue
        if explanation_kind not in allowed:
            allowed_str = " or ".join(f'"{k}"' for k in allowed)
            raise ValueError(
                f'plot_kind="{plot_kind}" requires the explainer to be fitted '
                f'with kind={allowed_str}, but got kind="{explanation_kind}".'
            )


def generate_pdp_plots(
    explanation,
    kinds: list[str] | None = None,
    feature_names: list[str] | None = None,
    max_ice_lines: int = 80,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> dict:
    """
    Build partial-dependence figures from a ``PDPExplanation``.

    Parameters
    ----------
    explanation : PDPExplanation
        Output of ``PDPUncertaintyExplainer.explain()``.
    kinds : list of str, optional
        Any of ``"pdp"``, ``"ice"``, ``"pdp_ice"``, ``"pdp_2d"``,
        ``"importance"``.
    max_ice_lines : int
        Cap on ICE curves drawn per feature.
    figsize : tuple, optional
        Size of the whole figure. Scales with the panel count when ``None``.

    Returns
    -------
    dict
        Maps each kind to its ``matplotlib`` figure.
    """

    if kinds is None:
        has_1d = len(explanation.features) > 0
        if explanation.kind == "individual":
            kinds = ["ice"] if has_1d else []
        elif explanation.kind == "both":
            kinds = ["pdp_ice"] if has_1d else []
        else:
            kinds = ["pdp"] if has_1d else []

    kinds = _validate_kinds(kinds, VALID_PDP_KINDS, "PDP")
    _validate_pdp_kinds(kinds, explanation.kind)

    pre_figs = set(plt.get_fignums())

    if feature_names is not None:
        explanation.feature_names = feature_names

    metric_name = metric_label(getattr(explanation, "metric", "width"))
    n = len(explanation.features)
    figures = {}

    def _fname(i: int) -> str:
        feat_idx = explanation.features[i]
        names = explanation.feature_names
        if names is not None and feat_idx < len(names):
            return names[feat_idx]
        return f"Feature {feat_idx}"

    def _suptitle(fig, text):
        # Only label the figure as a whole when it holds more than one panel.
        if len(fig.axes) > 1:
            fig.suptitle(text, fontsize=STYLE["title_size"] + 2)
            fig.tight_layout()

    if "pdp" in kinds and n:
        fig, axes = _grid(n, figsize)
        for i, ax in enumerate(axes):
            pdp_curve(explanation.grid_values[i], explanation.values[i], ax=ax,
                      feature_name=_fname(i), metric_name=metric_name,
                      title=f"PDP — {_fname(i)}" if n > 1 else None)
        _suptitle(fig, f"Partial dependence — {metric_name}")
        figures["pdp"] = fig

    if "ice" in kinds and n and explanation.individual is not None:
        fig, axes = _grid(n, figsize)
        for i, ax in enumerate(axes):
            ice_curves(explanation.grid_values[i], explanation.individual[i],
                       ax=ax, feature_name=_fname(i), metric_name=metric_name,
                       max_lines=max_ice_lines,
                       title=f"ICE — {_fname(i)}" if n > 1 else None)
        _suptitle(fig, f"ICE — {metric_name}")
        figures["ice"] = fig

    if "pdp_ice" in kinds and n and explanation.individual is not None:
        fig, axes = _grid(n, figsize)
        for i, ax in enumerate(axes):
            pdp_with_ice(explanation.grid_values[i], explanation.values[i],
                         explanation.individual[i], ax=ax,
                         feature_name=_fname(i), metric_name=metric_name,
                         max_lines=max_ice_lines,
                         title=f"PDP + ICE — {_fname(i)}" if n > 1 else None)
        _suptitle(fig, f"PDP + ICE — {metric_name}")
        figures["pdp_ice"] = fig

    if "importance" in kinds and n:
        importance = explanation.importance
        if importance is None:
            importance = [float(np.std(v)) for v in explanation.values]
        ax = pdp_importance(importance, [_fname(i) for i in range(n)],
                            figsize=figsize or (8, max(3.0, 0.6 * n + 1.5)),
                            metric_name=metric_name)
        figures["importance"] = ax.figure

    if "pdp_2d" in kinds:
        pairs = explanation.feature_pairs or []
        if not pairs or explanation.values_2d is None:
            raise ValueError(
                'plot_kind="pdp_2d" requires the explainer to be fit with '
                "at least one feature pair as a tuple, e.g. features=[(0, 1)]. "
                "No 2D partial dependence was computed."
            )
        names = explanation.feature_names
        fig, axes = _grid(len(pairs), figsize, per_panel=(6.5, 5.0), max_cols=2)
        for i, ax in enumerate(axes):
            gx, gy = explanation.grid_values_2d[i]
            fa, fb = pairs[i]
            labels = (
                names[fa] if names is not None and fa < len(names) else f"Feature {fa}",
                names[fb] if names is not None and fb < len(names) else f"Feature {fb}",
            )
            pdp_interaction(gx, gy, explanation.values_2d[i], ax=ax,
                            feature_names=labels, metric_name=metric_name,
                            title=f"{labels[0]} × {labels[1]}"
                            if len(pairs) > 1 else None)
        _suptitle(fig, f"Two-way partial dependence — {metric_name}")
        figures["pdp_2d"] = fig

    if show:
        plt.show()
        _close_new_figures(pre_figs)

    return figures
