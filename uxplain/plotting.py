"""
Plotting primitives for uncertainty explanations.

Every figure in \\pkg{uxplain} is drawn here with plain \\pkg{matplotlib} from the
raw arrays produced by the explainers — SHAP values, partial-dependence grids,
LIME coefficients. Nothing is delegated to the plotting layers of ``shap``,
``sklearn.inspection`` or ``lime``, so the caller owns every artist.

Each primitive follows the same contract:

- it accepts an existing ``ax`` (and draws into it) or creates its own figure,
- it returns the ``Axes`` it drew on,
- it takes styling from :data:`STYLE`, which can be mutated globally or
  overridden per call.

That makes the primitives composable::

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    shap_bar(explanation, ax=axes[0])
    shap_beeswarm(explanation, ax=axes[1])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np

from .uncertainty.metrics import metric_label

__all__ = [
    "STYLE",
    "ice_curves",
    "lime_local",
    "pdp_curve",
    "pdp_importance",
    "pdp_interaction",
    "pdp_with_ice",
    "shap_bar",
    "shap_beeswarm",
    "shap_waterfall",
]


#: Global styling. Mutate to restyle every figure at once, or pass the same
#: keys per call to override a single plot. See :func:`use_theme` for the
#: bundled presets.
STYLE: dict[str, Any] = {
    # --- data ink -----------------------------------------------------------
    # One base tonality carries everything. The contrasting hue is spent only
    # where it distinguishes something the reader could not otherwise see:
    # marks that pull the metric in the opposite direction. When every mark in
    # a plot shares a sign, nothing is being distinguished, so the plot stays
    # entirely in the base tone (see `adaptive_sign_color`).
    "magnitude_color": "#3d5a73",  # the base tone
    "positive_color": "#3d5a73",   # same as base: the common direction
    "negative_color": "#b2493f",   # the contrasting tone
    "adaptive_sign_color": True,   # drop the contrast when signs are uniform
    "pdp_color": "#3d5a73",        # a lone line needs no second hue
    "ice_color": "#9aa5ae",        # individual conditional expectation lines
    # Sequential ramp for a continuous feature value: one hue, light -> dark.
    "colormap": ("#dce6ef", "#7ba0c0", "#1f4e79"),
    # --- chrome -------------------------------------------------------------
    "title_color": "#1a1a1a",
    "label_color": "#3f3f3f",
    "tick_color": "#5c5c5c",
    "grid_color": "#dcdcd8",
    "grid_linestyle": "-",
    "grid_linewidth": 0.8,
    "grid_alpha": 1.0,
    "baseline_color": "#b0b0ac",
    "reference_color": "#8c8c88",  # zero line, baseline marker
    # --- typography ---------------------------------------------------------
    "title_size": 17,
    "label_size": 15,
    "tick_size": 13,
    "legend_size": 13,
    "annotation_size": 13,
    "title_weight": "normal",
    "title_loc": "left",
    # --- marks --------------------------------------------------------------
    "point_size": 18,
    "point_alpha": 0.8,
    "bar_edge_color": "none",
    "bar_height": 0.62,
    "line_width": 2.2,
}


#: Bundled presets, applied with :func:`use_theme`.
THEMES: dict[str, dict[str, Any]] = {
    # Austere and print-first: no hue at all, so the figure survives grayscale
    # reproduction. Opposing marks take a lighter step of the same gray.
    "mono": {
        "magnitude_color": "#4a4a4a",
        "positive_color": "#4a4a4a",
        "negative_color": "#a5a5a5",
        "pdp_color": "#2b2b2b",
        "ice_color": "#c4c4c4",
        "colormap": ("#e8e8e8", "#9a9a9a", "#2b2b2b"),
    },
    # One muted slate carries every figure; a terracotta appears only on marks
    # that pull the metric the other way. Colorblind-safe.
    "diverging": {
        "magnitude_color": "#3d5a73",
        "positive_color": "#3d5a73",
        "negative_color": "#b2493f",
        "pdp_color": "#3d5a73",
        "ice_color": "#9aa5ae",
        "colormap": ("#dce6ef", "#7ba0c0", "#1f4e79"),
    },
    # A single editorial hue throughout; opposing marks take a lighter step of
    # that same hue rather than a second one.
    "editorial": {
        "magnitude_color": "#1f4e79",
        "positive_color": "#1f4e79",
        "negative_color": "#9db8cd",
        "pdp_color": "#1f4e79",
        "ice_color": "#c2ccd4",
        "colormap": ("#e4ebf1", "#89a8c2", "#173d5e"),
    },
}


def use_theme(name: str, **overrides) -> dict:
    """Apply a bundled theme to :data:`STYLE` in place.

    Parameters
    ----------
    name : {"mono", "diverging", "editorial"}
        Preset to apply. ``"mono"`` is grayscale-safe, ``"diverging"`` reserves
        hue for the sign of a contribution, ``"editorial"`` keeps a single hue.
    **overrides
        Extra style keys applied on top of the preset.

    Returns
    -------
    dict
        The updated :data:`STYLE`, so it can be inspected or stashed.
    """
    if name not in THEMES:
        raise ValueError(
            f"Unknown theme '{name}'. Choose from {sorted(THEMES)}."
        )
    STYLE.update(THEMES[name])
    STYLE.update(overrides)
    return STYLE


def _style(overrides: dict | None = None) -> dict:
    """Merge per-call overrides over the global style."""
    if not overrides:
        return dict(STYLE)
    merged = dict(STYLE)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def _feature_cmap(style: dict) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("uxplain", list(style["colormap"]))


def _new_ax(ax, figsize):
    """Return ``(fig, ax, created)``; create a figure only when ax is None."""
    if ax is not None:
        return ax.figure, ax, False
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, True


def _finish(ax, style: dict, *, title=None, xlabel=None, ylabel=None,
            grid_axis="x") -> None:
    """Apply the shared look: recessive chrome, muted ink, hairline grid."""
    if title:
        ax.set_title(title, fontsize=style["title_size"],
                     color=style["title_color"], loc=style["title_loc"],
                     fontweight=style["title_weight"], pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=style["label_size"],
                      color=style["label_color"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=style["label_size"],
                      color=style["label_color"])

    ax.tick_params(labelsize=style["tick_size"], colors=style["tick_color"],
                   length=0 if grid_axis else 3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(style["tick_color"])

    if grid_axis:
        ax.grid(True, axis=grid_axis, color=style["grid_color"],
                linestyle=style["grid_linestyle"],
                linewidth=style["grid_linewidth"],
                alpha=style["grid_alpha"], zorder=0)
        ax.set_axisbelow(True)

    # Keep only the axis the categories sit on; the grid carries the rest.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style["baseline_color"])
        ax.spines[side].set_linewidth(0.9)


def _resolve_names(explanation, feature_names, n_features) -> list[str]:
    names = feature_names or getattr(explanation, "feature_names", None)
    if names is None:
        return [f"Feature {i}" for i in range(n_features)]
    return list(names)


def _metric_text(explanation, metric=None) -> str:
    return metric_label(metric or getattr(explanation, "metric", "width"))


def _unpack_shap(explanation):
    """Extract ``(values, data, base_values)`` from a SHAP explanation."""
    values = np.asarray(explanation.values, dtype=float)
    data = getattr(explanation, "data", None)
    if data is not None:
        data = np.asarray(data, dtype=float)
    base = getattr(explanation, "base_values", None)
    if base is not None:
        base = np.asarray(base, dtype=float)
    return values, data, base


def _top_features(importance: np.ndarray, max_display: int | None):
    """Return indices of the ``max_display`` most important features, ascending."""
    order = np.argsort(importance)
    if max_display is not None and max_display < len(order):
        order = order[-max_display:]
    return order


def _sign_colors(values, style: dict) -> list[str]:
    """Colour signed marks, spending the contrasting hue only when it earns it.

    If every value points the same way there is nothing for a second hue to
    distinguish, so the whole plot stays in the base tone and matches the
    unsigned figures beside it.
    """
    values = np.asarray(values, dtype=float)
    mixed = (values > 0).any() and (values < 0).any()
    if style["adaptive_sign_color"] and not mixed:
        return [style["magnitude_color"]] * len(values)
    return [style["positive_color"] if v >= 0 else style["negative_color"]
            for v in values]


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def shap_bar(
    explanation,
    ax=None,
    feature_names: Sequence[str] | None = None,
    max_display: int | None = None,
    figsize: tuple[float, float] = (9, 5),
    show_values: bool = True,
    metric: str | None = None,
    title: str | None = None,
    **style_overrides,
):
    """Mean absolute SHAP contribution per feature, as a horizontal bar chart.

    Parameters
    ----------
    explanation
        Object exposing ``.values`` of shape ``(n_samples, n_features)``.
    ax
        Draw into this ``Axes`` when given; otherwise a new figure is made.
    max_display
        Keep only the this many highest-ranked features.
    show_values
        Annotate each bar with its numeric value.
    """
    style = _style(style_overrides)
    values, _, _ = _unpack_shap(explanation)
    if values.ndim == 1:
        values = values[None, :]
    names = _resolve_names(explanation, feature_names, values.shape[1])

    importance = np.abs(values).mean(axis=0)
    order = _top_features(importance, max_display)

    fig, ax, _ = _new_ax(ax, figsize)
    y = np.arange(len(order))
    bars = ax.barh(y, importance[order], color=style["magnitude_color"],
                   edgecolor=style["bar_edge_color"],
                   height=style["bar_height"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])

    if show_values:
        ax.bar_label(bars, fmt="%.2f", padding=5,
                     fontsize=style["annotation_size"],
                     color=style["label_color"])
        ax.set_xlim(0, importance[order].max() * 1.18)

    _finish(ax, style,
            title=title if title is not None
            else f"Mean absolute contribution to {_metric_text(explanation, metric).lower()}",
            xlabel="mean(|SHAP value|)")
    fig.tight_layout()
    return ax


def _swarm_offsets(vals: np.ndarray, n_bins: int = 100,
                   row_height: float = 0.4) -> np.ndarray:
    """Vertical offsets that spread overlapping points into a beeswarm.

    Points are bucketed along the x axis; within a bucket they are stacked
    outwards, alternating above and below the row centre.
    """
    n = len(vals)
    if n == 0:
        return np.zeros(0)
    span = vals.max() - vals.min()
    if span <= 0:
        return np.zeros(n)

    quant = np.round(n_bins * (vals - vals.min()) / span)
    order = np.argsort(quant + np.random.RandomState(0).randn(n) * 1e-6)

    offsets = np.zeros(n)
    layer, last_bin = 0, -1
    for idx in order:
        if quant[idx] != last_bin:
            layer = 0
        # 0, +1, -1, +2, -2, ... away from the row centre
        offsets[idx] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[idx]

    peak = np.abs(offsets).max()
    if peak > 0:
        offsets *= 0.9 * (row_height / peak)
    return offsets


def shap_beeswarm(
    explanation,
    X=None,
    ax=None,
    feature_names: Sequence[str] | None = None,
    max_display: int | None = None,
    figsize: tuple[float, float] = (9, 5),
    colorbar: bool = True,
    metric: str | None = None,
    title: str | None = None,
    **style_overrides,
):
    """Per-instance SHAP contributions, one swarm per feature.

    Point colour encodes the feature's own value, so the plot shows both how
    much a feature moves the metric and in which direction high/low values push.
    """
    style = _style(style_overrides)
    values, data, _ = _unpack_shap(explanation)
    if X is not None:
        data = np.asarray(X, dtype=float)
    names = _resolve_names(explanation, feature_names, values.shape[1])

    importance = np.abs(values).mean(axis=0)
    order = _top_features(importance, max_display)

    fig, ax, _ = _new_ax(ax, figsize)
    cmap = _feature_cmap(style)

    for row, feat in enumerate(order):
        v = values[:, feat]
        ys = row + _swarm_offsets(v)
        if data is not None:
            col = data[:, feat]
            # Robust scaling so a few outliers don't flatten the colour range.
            lo, hi = np.nanpercentile(col, [5, 95])
            norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
            ax.scatter(v, ys, c=col, cmap=cmap, norm=norm,
                       s=style["point_size"], linewidths=0,
                       alpha=style["point_alpha"], zorder=3)
        else:
            ax.scatter(v, ys, color=style["magnitude_color"],
                       s=style["point_size"], linewidths=0,
                       alpha=style["point_alpha"], zorder=3)

    ax.axvline(0, color=style["reference_color"], linewidth=0.9, zorder=2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_ylim(-0.6, len(order) - 0.4)

    if colorbar and data is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=32, fraction=0.035)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["low", "high"])
        cbar.ax.tick_params(labelsize=style["tick_size"], length=0,
                            colors=style["tick_color"])
        cbar.set_label("Feature value", fontsize=style["label_size"],
                       color=style["label_color"])
        cbar.outline.set_visible(False)

    metric_txt = _metric_text(explanation, metric).lower()
    _finish(ax, style,
            title=title if title is not None
            else f"Per-instance contribution to {metric_txt}",
            xlabel=f"SHAP value (effect on {metric_txt})",
            grid_axis="x")
    fig.tight_layout()
    return ax


def shap_waterfall(
    explanation,
    index: int = 0,
    ax=None,
    feature_names: Sequence[str] | None = None,
    max_display: int = 10,
    figsize: tuple[float, float] = (10, 6),
    metric: str | None = None,
    title: str | None = None,
    **style_overrides,
):
    """Additive decomposition of a single instance's metric value.

    Bars run cumulatively from the baseline ``E[f(X)]`` to the instance's
    ``f(x)``; features are ordered by absolute contribution and any remainder is
    collapsed into a single "other features" bar.
    """
    style = _style(style_overrides)
    values, data, base = _unpack_shap(explanation)
    if values.ndim == 1:
        values = values[None, :]
    names = _resolve_names(explanation, feature_names, values.shape[1])

    phi = values[index]
    base_value = 0.0
    if base is not None:
        base_value = float(base[index]) if base.ndim else float(base)
    row = data[index] if data is not None else None

    order = np.argsort(np.abs(phi))[::-1]
    shown, hidden = order[:max_display], order[max_display:]

    labels, contribs = [], []
    for feat in shown:
        label = names[feat]
        if row is not None:
            label = f"{row[feat]:.3g} = {label}"
        labels.append(label)
        contribs.append(phi[feat])
    if len(hidden):
        labels.append(f"{len(hidden)} other features")
        contribs.append(float(phi[hidden].sum()))

    contribs = np.asarray(contribs)
    # Draw smallest contribution at the top of the axis, largest at the bottom.
    labels, contribs = labels[::-1], contribs[::-1]

    starts = base_value + np.concatenate([[0.0], np.cumsum(contribs)[:-1]])
    total = base_value + contribs.sum()

    fig, ax, _ = _new_ax(ax, figsize)
    y = np.arange(len(contribs))
    colors = _sign_colors(contribs, style)
    ax.barh(y, contribs, left=starts, color=colors,
            edgecolor=style["bar_edge_color"],
            height=style["bar_height"], zorder=3)

    span = max(np.abs(contribs).max(), abs(total - base_value), 1e-9)
    pad = 0.02 * span
    for yi, (start, c) in enumerate(zip(starts, contribs)):
        tip = start + c
        ax.text(tip + (pad if c >= 0 else -pad), yi, f"{c:+.2f}",
                va="center", ha="left" if c >= 0 else "right",
                fontsize=style["annotation_size"], color=style["label_color"])

    ax.axvline(base_value, color=style["reference_color"],
               linestyle="--", linewidth=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    # Reserve a header band inside the axes for the baseline/outcome labels, so
    # they never collide with the title above or the tick labels below.
    header = len(contribs) - 0.3 + 0.7
    ax.set_ylim(-0.7, header)

    metric_txt = _metric_text(explanation, metric)
    _finish(ax, style,
            title=title if title is not None
            else f"How {metric_txt.lower()} is built up for sample {index}",
            xlabel=metric_txt)

    label_y = header - 0.25
    ax.text(base_value, label_y, f"$E[f(X)]$ = {base_value:.2f}",
            ha="center", va="center", fontsize=style["annotation_size"],
            color=style["tick_color"])
    ax.text(total, label_y, f"$f(x)$ = {total:.2f}",
            ha="center", va="center", fontsize=style["annotation_size"],
            color=style["title_color"])

    lo = min(starts.min(), base_value, total)
    hi = max((starts + contribs).max(), base_value, total)
    ax.set_xlim(lo - 0.22 * span, hi + 0.22 * span)
    fig.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Partial dependence
# ---------------------------------------------------------------------------

def pdp_curve(
    grid: np.ndarray,
    values: np.ndarray,
    ax=None,
    feature_name: str = "",
    figsize: tuple[float, float] = (8, 6),
    metric_name: str = "Interval width",
    title: str | None = None,
    label: str | None = "PDP",
    **style_overrides,
):
    """Averaged partial-dependence curve for one feature."""
    style = _style(style_overrides)
    fig, ax, _ = _new_ax(ax, figsize)
    ax.plot(grid, values, color=style["pdp_color"],
            lw=style["line_width"], label=label, zorder=5)
    _finish(ax, style,
            title=title if title is not None
            else f"Average {metric_name.lower()} across {feature_name}",
            xlabel=feature_name, ylabel=metric_name, grid_axis="both")
    fig.tight_layout()
    return ax


def ice_curves(
    grid: np.ndarray,
    individual: np.ndarray,
    ax=None,
    feature_name: str = "",
    figsize: tuple[float, float] = (8, 6),
    metric_name: str = "Interval width",
    max_lines: int = 80,
    title: str | None = None,
    **style_overrides,
):
    """Individual conditional expectation curves, one per observation."""
    style = _style(style_overrides)
    fig, ax, _ = _new_ax(ax, figsize)

    lines = np.asarray(individual)
    if len(lines) > max_lines:
        idx = np.random.RandomState(0).choice(len(lines), max_lines, replace=False)
        lines = lines[idx]
    alpha = float(np.clip(25 / max(len(lines), 1), 0.10, 0.45))
    for line in lines:
        ax.plot(grid, line, color=style["ice_color"], lw=0.7,
                alpha=alpha, zorder=3)

    _finish(ax, style,
            title=title if title is not None
            else f"{metric_name} per observation across {feature_name}",
            xlabel=feature_name, ylabel=metric_name, grid_axis="both")
    fig.tight_layout()
    return ax


def pdp_with_ice(
    grid: np.ndarray,
    values: np.ndarray,
    individual: np.ndarray,
    ax=None,
    feature_name: str = "",
    figsize: tuple[float, float] = (8, 6),
    metric_name: str = "Interval width",
    max_lines: int = 80,
    title: str | None = None,
    **style_overrides,
):
    """ICE curves with the averaged partial-dependence curve on top."""
    style = _style(style_overrides)
    fig, ax, _ = _new_ax(ax, figsize)
    ice_curves(grid, individual, ax=ax, feature_name=feature_name,
               metric_name=metric_name, max_lines=max_lines,
               title="", **style_overrides)
    ax.plot(grid, values, color=style["pdp_color"],
            lw=style["line_width"] + 0.8, label="Average", zorder=6)
    legend = ax.legend(fontsize=style["legend_size"], frameon=False,
                       loc="best")
    for text in legend.get_texts():
        text.set_color(style["label_color"])
    _finish(ax, style,
            title=title if title is not None
            else f"Average {metric_name.lower()} across {feature_name}",
            xlabel=feature_name, ylabel=metric_name, grid_axis="both")
    fig.tight_layout()
    return ax


def pdp_importance(
    importance: Sequence[float],
    feature_names: Sequence[str],
    ax=None,
    figsize: tuple[float, float] = (8, 5),
    metric_name: str = "Interval width",
    title: str | None = None,
    **style_overrides,
):
    """Global PDP-based importance: the spread of each feature's PDP curve."""
    style = _style(style_overrides)
    imp = np.asarray(importance, dtype=float)
    order = np.argsort(imp)

    fig, ax, _ = _new_ax(ax, figsize)
    y = np.arange(len(order))
    bars = ax.barh(y, imp[order], color=style["magnitude_color"],
                   edgecolor=style["bar_edge_color"],
                   height=style["bar_height"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=style["annotation_size"],
                 color=style["label_color"])
    ax.set_xlim(0, max(imp.max() * 1.18, 1e-9))

    _finish(ax, style,
            title=title if title is not None
            else f"How strongly each feature moves {metric_name.lower()}",
            xlabel=f"PDP importance (std of the {metric_name.lower()} curve)")
    fig.tight_layout()
    return ax


def pdp_interaction(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    values: np.ndarray,
    ax=None,
    feature_names: tuple[str, str] = ("", ""),
    figsize: tuple[float, float] = (8, 6),
    metric_name: str = "Interval width",
    levels: int = 14,
    title: str | None = None,
    **style_overrides,
):
    """Two-way partial dependence as a filled contour map."""
    style = _style(style_overrides)
    fig, ax, _ = _new_ax(ax, figsize)

    # partial_dependence returns (len(grid_x), len(grid_y)); transpose for
    # contourf, which expects (n_rows=y, n_cols=x).
    Z = np.asarray(values).T
    cs = ax.contourf(grid_x, grid_y, Z, levels=levels, cmap=_feature_cmap(style))
    cbar = fig.colorbar(cs, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label(metric_name, fontsize=style["label_size"],
                   color=style["label_color"])
    cbar.ax.tick_params(labelsize=style["tick_size"], length=0,
                        colors=style["tick_color"])
    cbar.outline.set_visible(False)

    _finish(ax, style,
            title=title if title is not None
            else f"{metric_name} across {feature_names[0]} and {feature_names[1]}",
            xlabel=feature_names[0], ylabel=feature_names[1], grid_axis=None)
    fig.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def lime_local(
    coefficients: np.ndarray,
    feature_names: Sequence[str],
    ax=None,
    figsize: tuple[float, float] = (9, 5),
    max_display: int | None = None,
    metric_name: str = "Interval width",
    sample_index: int | None = None,
    title: str | None = None,
    **style_overrides,
):
    """Local LIME surrogate coefficients for a single instance."""
    style = _style(style_overrides)
    coefs = np.asarray(coefficients, dtype=float)
    order = _top_features(np.abs(coefs), max_display)

    fig, ax, _ = _new_ax(ax, figsize)
    y = np.arange(len(order))
    colors = _sign_colors(coefs[order], style)
    ax.barh(y, coefs[order], color=colors,
            edgecolor=style["bar_edge_color"],
            height=style["bar_height"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.axvline(0, color=style["reference_color"], linewidth=0.9, zorder=2)

    if title is None:
        title = f"Local drivers of {metric_name.lower()}"
        if sample_index is not None:
            title += f" for sample {sample_index}"
    _finish(ax, style, title=title,
            xlabel=f"LIME coefficient (effect on {metric_name.lower()})")
    fig.tight_layout()
    return ax
