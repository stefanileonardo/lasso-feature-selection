"""Figures of the experiments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Colours: light surface, dark text, recessive grid, colour-blind-safe series.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _new_figure(n_rows=1, figsize=(7.5, 4.2), sharex=False):
    """Create a figure whose axes share the same quiet style."""
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=sharex, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[:, 0]:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(which="both", color=AXIS, labelcolor=INK_MUTED, labelsize=9)
        ax.xaxis.label.set_color(INK_SECONDARY)
        ax.yaxis.label.set_color(INK_SECONDARY)
    return fig, axes[:, 0]


def _title(ax, text):
    # The padding leaves room between title and plot for the reference labels.
    ax.set_title(text, loc="left", color=INK, fontsize=11.5, fontweight="bold", pad=36)


def _legend(ax, **kwargs):
    legend = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return legend


def _axis_position(ax, x):
    """Horizontal position of ``x`` inside the axes (0 = left edge, 1 = right edge)."""
    low, high = ax.get_xlim()
    if ax.get_xscale() == "log":
        return np.log(x / low) / np.log(high / low)
    return (x - low) / (high - low)


def _mark(ax, x, label, row=0, text_on_left=None):
    """Vertical reference line at ``x``, labelled just above the plot area.

    The label is written beside the top of the line, on the side where there
    is more room unless ``text_on_left`` is given. Call it after the data
    have been plotted, so that the axis limits are final.
    """
    if text_on_left is None:
        text_on_left = _axis_position(ax, x) > 0.6
    height = 4 + 12 * row  # points above the plot area

    ax.axvline(x, color=INK_MUTED, linewidth=1.0, zorder=1)
    where = dict(xy=(x, 1.0), xycoords=("data", "axes fraction"), textcoords="offset points")
    ax.annotate("", xytext=(0, height + 9), **where,  # leader from the plot to the label
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1.0,
                                shrinkA=0, shrinkB=0))
    ax.annotate(label, xytext=(-4 if text_on_left else 4, height), **where,
                ha="right" if text_on_left else "left", va="bottom",
                fontsize=8.5, color=INK_SECONDARY)


def _mark_alphas(ax, alpha_min=None, alpha_selected=None, selected_label="selected"):
    """Reference lines at the alphas chosen on the validation curve."""
    marks = [(alpha_min, "lowest validation error"), (alpha_selected, selected_label)]
    marks = sorted((alpha, label) for alpha, label in marks if alpha is not None)

    if len(marks) == 2 and np.isclose(marks[0][0], marks[1][0]):
        marks = [(marks[0][0], f"lowest validation error = {selected_label}")]
    if len(marks) == 1:
        _mark(ax, *marks[0])
    elif len(marks) == 2:
        # Two labels: both on the same side of their lines and on different
        # rows, the outer line on the upper row, so that no label crosses the
        # leader of the other one.
        (left_x, left_label), (right_x, right_label) = marks
        text_on_left = _axis_position(ax, np.sqrt(left_x * right_x)) > 0.6
        _mark(ax, left_x, left_label, row=0 if text_on_left else 1, text_on_left=text_on_left)
        _mark(ax, right_x, right_label, row=1 if text_on_left else 0, text_on_left=text_on_left)


def _save(fig, path):
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
    return fig


def plot_cv_curve(cv, alpha_min=None, alpha_selected=None, selected_label="selected",
                  title="Cross-validation error", path=None):
    """Training and validation error as a function of ``alpha``.

    The shaded band is one standard error of the validation error across the
    folds, on each side.
    """
    fig, (ax,) = _new_figure()

    ax.plot(cv.alphas, cv.train_errors, color=SERIES[0], linewidth=2, label="Training")
    ax.plot(cv.alphas, cv.val_errors, color=SERIES[1], linewidth=2, label="Validation",
            marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.fill_between(cv.alphas, cv.val_errors - cv.val_std_errors,
                    cv.val_errors + cv.val_std_errors, color=SERIES[1], alpha=0.10, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("Regularisation strength alpha (log scale)")
    ax.set_ylabel("Mean squared error")
    _mark_alphas(ax, alpha_min, alpha_selected, selected_label)
    _title(ax, title)
    _legend(ax, loc="center left")
    return _save(fig, path)


def plot_coefficient_path(alphas, betas, feature_names, selected=None,
                          alpha_min=None, alpha_selected=None, selected_label="selected",
                          title="Lasso coefficient path", path=None):
    """Coefficient of every feature as a function of ``alpha``.

    Reading the figure from right to left (decreasing regularisation), the
    features enter the model one after the other, the most important first.

    With at most 8 features every feature has its own colour; with more, the
    features in ``selected`` are highlighted and the others are drawn in grey.
    """
    n_features = betas.shape[1]
    fig, (ax,) = _new_figure(figsize=(7.5, 4.6))
    ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)

    if n_features <= len(SERIES):
        for j in range(n_features):
            ax.plot(alphas, betas[:, j], color=SERIES[j], linewidth=2, label=feature_names[j])
        _legend(ax, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:
        is_selected = np.zeros(n_features, dtype=bool)
        if selected is not None:
            is_selected[np.asarray(selected)] = True
        for j in np.flatnonzero(~is_selected):
            ax.plot(alphas, betas[:, j], color=AXIS, linewidth=1.2, zorder=2)
        for j in np.flatnonzero(is_selected):
            ax.plot(alphas, betas[:, j], color=SERIES[0], linewidth=2, zorder=3)
        handles = [Line2D([], [], color=SERIES[0], linewidth=2),
                   Line2D([], [], color=AXIS, linewidth=1.2)]
        labels = [f"Selected features ({is_selected.sum()})",
                  f"Discarded features ({n_features - is_selected.sum()})"]
        _legend(ax, handles=handles, labels=labels, loc="best")

    ax.set_xscale("log")
    ax.set_xlabel("Regularisation strength alpha (log scale)")
    ax.set_ylabel("Coefficient (standardised features)")
    _mark_alphas(ax, alpha_min, alpha_selected, selected_label)
    _title(ax, title)
    return _save(fig, path)


def plot_tradeoff(rows, task="regression", n_selected=None, selected_label="selected",
                  title="Effect of the number of selected features", path=None):
    """Test performance and training time against the number of features.

    ``rows`` is the output of :func:`src.feature_selection.selection_sweep`.
    The two quantities have different units, so each one has its own panel.
    """
    n_features = np.array([row["n_features"] for row in rows])
    seconds = np.array([row["time"] for row in rows])
    if task == "classification":
        score = 100.0 * np.array([row["accuracy"] for row in rows])
        score_label = "Test accuracy (%)"
    else:
        score = np.array([row["mse"] for row in rows])
        score_label = "Test mean squared error"

    fig, (ax_score, ax_time) = _new_figure(n_rows=2, figsize=(7.5, 5.6), sharex=True)
    marker = dict(marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax_score.plot(n_features, score, color=SERIES[0], linewidth=2, **marker)
    ax_time.plot(n_features, seconds, color=SERIES[0], linewidth=2, **marker)

    ax_score.set_ylabel(score_label)
    ax_time.set_ylabel("Training time (s)")
    ax_time.set_ylim(bottom=0.0)
    ax_time.set_xlabel("Number of features used by the model")
    ax_time.set_xlim(0.5, n_features.max() + 0.5)
    ax_time.xaxis.set_major_locator(MaxNLocator(integer=True))
    if n_selected is not None:
        _mark(ax_score, n_selected, selected_label)
        ax_time.axvline(n_selected, color=INK_MUTED, linewidth=1.0, zorder=1)
    _title(ax_score, title)
    return _save(fig, path)
