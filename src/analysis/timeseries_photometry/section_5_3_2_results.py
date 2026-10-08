"""Section 5.3.2 — TESS Time-Series Photometry.

Thesis reporting driver for the seven paired TESS time-series experiments.

Production inputs are fixed. Scientific calculations are delegated to the
production analysis/metrics modules. This section module performs only:
    1. production analysis execution,
    2. thesis-specific validation,
    3. target-level presentation tables,
    4. section-level descriptive synthesis,
    5. thesis figures, including signal-preservation diagnostics.

Run from the project root:

    python -m analysis.timeseries_photometry.section_5_3_2_results
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import matplotlib.pyplot as plt
import pandas as pd

from .analysis import run_analysis


# ---------------------------------------------------------------------------
# Fixed production and thesis locations
# ---------------------------------------------------------------------------

PRODUCTION_INPUT_DIR: Final[Path] = Path(
    "results/exports/timeseries_photometry"
)

ANALYSIS_DIR: Final[Path] = Path(
    "results/analysis/timeseries_photometry"
)

THESIS_DIR: Final[Path] = (
    Path("results/figures_and_tables/timeseries_photometry")
    / "5_3_2_tess_time_series_photometry"
)

EXPECTED_TARGETS: Final[int] = 7
EXPECTED_CADENCES: Final[int] = 300
EXPECTED_PIPELINES: Final[tuple[str, str]] = ("Baseline", "SCARLET")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(condition: object, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    _require(
        not missing,
        f"{name} is missing required columns: {', '.join(missing)}",
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _normalised_flux_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Return the Baseline/SCARLET normalised-flux columns in the production export.

    The production paired-normalised export has used both naming conventions
    across development versions. Accept the canonical current name first,
    while retaining compatibility with the earlier export name.
    """
    candidates = (
        ("baseline_flux_normalised", "scarlet_flux_normalised"),
        ("baseline_normalised_flux", "scarlet_normalised_flux"),
    )

    for baseline, scarlet in candidates:
        if baseline in frame.columns and scarlet in frame.columns:
            return baseline, scarlet

    raise ValueError(
        "Paired time-series export does not contain a recognised pair of "
        "normalised flux columns. Expected either "
        "baseline_flux_normalised/scarlet_flux_normalised or "
        "baseline_normalised_flux/scarlet_normalised_flux."
    )


# ---------------------------------------------------------------------------
# Thesis table
# ---------------------------------------------------------------------------

def _presentation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build the principal Section 5.3.2 target-level results table."""

    required = {
        "target_name",
        "paired_cadence_count",
        "baseline_flux_scatter_std",
        "scarlet_flux_scatter_std",
        "flux_scatter_std_change_percent",
        "baseline_magnitude_scatter_std",
        "scarlet_magnitude_scatter_std",
        "magnitude_scatter_std_change_percent",
        "baseline_temporal_stability_std_diff",
        "scarlet_temporal_stability_std_diff",
        "temporal_stability_std_diff_change_percent",
        "lightcurve_correlation",
    }
    _require_columns(metrics, required, "target_timeseries_metrics.csv")

    table = metrics[
        [
            "target_name",
            "paired_cadence_count",
            "baseline_flux_scatter_std",
            "scarlet_flux_scatter_std",
            "flux_scatter_std_change_percent",
            "baseline_magnitude_scatter_std",
            "scarlet_magnitude_scatter_std",
            "magnitude_scatter_std_change_percent",
            "baseline_temporal_stability_std_diff",
            "scarlet_temporal_stability_std_diff",
            "temporal_stability_std_diff_change_percent",
            "lightcurve_correlation",
        ]
    ].copy()

    table = table.rename(
        columns={
            "target_name": "Target",
            "paired_cadence_count": "Paired cadences",
            "baseline_flux_scatter_std": "Flux scatter SD — Baseline",
            "scarlet_flux_scatter_std": "Flux scatter SD — SCARLET",
            "flux_scatter_std_change_percent": "Flux scatter change (%)",
            "baseline_magnitude_scatter_std": (
                "Magnitude scatter SD — Baseline (mag)"
            ),
            "scarlet_magnitude_scatter_std": (
                "Magnitude scatter SD — SCARLET (mag)"
            ),
            "magnitude_scatter_std_change_percent": (
                "Magnitude scatter change (%)"
            ),
            "baseline_temporal_stability_std_diff": (
                "Cadence-to-cadence variability SD(diff) — Baseline"
            ),
            "scarlet_temporal_stability_std_diff": (
                "Cadence-to-cadence variability SD(diff) — SCARLET"
            ),
            "temporal_stability_std_diff_change_percent": (
                "Cadence-to-cadence variability change (%)"
            ),
            "lightcurve_correlation": "Light-curve correlation",
        }
    )

    numeric_columns = [
        column for column in table.columns if column != "Target"
    ]
    table[numeric_columns] = table[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    table["Paired cadences"] = table["Paired cadences"].astype("int64")

    metric_columns = [
        column
        for column in numeric_columns
        if column != "Paired cadences"
    ]
    table[metric_columns] = table[metric_columns].round(6)

    return (
        table.sort_values("Target", kind="stable")
        .reset_index(drop=True)
    )


def _build_section_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create the compact descriptive synthesis for Section 5.3.2."""

    flux = _numeric(
        metrics,
        "flux_scatter_std_change_percent",
    ).dropna()

    magnitude = _numeric(
        metrics,
        "magnitude_scatter_std_change_percent",
    ).dropna()

    temporal = _numeric(
        metrics,
        "temporal_stability_std_diff_change_percent",
    ).dropna()

    correlation = _numeric(
        metrics,
        "lightcurve_correlation",
    ).dropna()

    return pd.DataFrame(
        [
            {
                "target_count": int(len(metrics)),
                "paired_cadences_per_target": EXPECTED_CADENCES,
                "flux_scatter_lower_count": int((flux < 0).sum()),
                "flux_scatter_higher_count": int((flux > 0).sum()),
                "flux_scatter_unchanged_count": int((flux == 0).sum()),
                "flux_scatter_median_change_percent": float(flux.median()),
                "magnitude_scatter_lower_count": int(
                    (magnitude < 0).sum()
                ),
                "magnitude_scatter_higher_count": int(
                    (magnitude > 0).sum()
                ),
                "magnitude_scatter_unchanged_count": int(
                    (magnitude == 0).sum()
                ),
                "magnitude_scatter_median_change_percent": float(
                    magnitude.median()
                ),
                "temporal_stability_lower_count": int(
                    (temporal < 0).sum()
                ),
                "temporal_stability_higher_count": int(
                    (temporal > 0).sum()
                ),
                "temporal_stability_unchanged_count": int(
                    (temporal == 0).sum()
                ),
                "temporal_stability_median_change_percent": float(
                    temporal.median()
                ),
                "median_lightcurve_correlation": float(
                    correlation.median()
                ),
                "minimum_lightcurve_correlation": float(
                    correlation.min()
                ),
            }
        ]
    )



# ---------------------------------------------------------------------------
# Signal-preservation diagnostics
# ---------------------------------------------------------------------------

def _signal_preservation_metrics(
    paired: pd.DataFrame,
    targets: list[str],
) -> pd.DataFrame:
    """Quantify cadence-level preservation of the measured light-curve shape.

    This is a *paired-method agreement* analysis, not a claim that either
    pipeline is the astrophysical truth.  For each target, Baseline and
    SCARLET normalised flux deviations are compared at the same cadence.

    Diagnostics:
      * Pearson correlation: preservation of temporal shape.
      * Spearman correlation: preservation of temporal ordering, less
        sensitive to amplitude scaling and outliers.
      * OLS slope (SCARLET on Baseline): amplitude/gain preservation.
      * Amplitude ratio: SD(SCARLET) / SD(Baseline).
      * RMSE: absolute cadence-level disagreement in normalised flux.
      * NRMSE: RMSE / SD(Baseline), allowing comparison across targets.
      * Median absolute error: robust cadence-level disagreement.

    No composite "signal preservation score" is created because that would
    introduce an arbitrary weighting of distinct scientific properties.
    """
    _require_columns(
        paired,
        {"target_name", "cadence_index"},
        "paired_timeseries_normalised.csv",
    )

    baseline_column, scarlet_column = _normalised_flux_columns(paired)
    rows: list[dict[str, float | int | str]] = []

    for target in targets:
        current = paired.loc[
            paired["target_name"].astype(str).eq(target)
        ].sort_values("cadence_index")

        _require(
            len(current) == EXPECTED_CADENCES,
            (
                f"{target}: expected {EXPECTED_CADENCES} paired cadences; "
                f"found {len(current)}."
            ),
        )

        baseline = pd.to_numeric(
            current[baseline_column], errors="coerce"
        ).to_numpy(dtype=float)
        scarlet = pd.to_numeric(
            current[scarlet_column], errors="coerce"
        ).to_numpy(dtype=float)

        mask = pd.Series(baseline).notna().to_numpy() & pd.Series(scarlet).notna().to_numpy()
        baseline = baseline[mask]
        scarlet = scarlet[mask]

        _require(
            len(baseline) >= 3,
            f"{target}: fewer than three finite paired flux values.",
        )

        # Work in deviation-from-median space.  This makes the diagnostics
        # invariant to the absolute flux scale of each target.
        x = baseline - 1.0
        y = scarlet - 1.0

        x_mean = float(x.mean())
        y_mean = float(y.mean())
        x_centered = x - x_mean
        y_centered = y - y_mean

        x_ss = float((x_centered ** 2).sum())
        y_ss = float((y_centered ** 2).sum())
        xy_ss = float((x_centered * y_centered).sum())

        _require(
            x_ss > 0.0 and y_ss > 0.0,
            f"{target}: zero variance prevents signal-preservation analysis.",
        )

        pearson = xy_ss / (x_ss ** 0.5 * y_ss ** 0.5)
        slope = xy_ss / x_ss
        intercept = y_mean - slope * x_mean

        # Spearman correlation is Pearson correlation of the ranks.  Use
        # pandas ranking so ties are handled consistently.
        x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        xr = x_rank - x_rank.mean()
        yr = y_rank - y_rank.mean()
        spearman = float((xr * yr).sum() / ((xr ** 2).sum() ** 0.5 * (yr ** 2).sum() ** 0.5))

        residual = y - x
        rmse = float((residual ** 2).mean() ** 0.5)
        baseline_sd = float(x.std(ddof=1))
        scarlet_sd = float(y.std(ddof=1))
        amplitude_ratio = scarlet_sd / baseline_sd
        nrmse = rmse / baseline_sd
        median_abs_error = float(pd.Series(residual).abs().median())

        rows.append(
            {
                "target_name": target,
                "paired_cadence_count": int(len(x)),
                "pearson_correlation": float(pearson),
                "spearman_correlation": spearman,
                "signal_gain_slope": float(slope),
                "signal_gain_intercept": float(intercept),
                "amplitude_ratio_scarlet_to_baseline": float(amplitude_ratio),
                "amplitude_change_percent": float((amplitude_ratio - 1.0) * 100.0),
                "signal_rmse": rmse,
                "signal_nrmse": float(nrmse),
                "median_absolute_signal_error": median_abs_error,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("target_name", kind="stable")
        .reset_index(drop=True)
    )


def _write_signal_preservation_figure(
    paired: pd.DataFrame,
    targets: list[str],
    signal_metrics: pd.DataFrame,
) -> Path:
    """Write Baseline-versus-SCARLET signal-preservation diagnostics."""
    path = (
        THESIS_DIR
        / "figure_5_3_2d_tess_signal_preservation.png"
    )

    baseline_column, scarlet_column = _normalised_flux_columns(paired)

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(13, 13),
        constrained_layout=True,
    )
    axes_flat = list(axes.ravel())

    metric_index = signal_metrics.set_index("target_name")

    for index, target in enumerate(targets):
        ax = axes_flat[index]
        current = paired.loc[
            paired["target_name"].astype(str).eq(target)
        ].sort_values("cadence_index")

        x = pd.to_numeric(current[baseline_column], errors="coerce").to_numpy(dtype=float) - 1.0
        y = pd.to_numeric(current[scarlet_column], errors="coerce").to_numpy(dtype=float) - 1.0
        mask = pd.Series(x).notna().to_numpy() & pd.Series(y).notna().to_numpy()
        x = x[mask]
        y = y[mask]

        row = metric_index.loc[target]
        slope = cast(float, row["signal_gain_slope"])
        intercept = cast(float, row["signal_gain_intercept"])

        ax.scatter(x, y, s=9, alpha=0.45)

        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        span = hi - lo
        pad = 0.05 * span if span > 0 else 1e-6
        line_x = pd.Series([lo - pad, hi + pad])

        ax.plot(
            line_x,
            line_x,
            linewidth=0.8,
            linestyle="--",
            label="1:1",
        )
        ax.plot(
            line_x,
            slope * line_x + intercept,
            linewidth=1.0,
            label="OLS fit",
        )

        ax.set_title(
            f"{target}\nr={row['pearson_correlation']:.4f}, "
            f"slope={slope:.3f}, NRMSE={row['signal_nrmse']:.3f}"
        )
        ax.set_xlabel("Baseline normalised flux deviation")
        ax.set_ylabel("SCARLET normalised flux deviation")
        ax.grid(alpha=0.18, linewidth=0.5)

        if index == 0:
            ax.legend(frameon=False, fontsize=8)

    axes_flat[-1].axis("off")

    fig.suptitle(
        "TESS time-series photometry: signal-preservation diagnostics",
        fontsize=14,
    )
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return path


# ---------------------------------------------------------------------------
# Figure 5.3.2a — paired target light curves
# ---------------------------------------------------------------------------

def _write_lightcurve_figure(
    paired: pd.DataFrame,
    targets: list[str],
) -> Path:
    """Write seven paired target light curves.

    Important: normalisation is recalculated per target and per pipeline.
    The paired production export is not assumed to contain target-local
    normalisation.
    """

    path = THESIS_DIR / "figure_5_3_2a_tess_target_light_curves.png"

    _require_columns(
        paired,
        {"target_name", "cadence_index"},
        "paired_timeseries_normalised.csv",
    )

    baseline_flux_column, scarlet_flux_column = _normalised_flux_columns(
        paired
    )

    # This export is already the result of successful Baseline/SCARLET
    # cadence pairing. It therefore does not need a pair_status column.
    paired_only = paired.copy()
    paired_only["baseline_flux_plot"] = pd.to_numeric(
        paired_only[baseline_flux_column],
        errors="coerce",
    )
    paired_only["scarlet_flux_plot"] = pd.to_numeric(
        paired_only[scarlet_flux_column],
        errors="coerce",
    )

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(13, 13),
        constrained_layout=True,
    )
    axes_flat = list(axes.ravel())

    for index, target in enumerate(targets):
        ax = axes_flat[index]

        current = paired_only.loc[
            paired_only["target_name"].astype(str).eq(target)
        ].sort_values("cadence_index")

        _require(
            len(current) == EXPECTED_CADENCES,
            (
                f"{target}: expected {EXPECTED_CADENCES} paired cadences; "
                f"found {len(current)}."
            ),
        )

        ax.plot(
            current["cadence_index"],
            (current["baseline_flux_plot"] - 1.0) * 100.0,
            label="Baseline",
            linewidth=0.9,
        )
        ax.plot(
            current["cadence_index"],
            (current["scarlet_flux_plot"] - 1.0) * 100.0,
            label="SCARLET",
            linewidth=0.9,
            alpha=0.85,
        )

        ax.axhline(
            0.0,
            linewidth=0.6,
            linestyle="--",
        )
        ax.set_title(target)
        ax.set_xlabel("Cadence")
        ax.set_ylabel("Normalised flux deviation (%)")
        ax.grid(alpha=0.18, linewidth=0.5)

        if index == 0:
            ax.legend(
                frameon=False,
                ncol=2,
                loc="upper right",
            )

    axes_flat[-1].axis("off")

    fig.suptitle(
        "TESS time-series photometry: paired target light curves",
        fontsize=14,
    )
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return path


# ---------------------------------------------------------------------------
# Figure 5.3.2b — absolute variability measures
# ---------------------------------------------------------------------------

def _write_absolute_metrics_figure(
    metrics: pd.DataFrame,
    targets: list[str],
) -> Path:
    path = (
        THESIS_DIR
        / "figure_5_3_2b_tess_scatter_and_stability.png"
    )

    ordered = (
        metrics.set_index("target_name")
        .loc[targets]
        .copy()
    )

    panels = [
        (
            "baseline_flux_scatter_std",
            "scarlet_flux_scatter_std",
            "Flux scatter SD",
            "Normalised flux SD (×10⁻³)",
            1e3,
        ),
        (
            "baseline_magnitude_scatter_std",
            "scarlet_magnitude_scatter_std",
            "Magnitude scatter SD",
            "Magnitude SD (mag)",
            1.0,
        ),
        (
            "baseline_temporal_stability_std_diff",
            "scarlet_temporal_stability_std_diff",
            "Cadence-to-cadence variability",
            "SD of cadence-to-cadence flux difference (×10⁻³)",
            1e3,
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.2),
        constrained_layout=True,
    )

    x = list(range(len(targets)))
    width = 0.38

    for ax, (
        baseline_column,
        scarlet_column,
        title,
        ylabel,
        scale,
    ) in zip(axes, panels):
        baseline = (
            _numeric(ordered, baseline_column).to_numpy() * scale
        )
        scarlet = (
            _numeric(ordered, scarlet_column).to_numpy() * scale
        )

        baseline_bars = ax.bar(
            [value - width / 2 for value in x],
            baseline,
            width=width,
            label="Baseline",
        )
        scarlet_bars = ax.bar(
            [value + width / 2 for value in x],
            scarlet,
            width=width,
            label="SCARLET",
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(
            targets,
            rotation=55,
            ha="right",
            fontsize=8,
        )
        ax.grid(
            axis="y",
            alpha=0.18,
            linewidth=0.5,
        )

        if ax is axes[0]:
            ax.legend(frameon=False, ncol=2)

        for bars in (baseline_bars, scarlet_bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3g}",
                    (
                        bar.get_x() + bar.get_width() / 2,
                        height,
                    ),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                )

    fig.suptitle(
        "TESS time-series photometry: absolute variability measures",
        fontsize=14,
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return path


# ---------------------------------------------------------------------------
# Figure 5.3.2c — percentage changes
# ---------------------------------------------------------------------------

def _write_change_figure(
    metrics: pd.DataFrame,
    targets: list[str],
) -> Path:
    path = (
        THESIS_DIR
        / "figure_5_3_2c_tess_baseline_scarlet_changes.png"
    )

    ordered = (
        metrics.set_index("target_name")
        .loc[targets]
        .copy()
    )

    panels = [
        (
            "flux_scatter_std_change_percent",
            "Flux scatter",
        ),
        (
            "magnitude_scatter_std_change_percent",
            "Magnitude scatter",
        ),
        (
            "temporal_stability_std_diff_change_percent",
            "Cadence-to-cadence variability",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.2),
        constrained_layout=True,
    )

    for ax, (column, title) in zip(axes, panels):
        values = _numeric(ordered, column).to_numpy()

        bars = ax.bar(
            range(len(targets)),
            values,
        )
        ax.axhline(
            0.0,
            linewidth=0.8,
            linestyle="--",
        )
        ax.set_title(title)
        ax.set_ylabel("Change relative to Baseline (%)")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(
            targets,
            rotation=55,
            ha="right",
            fontsize=8,
        )
        ax.grid(
            axis="y",
            alpha=0.18,
            linewidth=0.5,
        )

        for bar, value in zip(bars, values):
            if pd.isna(value):
                continue

            offset = 4 if value >= 0 else -12
            ax.annotate(
                f"{value:.1f}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    value,
                ),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
            )

    fig.suptitle(
        "TESS time-series photometry: SCARLET change relative to Baseline",
        fontsize=14,
    )
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict[str, Path]:
    """Run the complete Section 5.3.2 reporting workflow."""

    THESIS_DIR.mkdir(parents=True, exist_ok=True)

    analysis_outputs = run_analysis(
        input_directory=PRODUCTION_INPUT_DIR,
        output_directory=ANALYSIS_DIR,
        expected_cadences=EXPECTED_CADENCES,
        fail_on_validation_error=True,
    )

    metrics = pd.read_csv(
        analysis_outputs["target_metrics"],
    )
    paired = pd.read_csv(
        analysis_outputs["paired_timeseries_normalised"],
    )

    # These are optional audit outputs from the production analysis. The
    # section does not use them to calculate scientific metrics.
    source_context = pd.read_csv(
        analysis_outputs["source_context"],
    )
    experiment_context = pd.read_csv(
        analysis_outputs["experiment_context"],
    )

    # ------------------------------------------------------------------
    # Production metric schema validation.
    # ------------------------------------------------------------------

    metric_columns = {
        "target_name",
        "paired_cadence_count",
        "baseline_flux_scatter_std",
        "scarlet_flux_scatter_std",
        "flux_scatter_std_change_percent",
        "baseline_magnitude_scatter_std",
        "scarlet_magnitude_scatter_std",
        "magnitude_scatter_std_change_percent",
        "baseline_temporal_stability_std_diff",
        "scarlet_temporal_stability_std_diff",
        "temporal_stability_std_diff_change_percent",
        "lightcurve_correlation",
    }
    _require_columns(
        metrics,
        metric_columns,
        "target_timeseries_metrics.csv",
    )

    _require(
        len(metrics) == EXPECTED_TARGETS,
        (
            f"Expected {EXPECTED_TARGETS} target metric rows; "
            f"found {len(metrics)}."
        ),
    )

    targets = sorted(
        metrics["target_name"].astype(str).unique()
    )

    _require(
        len(targets) == EXPECTED_TARGETS,
        (
            f"Expected {EXPECTED_TARGETS} unique TESS targets; "
            f"found {len(targets)}."
        ),
    )

    # ------------------------------------------------------------------
    # Paired time-series validation.
    # ------------------------------------------------------------------

    _require_columns(
        paired,
        {"target_name", "cadence_index"},
        "paired_timeseries_normalised.csv",
    )
    _normalised_flux_columns(paired)

    # `paired_timeseries_normalised.csv` is already a paired product: one
    # Baseline row and one SCARLET row have been joined into each record.
    # Consequently the file intentionally contains no pair_status field.
    paired_only = paired.copy()

    _require(
        len(paired_only) == EXPECTED_TARGETS * EXPECTED_CADENCES,
        (
            "Unexpected number of paired target cadences: "
            f"{len(paired_only)}."
        ),
    )

    cadence_counts = (
        paired_only.groupby("target_name")
        .size()
    )

    _require(
        len(cadence_counts) == EXPECTED_TARGETS,
        "Paired time-series data do not contain exactly seven targets.",
    )

    _require(
        cadence_counts.eq(EXPECTED_CADENCES).all(),
        (
            "Every TESS target must contain exactly "
            f"{EXPECTED_CADENCES} paired cadences."
        ),
    )

    # Metric-level cadence counts must agree with the actual paired data.
    metric_counts = (
        metrics.set_index("target_name")["paired_cadence_count"]
        .astype(int)
    )

    _require(
        metric_counts.reindex(cadence_counts.index).eq(
            cadence_counts.astype(int)
        ).all(),
        "Metric cadence counts disagree with paired time-series data.",
    )

    # ------------------------------------------------------------------
    # Context validation.
    # ------------------------------------------------------------------

    _require(
        len(source_context) == EXPECTED_TARGETS * 2,
        (
            "Expected 14 target-source context rows "
            "(7 targets × 2 pipelines); "
            f"found {len(source_context)}."
        ),
    )

    _require(
        len(experiment_context) == EXPECTED_TARGETS * 2,
        (
            "Expected 14 experiment context rows "
            "(7 targets × 2 pipelines); "
            f"found {len(experiment_context)}."
        ),
    )

    _require(
        set(
            experiment_context["pipeline"]
            .astype(str)
            .unique()
        ) == set(EXPECTED_PIPELINES),
        "Experiment context must contain exactly Baseline and SCARLET.",
    )

    for target in targets:
        rows = source_context.loc[
            source_context["target_name"].astype(str).eq(target)
        ]
        _require(
            set(rows["pipeline"].astype(str))
            == set(EXPECTED_PIPELINES),
            f"{target}: missing Baseline or SCARLET source context.",
        )

    # ------------------------------------------------------------------
    # Thesis outputs.
    # ------------------------------------------------------------------

    table = _presentation_table(metrics)

    table_path = (
        THESIS_DIR
        / "table_5_3_2_tess_time_series_photometry.csv"
    )
    table.to_csv(table_path, index=False)

    section_summary = _build_section_summary(metrics)

    section_summary_path = (
        THESIS_DIR
        / "section_5_3_2_summary.csv"
    )
    section_summary.to_csv(
        section_summary_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Signal-preservation analysis.
    # ------------------------------------------------------------------

    signal_metrics = _signal_preservation_metrics(
        paired,
        targets,
    )

    signal_metrics_path = (
        THESIS_DIR
        / "table_5_3_2d_tess_signal_preservation.csv"
    )
    signal_metrics.to_csv(
        signal_metrics_path,
        index=False,
    )

    signal_figure_path = _write_signal_preservation_figure(
        paired,
        targets,
        signal_metrics,
    )

    provenance_columns = [
        column
        for column in (
            "experiment_id",
            "target_name",
            "survey",
            "experiment_type",
            "pipeline",
            "instrument",
            "filter_name",
            "dataset_identifier",
            "target_source_identifier",
        )
        if column in experiment_context.columns
    ]

    provenance = experiment_context[
        provenance_columns
    ].sort_values(
        ["target_name", "pipeline"],
        kind="stable",
    )

    provenance_path = (
        THESIS_DIR
        / "section_5_3_2_experiment_provenance.csv"
    )
    provenance.to_csv(
        provenance_path,
        index=False,
    )

    lightcurve_path = _write_lightcurve_figure(
        paired,
        targets,
    )

    absolute_metrics_path = _write_absolute_metrics_figure(
        metrics,
        targets,
    )

    change_path = _write_change_figure(
        metrics,
        targets,
    )

    return {
        "production_validation": analysis_outputs["validation_report"],
        "production_metrics": analysis_outputs["target_metrics"],
        "production_paired_timeseries": (
            analysis_outputs["paired_timeseries_normalised"]
        ),
        "thesis_table": table_path,
        "section_summary": section_summary_path,
        "experiment_provenance": provenance_path,
        "lightcurve_figure": lightcurve_path,
        "absolute_metrics_figure": absolute_metrics_path,
        "change_figure": change_path,
        "signal_preservation_table": signal_metrics_path,
        "signal_preservation_figure": signal_figure_path,
    }


def main() -> int:
    outputs = run()

    print("SECTION 5.3.2 ANALYSIS PASSED")
    for name, path in outputs.items():
        print(f"[PASS] {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
