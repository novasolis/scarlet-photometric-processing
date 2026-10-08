"""Generate final thesis tables and figures for Section 5.6 (RQ4).

Run from the project root:

    python -m analysis.differential_photometry.section_5_6_results

The script uses the production differential-photometry analysis configuration
and requires no command-line arguments.

Final thesis artefacts are written to:

    results/figures_and_tables/differential_photometry

Intermediate analysis CSV/JSON files remain in:

    results/analysis/differential_photometry
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import (
    DEFAULT_INPUT_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_TARGET_TIMESERIES_DIRECTORY,
    run_analysis,
)
from .dataset import load_differential_measurements


FIGURE_DPI = 300

THESIS_OUTPUT_DIRECTORY = (
    Path("results")
    / "figures_and_tables"
    / "differential_photometry"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_output(outputs: dict[str, Path], key: str) -> pd.DataFrame:
    """Read a tabular output produced by the production analysis."""
    path = outputs[key]

    if not path.exists():
        raise FileNotFoundError(
            f"Expected analysis output was not created: {path.resolve()}"
        )

    return pd.read_csv(path)


def _relative_change(baseline: float, scarlet: float) -> float:
    """Return SCARLET change relative to Baseline as a percentage."""
    if baseline == 0:
        return np.nan
    return (scarlet - baseline) / baseline * 100.0


def _paired_values(
    dataframe: pd.DataFrame,
    target: str,
    column: str,
) -> tuple[float, float]:
    """Return Baseline and SCARLET values for one target."""
    baseline = dataframe.loc[
        (dataframe["target_name"] == target)
        & (dataframe["pipeline"] == "Baseline"),
        column,
    ]

    scarlet = dataframe.loc[
        (dataframe["target_name"] == target)
        & (dataframe["pipeline"] == "SCARLET"),
        column,
    ]

    if baseline.empty or scarlet.empty:
        raise ValueError(
            f"Missing Baseline or SCARLET value for "
            f"target={target!r}, column={column!r}."
        )

    return float(baseline.iloc[0]), float(scarlet.iloc[0])


def _validate_measurements(dataframe: pd.DataFrame) -> None:
    """Validate the minimum structure required for thesis figures."""
    required = {
        "target_name",
        "pipeline",
        "cadence_index",
        "differential_flux",
        "differential_magnitude",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(
            "Differential measurement data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    targets = sorted(dataframe["target_name"].unique())

    if not targets:
        raise ValueError("No differential-photometry targets were found.")

    for target in targets:
        for pipeline in ("Baseline", "SCARLET"):
            subset = dataframe.loc[
                (dataframe["target_name"] == target)
                & (dataframe["pipeline"] == pipeline)
            ]

            if len(subset) != 300:
                raise ValueError(
                    f"{target} / {pipeline} contains {len(subset)} cadences; "
                    "expected 300."
                )


# ---------------------------------------------------------------------------
# Figure 5.6.1 — paired differential light curves
# ---------------------------------------------------------------------------

def _plot_differential_light_curves(
    differential: pd.DataFrame,
    output: Path,
) -> Path:
    """Plot the complete 300-cadence differential light curves."""
    targets = sorted(differential["target_name"].unique())

    fig, axes = plt.subplots(
        len(targets),
        2,
        figsize=(12, 3.0 * len(targets)),
        squeeze=False,
    )

    specifications = (
        (
            "differential_flux",
            "Differential flux",
            "Differential flux",
        ),
        (
            "differential_magnitude",
            "Differential magnitude (mag)",
            "Differential magnitude",
        ),
    )

    for row, target in enumerate(targets):
        target_data = differential[
            differential["target_name"] == target
        ].sort_values("cadence_index")

        for column_index, (column, ylabel, title) in enumerate(
            specifications
        ):
            ax = axes[row, column_index]

            for pipeline, label in (
                ("Baseline", "Baseline"),
                ("SCARLET", "SCARLET"),
            ):
                subset = target_data[
                    target_data["pipeline"] == pipeline
                ]

                ax.plot(
                    subset["cadence_index"],
                    subset[column],
                    linewidth=0.9,
                    alpha=0.85,
                    label=label,
                )

            ax.set_title(
                f"{target} — {title}",
                fontsize=10,
            )
            ax.set_xlabel("Cadence")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)

            if row == 0 and column_index == 0:
                ax.legend(frameon=False)

        axes[row, 0].set_ylabel(
            f"{target}\nDifferential flux"
        )

    fig.suptitle(
        "TESS differential photometry: paired Baseline and SCARLET measurements",
        fontsize=13,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    path = (
        output
        / "figure_5_6_1_differential_light_curves.png"
    )

    fig.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


# ---------------------------------------------------------------------------
# Figure 5.6.2 — relative metric changes
# ---------------------------------------------------------------------------

def _build_metric_change_table(
    timeseries: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact target-level relative-change table."""
    targets = sorted(timeseries["target_name"].unique())

    rows: list[dict[str, object]] = []

    for target in targets:
        baseline_flux, scarlet_flux = _paired_values(
            timeseries,
            target,
            "flux_scatter_percent",
        )

        baseline_mag, scarlet_mag = _paired_values(
            timeseries,
            target,
            "magnitude_scatter_sd",
        )

        baseline_temporal, scarlet_temporal = _paired_values(
            timeseries,
            target,
            "temporal_stability_percent",
        )

        rows.append(
            {
                "target_name": target,
                "flux_scatter_change_percent": _relative_change(
                    baseline_flux,
                    scarlet_flux,
                ),
                "magnitude_scatter_change_percent": _relative_change(
                    baseline_mag,
                    scarlet_mag,
                ),
                "temporal_stability_change_percent": _relative_change(
                    baseline_temporal,
                    scarlet_temporal,
                ),
            }
        )

    return pd.DataFrame(rows)


def _plot_metric_changes(
    changes: pd.DataFrame,
    output: Path,
) -> Path:
    """Plot paired relative changes with a zero-change reference line."""
    targets = changes["target_name"].tolist()
    x = np.arange(len(targets))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4.2),
        sharex=True,
    )

    specifications = (
        (
            "flux_scatter_change_percent",
            "Flux scatter",
        ),
        (
            "magnitude_scatter_change_percent",
            "Magnitude scatter",
        ),
        (
            "temporal_stability_change_percent",
            "Cadence-to-cadence variability",
        ),
    )

    for ax, (column, title) in zip(axes, specifications):
        values = changes[column].to_numpy(dtype=float)

        ax.axhline(
            0,
            linewidth=0.9,
            linestyle="--",
        )

        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.2,
        )

        for index, value in enumerate(values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:+.1f}%",
                    (index, value),
                    xytext=(0, 7 if value >= 0 else -13),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

        ax.set_title(title)
        ax.set_xticks(x, targets, rotation=30, ha="right")
        ax.set_ylabel("SCARLET change relative to Baseline (%)")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Relative changes in differential-photometry variability following SCARLET",
        fontsize=13,
    )

    fig.tight_layout()

    path = (
        output
        / "figure_5_6_2_differential_metric_changes.png"
    )

    fig.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


# ---------------------------------------------------------------------------
# Figure 5.6.3 — differential-flux signal preservation
# ---------------------------------------------------------------------------

def _plot_flux_correlations(
    differential: pd.DataFrame,
    correlations: pd.DataFrame,
    output: Path,
) -> Path:
    """Plot Baseline versus SCARLET differential-flux measurements."""
    targets = sorted(differential["target_name"].unique())

    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(4.2 * len(targets), 4.0),
        squeeze=False,
    )

    for index, target in enumerate(targets):
        ax = axes[0, index]

        baseline = differential.loc[
            (differential["target_name"] == target)
            & (differential["pipeline"] == "Baseline")
        ].sort_values("cadence_index")

        scarlet = differential.loc[
            (differential["target_name"] == target)
            & (differential["pipeline"] == "SCARLET")
        ].sort_values("cadence_index")

        paired = baseline.merge(
            scarlet,
            on="cadence_index",
            suffixes=("_baseline", "_scarlet"),
            how="inner",
        )

        x = paired["differential_flux_baseline"].to_numpy(
            dtype=float
        )
        y = paired["differential_flux_scarlet"].to_numpy(
            dtype=float
        )

        lower = float(np.nanmin(np.concatenate((x, y))))
        upper = float(np.nanmax(np.concatenate((x, y))))
        span = upper - lower

        if span == 0:
            span = 1.0

        padding = span * 0.05
        line_start = lower - padding
        line_end = upper + padding

        ax.plot(
            [line_start, line_end],
            [line_start, line_end],
            linestyle="--",
            linewidth=0.9,
        )

        ax.scatter(
            x,
            y,
            s=9,
            alpha=0.55,
        )

        correlation_row = correlations.loc[
            correlations["target_name"] == target
        ]

        if not correlation_row.empty:
            correlation_column = next(
                (
                    column
                    for column in (
                        "differential_flux_correlation",
                        "correlation",
                        "r",
                    )
                    if column in correlation_row.columns
                ),
                None,
            )

            if correlation_column is not None:
                r_value = float(
                    correlation_row.iloc[0][correlation_column]
                )

                ax.text(
                    0.04,
                    0.95,
                    f"r = {r_value:.3f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                )

        ax.set_title(target)
        ax.set_xlabel("Baseline differential flux")
        ax.set_ylabel("SCARLET differential flux")
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Differential-flux signal preservation after SCARLET preprocessing",
        fontsize=13,
    )

    fig.tight_layout()

    path = (
        output
        / "figure_5_6_3_differential_flux_correlation.png"
    )

    fig.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


# ---------------------------------------------------------------------------
# Compact thesis tables
# ---------------------------------------------------------------------------

def _write_compact_tables(
    differential: pd.DataFrame,
    timeseries_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    output: Path,
) -> list[Path]:
    """Write compact thesis tables directly from canonical/raw measurements.

    The production measurement-summary CSV does not expose the derived column
    names used by this reporting layer. To avoid coupling Section 5.6 to those
    internal column names, differential flux/magnitude scatter is calculated
    directly from the validated 300-cadence differential measurements.
    """
    targets = sorted(differential["target_name"].unique())

    measurement_rows: list[dict[str, object]] = []

    for target in targets:
        baseline = differential.loc[
            (differential["target_name"] == target)
            & (differential["pipeline"] == "Baseline")
        ].sort_values("cadence_index")

        scarlet = differential.loc[
            (differential["target_name"] == target)
            & (differential["pipeline"] == "SCARLET")
        ].sort_values("cadence_index")

        baseline_flux_sd = float(
            baseline["differential_flux"].std(ddof=1)
        )
        scarlet_flux_sd = float(
            scarlet["differential_flux"].std(ddof=1)
        )
        baseline_mag_sd = float(
            baseline["differential_magnitude"].std(ddof=1)
        )
        scarlet_mag_sd = float(
            scarlet["differential_magnitude"].std(ddof=1)
        )

        measurement_rows.append(
            {
                "Target": target,
                "Baseline flux SD": baseline_flux_sd,
                "SCARLET flux SD": scarlet_flux_sd,
                "Flux change (%)": _relative_change(
                    baseline_flux_sd,
                    scarlet_flux_sd,
                ),
                "Baseline mag SD": baseline_mag_sd,
                "SCARLET mag SD": scarlet_mag_sd,
                "Mag change (%)": _relative_change(
                    baseline_mag_sd,
                    scarlet_mag_sd,
                ),
            }
        )

    measurement_table = pd.DataFrame(measurement_rows)

    measurement_path = (
        output
        / "table_5_6_1_differential_measurements.csv"
    )

    measurement_table.to_csv(
        measurement_path,
        index=False,
    )

    stability_rows: list[dict[str, object]] = []

    for target in targets:
        baseline_temporal, scarlet_temporal = _paired_values(
            timeseries_summary,
            target,
            "temporal_stability_percent",
        )

        correlation_row = correlations.loc[
            correlations["target_name"] == target
        ]

        correlation_column = next(
            (
                column
                for column in (
                    "differential_flux_correlation",
                    "correlation",
                    "r",
                )
                if column in correlation_row.columns
            ),
            None,
        )

        correlation = (
            float(correlation_row.iloc[0][correlation_column])
            if correlation_column is not None
            and not correlation_row.empty
            else np.nan
        )

        stability_rows.append(
            {
                "Target": target,
                "Baseline temporal SD (%)": baseline_temporal,
                "SCARLET temporal SD (%)": scarlet_temporal,
                "Temporal change (%)": _relative_change(
                    baseline_temporal,
                    scarlet_temporal,
                ),
                "Differential flux r": correlation,
            }
        )

    stability_table = pd.DataFrame(stability_rows)

    stability_path = (
        output
        / "table_5_6_2_differential_timeseries_stability.csv"
    )

    stability_table.to_csv(
        stability_path,
        index=False,
    )

    return [
        measurement_path,
        stability_path,
    ]


# ---------------------------------------------------------------------------
# Audit artefacts
# ---------------------------------------------------------------------------

def _write_text_summary(
    measurement_summary: pd.DataFrame,
    timeseries_summary: pd.DataFrame,
    metric_changes: pd.DataFrame,
    correlations: pd.DataFrame,
    output: Path,
) -> Path:
    """Write a reproducible audit summary."""
    lines = [
        "RQ4 Differential Photometry — Section 5.6",
        "==========================================",
        "",
        "Canonical differential measurement summary:",
        measurement_summary.to_string(index=False),
        "",
        "Canonical differential time-series metrics:",
        timeseries_summary.to_string(index=False),
        "",
        "Canonical Baseline-to-SCARLET metric changes:",
        metric_changes.to_string(index=False),
        "",
        "Canonical Baseline/SCARLET differential-flux correlations:",
        correlations.to_string(index=False),
    ]

    path = output / "section_5_6_results.txt"

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


def _write_manifest(
    output: Path,
    artefacts: list[Path],
) -> Path:
    """Write the Section 5.6 artefact manifest."""
    manifest = {
        "section": "5.6",
        "research_question": "RQ4",
        "analysis_output_directory": str(
            Path(DEFAULT_OUTPUT_DIRECTORY).resolve()
        ),
        "thesis_output_directory": str(
            output.resolve()
        ),
        "artefacts": [
            path.name
            for path in artefacts
        ],
        "figures": [
            path.name
            for path in artefacts
            if path.suffix.lower() == ".png"
        ],
        "tables": [
            path.name
            for path in artefacts
            if path.suffix.lower() == ".csv"
        ],
    }

    path = output / "section_5_6_manifest.json"

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run RQ4 and generate final Section 5.6 thesis artefacts."""
    thesis_output = THESIS_OUTPUT_DIRECTORY

    thesis_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Run the production analysis first. Its canonical outputs remain in
    # results/analysis/differential_photometry.
    outputs = run_analysis(
        input_directory=DEFAULT_INPUT_DIRECTORY,
        output_directory=DEFAULT_OUTPUT_DIRECTORY,
        target_timeseries_directory=DEFAULT_TARGET_TIMESERIES_DIRECTORY,
    )

    differential = load_differential_measurements(
        DEFAULT_INPUT_DIRECTORY
        / "differential_timeseries_measurements.csv"
    )

    _validate_measurements(differential)

    measurement_summary = _read_output(
        outputs,
        "differential_measurement_summary",
    )

    timeseries_summary = _read_output(
        outputs,
        "differential_timeseries_metrics",
    )

    metric_changes = _read_output(
        outputs,
        "differential_metric_changes",
    )

    correlations = _read_output(
        outputs,
        "differential_light_curve_correlations",
    )

    artefacts: list[Path] = []

    # Compact tables. Differential scatter is derived directly from the
    # validated 300-cadence measurements; temporal stability and correlation
    # are taken from the canonical production analysis outputs.
    artefacts.extend(
        _write_compact_tables(
            differential,
            timeseries_summary,
            correlations,
            thesis_output,
        )
    )

    # Final thesis figures.
    artefacts.append(
        _plot_differential_light_curves(
            differential,
            thesis_output,
        )
    )

    metric_changes_for_plot = _build_metric_change_table(
        timeseries_summary
    )

    artefacts.append(
        _plot_metric_changes(
            metric_changes_for_plot,
            thesis_output,
        )
    )

    artefacts.append(
        _plot_flux_correlations(
            differential,
            correlations,
            thesis_output,
        )
    )

    # Audit text and manifest.
    artefacts.append(
        _write_text_summary(
            measurement_summary,
            timeseries_summary,
            metric_changes,
            correlations,
            thesis_output,
        )
    )

    manifest_path = _write_manifest(
        thesis_output,
        artefacts,
    )

    print(
        "RQ4 differential-photometry Section 5.6 results generated."
    )
    print(
        f"Analysis output directory: "
        f"{Path(DEFAULT_OUTPUT_DIRECTORY).resolve()}"
    )
    print(
        f"Thesis output directory: "
        f"{thesis_output.resolve()}"
    )
    print()
    print("Generated thesis artefacts:")

    for path in artefacts:
        print(f"  - {path.name}")

    print(f"  - {manifest_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
