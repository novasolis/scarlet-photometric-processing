"""Metrics for differential-photometry measurements and time series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _mad(values: np.ndarray) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _normalised_flux(flux: np.ndarray) -> np.ndarray:
    median = np.median(flux)
    if not np.isfinite(median) or median == 0:
        raise ValueError("Cannot normalise differential flux by a zero/non-finite median.")
    return flux / median - 1.0


def differential_measurement_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the paired differential measurements.

    Flux scatter is reported both as absolute SD and as SD relative to the
    median differential flux. Magnitude scatter is the standard deviation of
    differential magnitude.
    """
    rows = []

    for (target, pipeline), group in df.groupby(
        ["target_name", "pipeline"], sort=True
    ):
        flux = group["differential_flux"].to_numpy(dtype=float)
        magnitude = group["differential_magnitude"].to_numpy(dtype=float)

        normalised = _normalised_flux(flux)

        rows.append(
            {
                "target_name": target,
                "pipeline": pipeline,
                "n_cadences": len(group),
                "median_differential_flux": float(np.median(flux)),
                "median_differential_magnitude": float(np.median(magnitude)),
                "flux_sd": float(np.std(flux, ddof=1)),
                "flux_sd_relative_percent": float(np.std(normalised, ddof=1) * 100),
                "flux_mad_relative_percent": float(_mad(normalised) * 100),
                "magnitude_sd": float(np.std(magnitude, ddof=1)),
                "magnitude_mad": float(_mad(magnitude)),
            }
        )

    return pd.DataFrame(rows)


def differential_timeseries_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate variability and cadence-to-cadence stability for each target.

    The stability metric is the SD of first differences of median-normalised
    differential flux, expressed as percent. This mirrors the normalised
    cadence-to-cadence stability metric used for the ordinary target
    time-series analysis.
    """
    rows = []

    for (target, pipeline), group in df.groupby(
        ["target_name", "pipeline"], sort=True
    ):
        group = group.sort_values("cadence_index")
        flux = group["differential_flux"].to_numpy(dtype=float)
        magnitude = group["differential_magnitude"].to_numpy(dtype=float)

        if len(flux) < 2:
            raise ValueError(f"{target}/{pipeline} has fewer than two cadences.")

        normalised_flux = _normalised_flux(flux)
        flux_scatter_percent = float(np.std(normalised_flux, ddof=1) * 100)
        magnitude_scatter = float(np.std(magnitude, ddof=1))
        temporal_stability_percent = float(
            np.std(np.diff(normalised_flux), ddof=1) * 100
        )

        rows.append(
            {
                "target_name": target,
                "pipeline": pipeline,
                "n_cadences": len(group),
                "flux_scatter_percent": flux_scatter_percent,
                "magnitude_scatter_sd": magnitude_scatter,
                "temporal_stability_percent": temporal_stability_percent,
            }
        )

    return pd.DataFrame(rows)


def paired_metric_changes(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate SCARLET-minus-Baseline and relative changes for paired metrics.
    """
    key = "target_name"
    baseline = summary[summary["pipeline"] == "Baseline"].set_index(key)
    scarlet = summary[summary["pipeline"] == "SCARLET"].set_index(key)

    targets = sorted(set(baseline.index) & set(scarlet.index))
    rows = []

    metric_columns = [
        column
        for column in summary.columns
        if column not in {"target_name", "pipeline", "n_cadences"}
        and pd.api.types.is_numeric_dtype(summary[column])
    ]

    for target in targets:
        row = {"target_name": target}
        for metric in metric_columns:
            b = float(baseline.loc[target, metric])
            s = float(scarlet.loc[target, metric])
            row[f"{metric}_baseline"] = b
            row[f"{metric}_scarlet"] = s
            row[f"{metric}_change"] = s - b
            row[f"{metric}_relative_change_percent"] = (
                (s - b) / abs(b) * 100 if b != 0 else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def paired_light_curve_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Baseline-vs-SCARLET Pearson correlation by target."""
    rows = []

    for target, group in df.groupby("target_name", sort=True):
        baseline = (
            group[group["pipeline"] == "Baseline"]
            .sort_values("cadence_index")[["cadence_index", "differential_flux"]]
            .rename(columns={"differential_flux": "baseline"})
        )
        scarlet = (
            group[group["pipeline"] == "SCARLET"]
            .sort_values("cadence_index")[["cadence_index", "differential_flux"]]
            .rename(columns={"differential_flux": "scarlet"})
        )

        paired = baseline.merge(scarlet, on="cadence_index", how="inner")
        if len(paired) < 2:
            raise ValueError(f"{target} has insufficient paired observations.")

        rows.append(
            {
                "target_name": target,
                "n_paired_cadences": len(paired),
                "differential_flux_correlation": float(
                    paired["baseline"].corr(paired["scarlet"])
                ),
            }
        )

    return pd.DataFrame(rows)
