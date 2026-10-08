"""Paired statistics and cross-target synthesis for TESS time-series results."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import classify_scatter_change


PRIMARY_METRIC = "flux_scatter_std_change_percent"


def _median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else math.nan


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else math.nan


def build_target_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive direction labels to target-level metrics."""
    result = metrics.copy()

    result["primary_scatter_change_direction"] = result[
        PRIMARY_METRIC
    ].apply(
        lambda value: classify_scatter_change(float(value))
        if pd.notna(value)
        else "not_available"
    )

    return result.sort_values("target_name").reset_index(drop=True)


def build_overall_summary(
    metrics: pd.DataFrame,
    *,
    survey: str = "TESS",
) -> pd.DataFrame:
    """Produce the compact synthesis used for Section 5.3.3."""
    if metrics.empty:
        return pd.DataFrame(
            [
                {
                    "survey": survey,
                    "target_count": 0,
                }
            ]
        )

    changes = pd.to_numeric(
        metrics[PRIMARY_METRIC],
        errors="coerce",
    )

    finite = changes.dropna()

    lower = int((finite < 0).sum())
    higher = int((finite > 0).sum())
    unchanged = int((finite == 0).sum())

    row = {
        "survey": survey,
        "target_count": int(len(metrics)),
        "targets_with_lower_scatter": lower,
        "targets_with_higher_scatter": higher,
        "targets_unchanged": unchanged,
        "targets_with_available_primary_metric": int(len(finite)),
        "median_scatter_change_percent": _median(finite),
        "mean_scatter_change_percent": _mean(finite),
        "minimum_scatter_change_percent": (
            float(finite.min()) if not finite.empty else math.nan
        ),
        "maximum_scatter_change_percent": (
            float(finite.max()) if not finite.empty else math.nan
        ),
    }

    return pd.DataFrame([row])


def build_experiment_context(
    experiment_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Retain one authoritative metadata row per experiment."""
    columns = [
        column
        for column in (
            "experiment_id",
            "target_name",
            "survey",
            "experiment_type",
            "source_scope",
            "pipeline",
            "observation_time",
            "observation_time_scale",
            "observation_time_format",
            "instrument",
            "filter_name",
            "exposure_time",
            "dataset_identifier",
            "detected_source_count",
            "target_source_identifier",
        )
        if column in experiment_summary.columns
    ]

    return (
        experiment_summary[columns]
        .drop_duplicates("experiment_id")
        .sort_values("experiment_id")
        .reset_index(drop=True)
    )


def build_source_context(
    source_measurements: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact target-level source context table.

    This preserves the source-level export's contribution to the final
    analysis without duplicating cadence-level rows.
    """
    target = source_measurements.loc[
        source_measurements["is_target_source"].fillna(False).astype(bool)
    ].copy()

    columns = [
        column
        for column in (
            "experiment_id",
            "target_name",
            "survey",
            "experiment_type",
            "pipeline",
            "source_identifier",
            "net_flux",
            "instrumental_magnitude",
            "background_level",
            "background_fraction",
            "aperture_coverage",
            "annulus_coverage",
            "valid",
            "quality_flag",
        )
        if column in target.columns
    ]

    return target[columns].sort_values(
        ["target_name", "pipeline"],
    ).reset_index(drop=True)
