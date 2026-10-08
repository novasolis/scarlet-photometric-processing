"""Dataset selection and relational joins for TESS time-series analysis."""

from __future__ import annotations

import pandas as pd

from .io import ExperimentSet


def target_timeseries(data: ExperimentSet) -> pd.DataFrame:
    """Return cadence-level rows for the target source only."""
    df = data.timeseries_measurements.copy()

    mask = df["is_target_source"].fillna(False).astype(bool)
    return df.loc[mask].copy()


def target_sources(data: ExperimentSet) -> pd.DataFrame:
    """Return source-level target rows only."""
    df = data.source_measurements.copy()

    mask = df["is_target_source"].fillna(False).astype(bool)
    return df.loc[mask].copy()


def experiment_pairs(data: ExperimentSet) -> pd.DataFrame:
    """Create one metadata row for each target's Baseline/SCARLET pair.

    The pair is identified by target_name and pipeline. experiment_id remains
    the authoritative relational key into the source/time-series tables.
    """
    summary = data.experiment_summary.copy()

    required = {
        "experiment_id",
        "target_name",
        "pipeline",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            f"Experiment summary missing required columns: {sorted(missing)}"
        )

    rows: list[dict[str, object]] = []

    for target_value, group in summary.groupby("target_name", sort=True):
        target_name = str(target_value)
        for pipeline in ("Baseline", "SCARLET"):
            current = group.loc[
                group["pipeline"].astype("string") == pipeline
            ]
            if len(current) != 1:
                continue

            row = current.iloc[0]
            rows.append(
                {
                    "target_name": target_name,
                    "pipeline": pipeline,
                    "experiment_id": str(row["experiment_id"]),
                }
            )

    return pd.DataFrame(rows)


def attach_source_context(
    data: ExperimentSet,
    timeseries: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the corresponding source-level measurement to each cadence row.

    The join is deliberately performed using experiment_id + source_identifier,
    because experiment_id distinguishes Baseline from SCARLET and source IDs are
    persistent within an experiment.
    """
    sources = data.source_measurements.copy()

    columns = [
        "experiment_id",
        "source_identifier",
    ]

    context_columns = [
        column
        for column in (
            "net_flux",
            "instrumental_magnitude",
            "background_level",
            "background_flux",
            "background_fraction",
            "aperture_coverage",
            "annulus_coverage",
            "valid",
            "quality_flag",
            "x_centroid",
            "y_centroid",
        )
        if column in sources.columns
    ]

    source_context = sources[columns + context_columns].copy()

    rename_map = {
        column: f"source_{column}"
        for column in context_columns
    }
    source_context = source_context.rename(columns=rename_map)

    return timeseries.merge(
        source_context,
        on=columns,
        how="left",
        validate="many_to_one",
    )
