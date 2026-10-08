"""Paired Baseline/SCARLET cadence construction."""

from __future__ import annotations

import pandas as pd

from .io import ExperimentSet


PAIR_COLUMNS = [
    "target_name",
    "cadence_index",
    "source_identifier",
]


def pair_target_timeseries(data: ExperimentSet) -> pd.DataFrame:
    """Return one row per paired target/source/cadence observation.

    Baseline and SCARLET experiment IDs are intentionally not part of the
    pairing key because the two pipelines are separate experiments. Their
    experiment IDs are retained as provenance columns.
    """
    ts = data.timeseries_measurements.copy()
    ts = ts.loc[
        ts["is_target_source"].fillna(False).astype(bool)
    ].copy()

    baseline = ts.loc[
        ts["pipeline"].astype("string") == "Baseline"
    ].copy()
    scarlet = ts.loc[
        ts["pipeline"].astype("string") == "SCARLET"
    ].copy()

    baseline = baseline.rename(
        columns={
            "experiment_id": "baseline_experiment_id",
            "observation_time": "baseline_time",
            "net_flux": "baseline_net_flux",
            "instrumental_magnitude": "baseline_magnitude",
            "background_level": "baseline_background_level",
            "background_flux": "baseline_background_flux",
            "background_fraction": "baseline_background_fraction",
            "valid": "baseline_valid",
            "quality_flag": "baseline_quality_flag",
            "x_centroid": "baseline_x_centroid",
            "y_centroid": "baseline_y_centroid",
        }
    )

    scarlet = scarlet.rename(
        columns={
            "experiment_id": "scarlet_experiment_id",
            "observation_time": "scarlet_time",
            "net_flux": "scarlet_net_flux",
            "instrumental_magnitude": "scarlet_magnitude",
            "background_level": "scarlet_background_level",
            "background_flux": "scarlet_background_flux",
            "background_fraction": "scarlet_background_fraction",
            "valid": "scarlet_valid",
            "quality_flag": "scarlet_quality_flag",
            "x_centroid": "scarlet_x_centroid",
            "y_centroid": "scarlet_y_centroid",
        }
    )

    columns = PAIR_COLUMNS + [
        column
        for column in baseline.columns
        if column not in PAIR_COLUMNS
    ]
    baseline = baseline[columns]

    scarlet_columns = PAIR_COLUMNS + [
        column
        for column in scarlet.columns
        if column not in PAIR_COLUMNS
    ]
    scarlet = scarlet[scarlet_columns]

    paired = baseline.merge(
        scarlet,
        on=PAIR_COLUMNS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    paired["pair_status"] = paired["_merge"].map(
        {
            "both": "paired",
            "left_only": "baseline_only",
            "right_only": "scarlet_only",
        }
    )
    paired = paired.drop(columns=["_merge"])

    return paired.sort_values(
        ["target_name", "source_identifier", "cadence_index"],
        kind="stable",
    ).reset_index(drop=True)
