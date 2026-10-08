"""Canonicalisation of the three TESS time-series input datasets."""

from __future__ import annotations

import pandas as pd

from .io import ExperimentSet


BOOLEAN_TRUE = {"true", "1", "yes", "y", "t"}
BOOLEAN_FALSE = {"false", "0", "no", "n", "f"}


def _normalise_boolean(series: pd.Series) -> pd.Series:
    """Normalise common CSV boolean representations to pandas BooleanDtype."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    values = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[values.isin(BOOLEAN_TRUE)] = True
    result.loc[values.isin(BOOLEAN_FALSE)] = False
    return result


def _normalise_common(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    for column in (
        "experiment_id",
        "target_name",
        "survey",
        "experiment_type",
        "source_scope",
        "pipeline",
        "source_identifier",
    ):
        if column in result.columns:
            result[column] = result[column].astype("string").str.strip()

    for column in ("is_target_source", "valid", "crowded"):
        if column in result.columns:
            result[column] = _normalise_boolean(result[column])

    return result


def canonicalize(experiment_set: ExperimentSet) -> ExperimentSet:
    """Return canonical copies while preserving the three-table structure."""
    summary = _normalise_common(experiment_set.experiment_summary)
    sources = _normalise_common(experiment_set.source_measurements)
    timeseries = _normalise_common(experiment_set.timeseries_measurements)

    for dataframe in (sources, timeseries):
        if "cadence_index" in dataframe.columns:
            dataframe["cadence_index"] = pd.to_numeric(
                dataframe["cadence_index"],
                errors="coerce",
            ).astype("Int64")

        for column in (
            "observation_time",
            "net_flux",
            "instrumental_magnitude",
            "background_level",
            "background_flux",
            "background_fraction",
            "aperture_coverage",
            "annulus_coverage",
            "x_centroid",
            "y_centroid",
            "target_x",
            "target_y",
            "target_source_separation_pixels",
        ):
            if column in dataframe.columns:
                dataframe[column] = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )

    return ExperimentSet(
        experiment_summary=summary,
        source_measurements=sources,
        timeseries_measurements=timeseries,
        input_directory=experiment_set.input_directory,
    )
