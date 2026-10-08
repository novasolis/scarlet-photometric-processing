"""Input/output boundary for the TESS time-series photometry analysis.

The production time-series experiment is an experiment set consisting of
three related exports:

* experiment_summary.csv
* source_measurements.csv
* timeseries_measurements.csv

This module owns file discovery, loading, and basic schema validation.
Scientific calculations belong to the other modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd


EXPERIMENT_SUMMARY_FILE: Final[str] = "experiment_summary.csv"
SOURCE_MEASUREMENTS_FILE: Final[str] = "source_measurements.csv"
TIMESERIES_MEASUREMENTS_FILE: Final[str] = "timeseries_measurements.csv"


class InputSchemaError(ValueError):
    """Raised when a production input does not satisfy the required schema."""


@dataclass(frozen=True, slots=True)
class ExperimentSet:
    """The three persisted datasets belonging to one time-series experiment set."""

    experiment_summary: pd.DataFrame
    source_measurements: pd.DataFrame
    timeseries_measurements: pd.DataFrame
    input_directory: Path


def _require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise InputSchemaError(
            f"{name} is missing required columns: {', '.join(missing)}"
        )


def load_csv(path: Path) -> pd.DataFrame:
    """Load one CSV without silently accepting an absent file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise InputSchemaError(f"Required input file is empty: {path}")

    return dataframe


def load_experiment_set(
    input_directory: Path | str,
    *,
    validate_schema: bool = True,
) -> ExperimentSet:
    """Load all three production exports from one experiment-set directory."""
    directory = Path(input_directory).expanduser().resolve()

    summary = load_csv(directory / EXPERIMENT_SUMMARY_FILE)
    sources = load_csv(directory / SOURCE_MEASUREMENTS_FILE)
    timeseries = load_csv(directory / TIMESERIES_MEASUREMENTS_FILE)

    if validate_schema:
        _require_columns(
            summary,
            {
                "experiment_id",
                "target_name",
                "survey",
                "experiment_type",
                "pipeline",
            },
            "experiment_summary.csv",
        )

        _require_columns(
            sources,
            {
                "experiment_id",
                "target_name",
                "survey",
                "experiment_type",
                "pipeline",
                "source_identifier",
                "is_target_source",
                "net_flux",
                "instrumental_magnitude",
                "valid",
            },
            "source_measurements.csv",
        )

        _require_columns(
            timeseries,
            {
                "experiment_id",
                "target_name",
                "survey",
                "experiment_type",
                "pipeline",
                "cadence_index",
                "observation_time",
                "source_identifier",
                "is_target_source",
                "net_flux",
                "instrumental_magnitude",
                "valid",
            },
            "timeseries_measurements.csv",
        )

    return ExperimentSet(
        experiment_summary=summary,
        source_measurements=sources,
        timeseries_measurements=timeseries,
        input_directory=directory,
    )


def write_csv(dataframe: pd.DataFrame, path: Path | str) -> Path:
    """Write a derived analysis table deterministically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)
    return destination
