"""Input loading and canonicalisation for differential photometry."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


DIFFERENTIAL_REQUIRED_COLUMNS = {
    "experiment_id",
    "target_name",
    "survey",
    "experiment_type",
    "pipeline",
    "cadence_index",
    "observation_time",
    "target_net_flux",
    "reference_net_fluxes",
    "reference_flux",
    "differential_flux",
    "differential_magnitude",
}


def _parse_list(value):
    """Parse list-like CSV cells produced by the experiment exporter."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            raise ValueError(f"Unable to parse list-like value: {value!r}") from None


def load_differential_measurements(csv_path: str | Path) -> pd.DataFrame:
    """Load and lightly canonicalise differential time-series measurements."""
    df = pd.read_csv(csv_path)

    missing = DIFFERENTIAL_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"differential_timeseries_measurements.csv is missing columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()
    df["cadence_index"] = pd.to_numeric(df["cadence_index"], errors="raise").astype(int)
    df["observation_time"] = pd.to_numeric(df["observation_time"], errors="raise")

    numeric_columns = [
        "target_net_flux",
        "reference_flux",
        "differential_flux",
        "differential_magnitude",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df["reference_net_fluxes"] = df["reference_net_fluxes"].map(_parse_list)
    df["reference_count"] = df["reference_net_fluxes"].map(len)

    return df.sort_values(
        ["target_name", "pipeline", "cadence_index"]
    ).reset_index(drop=True)


def load_target_timeseries(csv_path: str | Path) -> pd.DataFrame:
    """Load target rows from the ordinary TESS time-series measurements."""
    df = pd.read_csv(csv_path)

    required = {
        "experiment_id",
        "target_name",
        "pipeline",
        "cadence_index",
        "is_target_source",
        "net_flux",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"timeseries_measurements.csv is missing columns: {sorted(missing)}"
        )

    df = df[df["is_target_source"].astype(bool)].copy()
    df["cadence_index"] = pd.to_numeric(df["cadence_index"], errors="raise").astype(int)
    df["net_flux"] = pd.to_numeric(df["net_flux"], errors="raise")

    return df.sort_values(
        ["target_name", "pipeline", "cadence_index"]
    ).reset_index(drop=True)
