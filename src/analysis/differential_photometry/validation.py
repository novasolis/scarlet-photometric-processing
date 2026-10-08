"""Validation checks for differential-photometry experiment outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_differential_measurements(
    df: pd.DataFrame,
    expected_targets: tuple[str, ...] | None = None,
    expected_cadences: int = 300,
) -> dict:
    """Validate completeness, finiteness, and Baseline/SCARLET pairing."""
    errors: list[str] = []

    if expected_targets is not None:
        actual = tuple(sorted(df["target_name"].unique()))
        expected = tuple(sorted(expected_targets))
        if actual != expected:
            errors.append(f"Targets differ: expected {expected}, found {actual}")

    if set(df["pipeline"].unique()) != {"Baseline", "SCARLET"}:
        errors.append("Expected exactly Baseline and SCARLET pipelines.")

    numeric_columns = [
        "target_net_flux",
        "reference_flux",
        "differential_flux",
        "differential_magnitude",
    ]
    for column in numeric_columns:
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            errors.append(f"Non-finite values found in {column}.")

    duplicate_keys = df.duplicated(
        ["target_name", "pipeline", "cadence_index"], keep=False
    )
    if duplicate_keys.any():
        errors.append("Duplicate target/pipeline/cadence rows found.")

    for target, group in df.groupby("target_name"):
        counts = group.groupby("pipeline")["cadence_index"].nunique().to_dict()

        if set(counts) != {"Baseline", "SCARLET"}:
            errors.append(f"{target} does not contain both pipelines.")
            continue

        if any(count != expected_cadences for count in counts.values()):
            errors.append(
                f"{target} does not contain {expected_cadences} cadences per pipeline: "
                f"{counts}"
            )

        baseline = set(
            group.loc[group["pipeline"] == "Baseline", "cadence_index"]
        )
        scarlet = set(
            group.loc[group["pipeline"] == "SCARLET", "cadence_index"]
        )
        if baseline != scarlet:
            errors.append(f"{target} Baseline/SCARLET cadence sets do not match.")

    result = {
        "valid": not errors,
        "errors": errors,
        "rows": int(len(df)),
        "targets": int(df["target_name"].nunique()),
        "pipelines": sorted(df["pipeline"].unique().tolist()),
    }

    if errors:
        raise ValueError("Differential-photometry validation failed:\n- " + "\n- ".join(errors))

    return result


def validate_experiment_summary(summary_df: pd.DataFrame) -> dict:
    """Confirm that the experiment summary contains the expected six runs."""
    required = {"experiment_id", "target_name", "pipeline", "experiment_type"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"Experiment summary missing columns: {sorted(missing)}")

    expected_pairs = 6
    actual_pairs = len(summary_df)

    if actual_pairs != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} differential-photometry experiments; "
            f"found {actual_pairs}."
        )

    if set(summary_df["pipeline"]) != {"Baseline", "SCARLET"}:
        raise ValueError("Experiment summary does not contain both pipelines.")

    return {
        "valid": True,
        "experiment_count": actual_pairs,
        "targets": sorted(summary_df["target_name"].unique().tolist()),
    }
