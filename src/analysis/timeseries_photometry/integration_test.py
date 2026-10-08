from __future__ import annotations

"""Integration tests for the complete TESS time-series experiment set.

Run from the project root:

    python -m analysis.timeseries_photometry.integration_test

The default input directory is:

    results/exports/timeseries_photometry/

The test validates all three production exports, relational integrity,
Baseline/SCARLET pairing, negative validation cases, metric generation,
and the end-to-end analysis.

Important:
    Raw TESS cadence counts may differ between targets because each target
    can have a different number of available observations.

    The controlled scientific comparison is therefore based on the paired
    time-series dataset, which is expected to contain 300 paired cadences
    per target.
"""

import argparse
import math
import tempfile
from pathlib import Path
from dataclasses import replace

import pandas as pd

from .io import ExperimentSet
from .analysis import run_analysis
from .canonical import canonicalize
from .io import load_experiment_set
from .metrics import add_normalised_flux_columns, calculate_target_metrics
from .pairing import pair_target_timeseries
from .validation import validate_structure, validation_summary


# ---------------------------------------------------------------------------
# Production experiment invariants
# ---------------------------------------------------------------------------

EXPECTED_TARGET_COUNT = 7
EXPECTED_PIPELINE_COUNT = 2
EXPECTED_EXPERIMENT_COUNT = EXPECTED_TARGET_COUNT * EXPECTED_PIPELINE_COUNT

EXPECTED_PIPELINES = {"Baseline", "SCARLET"}

# This is the controlled comparison used throughout the thesis analysis.
EXPECTED_CADENCES_PER_TARGET_PIPELINE = 300

# 7 targets × 300 paired cadences.
EXPECTED_PAIRED_TARGET_ROWS = (
    EXPECTED_TARGET_COUNT * EXPECTED_CADENCES_PER_TARGET_PIPELINE
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class IntegrationFailure(AssertionError):
    """Raised when a production integration check fails."""


def check(condition: object, message: str) -> None:
    """Assert a scalar condition while accepting NumPy/Pandas booleans."""
    if not bool(condition):
        raise IntegrationFailure(message)


# ---------------------------------------------------------------------------
# Production input validation
# ---------------------------------------------------------------------------


def test_input_set(input_directory: Path) -> ExperimentSet:
    """Load production inputs and validate their basic experiment structure."""

    data = canonicalize(load_experiment_set(input_directory))

    timeseries = data.timeseries_measurements

    # ------------------------------------------------------------------
    # Raw time-series data
    #
    # Do NOT enforce one global raw-row count.
    #
    # Different TESS targets legitimately have different numbers of
    # available cadences. The important requirements are that:
    #
    #   - data exist
    #   - all expected targets exist
    #   - both pipelines exist for every target
    #   - Baseline and SCARLET contain the same number of raw cadences
    #     for a given target
    # ------------------------------------------------------------------

    check(
        len(timeseries) > 0,
        "Time-series measurements are empty.",
    )

    check(
        timeseries["experiment_id"].nunique() == EXPECTED_EXPERIMENT_COUNT,
        (
            "Unexpected number of time-series experiments: "
            f"{timeseries['experiment_id'].nunique()} "
            f"(expected {EXPECTED_EXPERIMENT_COUNT})"
        ),
    )

    check(
        timeseries["pipeline"].dropna().astype(str).isin(EXPECTED_PIPELINES).all(),
        "Unexpected pipeline value found in time-series measurements.",
    )

    target = timeseries.loc[
        timeseries["is_target_source"]
        .fillna(False)
        .astype(bool)
    ].copy()

    check(
        len(target) > 0,
        "No target-source time-series measurements were found.",
    )

    targets = set(target["target_name"].dropna().astype(str))

    check(
        len(targets) == EXPECTED_TARGET_COUNT,
        (
            f"Expected {EXPECTED_TARGET_COUNT} targets; "
            f"found {len(targets)}: {sorted(targets)}"
        ),
    )

    # ------------------------------------------------------------------
    # Every target must have both Baseline and SCARLET.
    # ------------------------------------------------------------------

    target_pipeline_pairs = set(
        zip(
            target["target_name"].astype(str),
            target["pipeline"].astype(str),
        )
    )

    expected_pairs = {
        (target_name, pipeline)
        for target_name in targets
        for pipeline in EXPECTED_PIPELINES
    }

    check(
        target_pipeline_pairs == expected_pairs,
        (
            "Target/pipeline coverage is incomplete. "
            f"Expected {sorted(expected_pairs)}, "
            f"found {sorted(target_pipeline_pairs)}."
        ),
    )

    # ------------------------------------------------------------------
    # Raw cadence counts.
    #
    # These are intentionally NOT required to equal 300.
    # Instead, Baseline and SCARLET must have matching raw cadence counts
    # for each target so that the later pairing stage has comparable data.
    # ------------------------------------------------------------------

    raw_counts = (
        target.groupby(
            ["target_name", "pipeline"],
            dropna=False,
        )["cadence_index"]
        .nunique()
        .unstack("pipeline", fill_value=0)
    )

    check(
        set(raw_counts.columns) == EXPECTED_PIPELINES,
        "Raw cadence counts do not contain both Baseline and SCARLET.",
    )

    check(
        bool(
            (
                raw_counts["Baseline"]
                == raw_counts["SCARLET"]
            ).all()
        ),
        (
            "Baseline and SCARLET raw cadence counts differ for at least "
            "one target:\n"
            f"{raw_counts.to_string()}"
        ),
    )

    check(
        bool((raw_counts > 0).all().all()),
        (
            "At least one target/pipeline has zero raw cadences:\n"
            f"{raw_counts.to_string()}"
        ),
    )

    # ------------------------------------------------------------------
    # The current production dataset should have 7 unique target names
    # and exactly two pipelines.
    # ------------------------------------------------------------------

    check(
        target["pipeline"].nunique() == EXPECTED_PIPELINE_COUNT,
        (
            "Unexpected number of pipelines: "
            f"{target['pipeline'].nunique()}"
        ),
    )

    return data


# ---------------------------------------------------------------------------
# Structural and relational validation
# ---------------------------------------------------------------------------


def test_validation(data: ExperimentSet) -> None:
    result = validate_structure(
        data,
        expected_cadences=EXPECTED_CADENCES_PER_TARGET_PIPELINE,
    )

    summary = validation_summary(result)

    if not result.passed:
        print("\nValidation issues:")
        print(result.issues.to_string(index=False))

    check(
        result.passed,
        f"Validation failed: {summary}",
    )

    check(
        summary["errors"] == 0,
        f"Validation returned errors: {summary}",
    )


# ---------------------------------------------------------------------------
# Negative timestamp validation
# ---------------------------------------------------------------------------


def test_negative_timestamp_detection(data: ExperimentSet) -> None:
    experiment_set = data
    broken = experiment_set.timeseries_measurements.copy()

    mask = (
        broken["is_target_source"]
        .fillna(False)
        .astype(bool)
        & broken["pipeline"].astype("string").eq("SCARLET")
        & broken["cadence_index"].eq(0)
    )

    check(
        bool(mask.any()),
        "Could not locate a test SCARLET cadence.",
    )

    index = broken.index[mask][0]
    selected_mask = broken.index == index

    selected_times = pd.to_numeric(
        broken.loc[selected_mask, "observation_time"],
        errors="raise",
    )

    broken.loc[selected_mask, "observation_time"] = selected_times + 1.0

    broken_set = replace(
        experiment_set,
        timeseries_measurements=broken,
    )

    result = validate_structure(broken_set)
    codes = set(result.issues["code"].astype(str))

    check(
        "TIMESTAMP_MISMATCH" in codes,
        "Timestamp corruption was not detected.",
    )


# ---------------------------------------------------------------------------
# Negative duplicate-key validation
# ---------------------------------------------------------------------------


def test_negative_duplicate_detection(data: ExperimentSet) -> None:
    experiment_set = data

    target = experiment_set.timeseries_measurements.loc[
        experiment_set.timeseries_measurements["is_target_source"]
        .fillna(False)
        .astype(bool)
    ].copy()

    check(
        not target.empty,
        "No target time-series measurements available for duplicate test.",
    )

    duplicate = target.iloc[[0]].copy()

    broken = pd.concat(
        [
            experiment_set.timeseries_measurements,
            duplicate,
        ],
        ignore_index=True,
    )

    broken_set = replace(
        experiment_set,
        timeseries_measurements=broken,
    )

    result = validate_structure(broken_set)
    codes = set(result.issues["code"].astype(str))

    check(
        "DUPLICATE_TIMESERIES_KEYS" in codes,
        "Duplicate cadence corruption was not detected.",
    )


# ---------------------------------------------------------------------------
# Pairing and scientific metrics
# ---------------------------------------------------------------------------


def test_pairing_and_metrics(
    data: ExperimentSet,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    experiment_set = data

    paired = pair_target_timeseries(experiment_set)

    paired_count = int(
        (paired["pair_status"] == "paired").sum()
    )

    check(
        paired_count == EXPECTED_PAIRED_TARGET_ROWS,
        (
            "Unexpected paired target rows: "
            f"{paired_count}; "
            f"expected {EXPECTED_PAIRED_TARGET_ROWS}."
        ),
    )

    # ---------------------------------------------------------------
    # Confirm 300 paired cadences for each target.
    # ---------------------------------------------------------------

    paired_target = paired.loc[
        paired["pair_status"].eq("paired")
    ].copy()

    paired_counts = (
        paired_target.groupby("target_name")
        .size()
    )

    check(
        len(paired_counts) == EXPECTED_TARGET_COUNT,
        (
            "Unexpected number of targets in paired data: "
            f"{len(paired_counts)}"
        ),
    )

    check(
        bool(
            (
                paired_counts
                == EXPECTED_CADENCES_PER_TARGET_PIPELINE
            ).all()
        ),
        (
            "Not every target contains exactly "
            f"{EXPECTED_CADENCES_PER_TARGET_PIPELINE} "
            "paired cadences:\n"
            f"{paired_counts.to_string()}"
        ),
    )

    # ---------------------------------------------------------------
    # Normalised flux
    # ---------------------------------------------------------------

    paired = add_normalised_flux_columns(paired)

    for column in (
        "baseline_flux_normalised",
        "scarlet_flux_normalised",
    ):
        values = pd.to_numeric(
            paired[column],
            errors="coerce",
        )

        check(
            bool(values.notna().any()),
            f"No usable values were generated for {column}.",
        )

        check(
            bool(values.notna().all()),
            f"Missing values were generated for {column}.",
        )

    # ---------------------------------------------------------------
    # Scientific metrics
    # ---------------------------------------------------------------

    metrics = calculate_target_metrics(paired)

    check(
        len(metrics) == EXPECTED_TARGET_COUNT,
        (
            f"Expected {EXPECTED_TARGET_COUNT} metric rows; "
            f"found {len(metrics)}."
        ),
    )

    required = {
        "flux_scatter_std_change_percent",
        "magnitude_scatter_std_change_percent",
        "temporal_stability_std_diff_change_percent",
        "lightcurve_correlation",
    }

    check(
        required.issubset(metrics.columns),
        "Metric output is missing one or more required columns.",
    )

    # ---------------------------------------------------------------
    # Basic numerical sanity checks.
    # ---------------------------------------------------------------

    for column in required:
        values = pd.to_numeric(
            metrics[column],
            errors="coerce",
        )

        check(
            bool(values.notna().all()),
            f"Metric column contains missing/non-numeric values: {column}",
        )

    correlation = pd.to_numeric(
        metrics["lightcurve_correlation"],
        errors="coerce",
    )

    check(
        bool(((correlation >= -1.0) & (correlation <= 1.0)).all()),
        "Light-curve correlation contains values outside [-1, 1].",
    )

    return paired, metrics


# ---------------------------------------------------------------------------
# End-to-end analysis
# ---------------------------------------------------------------------------


def test_end_to_end(input_directory: Path) -> None:
    with tempfile.TemporaryDirectory(
        prefix="scarlet_ts_test_"
    ) as temp:

        outputs = run_analysis(
            input_directory,
            Path(temp),
            expected_cadences=EXPECTED_CADENCES_PER_TARGET_PIPELINE,
        )

        required_outputs = {
            "validation_report",
            "validation_summary",
            "experiment_context",
            "source_context",
            "target_timeseries_canonical",
            "paired_timeseries",
            "paired_timeseries_normalised",
            "target_metrics",
            "target_summary",
            "overall_summary",
            "analysis_metadata",
        }

        check(
            required_outputs.issubset(outputs),
            "End-to-end output map is incomplete.",
        )

        for path in outputs.values():
            check(
                Path(path).is_file(),
                f"Missing output file: {path}",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "results/exports/timeseries_photometry"
        ),
    )

    args = parser.parse_args()

    print(
        "[1/6] Loading and validating the three production inputs"
    )

    data = test_input_set(args.input_dir)

    print("      PASS")

    print(
        "[2/6] Structural and relational validation"
    )

    test_validation(data)

    print("      PASS")

    print(
        "[3/6] Negative timestamp validation"
    )

    test_negative_timestamp_detection(data)

    print("      PASS")

    print(
        "[4/6] Negative duplicate-key validation"
    )

    test_negative_duplicate_detection(data)

    print("      PASS")

    print(
        "[5/6] Pairing and scientific metrics"
    )

    test_pairing_and_metrics(data)

    print("      PASS")

    print(
        "[6/6] End-to-end production analysis"
    )

    test_end_to_end(args.input_dir)

    print("      PASS")

    print()
    print("INTEGRATION TEST PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())