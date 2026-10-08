import pandas as pd


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

REQUIRED_SUMMARY_COLUMNS = {
    "experiment_id",
    "target_name",
    "survey",
    "experiment_type",
    "pipeline",
    "detected_source_count",
    "target_source_identifier",
    "valid_measurement_count",
    "valid_measurement_fraction",
    "invalid_measurement_count",
    "missing_measurement_count",
    "non_positive_flux_count",
    "crowded_source_count",
    "crowding_fraction",
    "crowding_radius",
}


REQUIRED_MEASUREMENT_COLUMNS = {
    "experiment_id",
    "target_name",
    "survey",
    "experiment_type",
    "pipeline",
    "source_identifier",
    "is_target_source",
    "net_flux",
    "instrumental_magnitude",
    "background_level",
    "background_flux",
    "background_fraction",
    "aperture_coverage",
    "annulus_coverage",
    "valid",
    "quality_flag",
}


# ---------------------------------------------------------------------------
# Experiment pairing keys
# ---------------------------------------------------------------------------

EXPERIMENT_PAIR_COLUMNS = [
    "target_name",
    "survey",
    "experiment_type",
]


EXPECTED_PIPELINES = {
    "Baseline",
    "SCARLET",
}


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

def validate_required_columns(
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
) -> None:
    """Verify that all required columns are present."""

    missing_summary = (
        REQUIRED_SUMMARY_COLUMNS
        - set(summary.columns)
    )

    missing_measurements = (
        REQUIRED_MEASUREMENT_COLUMNS
        - set(measurements.columns)
    )

    if missing_summary:
        raise ValueError(
            "Missing required experiment-summary columns: "
            f"{sorted(missing_summary)}"
        )

    if missing_measurements:
        raise ValueError(
            "Missing required source-measurement columns: "
            f"{sorted(missing_measurements)}"
        )


# ---------------------------------------------------------------------------
# Experiment-level validation
# ---------------------------------------------------------------------------

def validate_experiment_count(
    summary: pd.DataFrame,
) -> None:
    """
    Verify that the summary contains an even number of experiments.

    TARGET_PHOTOMETRY contains one Baseline and one SCARLET experiment
    for each target/survey/experiment-type combination.
    """

    actual = len(summary)

    if actual == 0:
        raise ValueError(
            "Experiment summary contains no experiments."
        )

    if actual % 2 != 0:
        raise ValueError(
            "Expected an even number of experiments for "
            "Baseline/SCARLET pairing, "
            f"found {actual}."
        )


def validate_experiment_ids(
    summary: pd.DataFrame,
) -> None:
    """Verify that experiment IDs are unique."""

    duplicate_count = (
        summary["experiment_id"]
        .duplicated()
        .sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate experiment IDs."
        )


# ---------------------------------------------------------------------------
# Experiment pairing validation
# ---------------------------------------------------------------------------

def validate_experiment_pairing(
    summary: pd.DataFrame,
) -> None:
    """
    Verify that each target/survey/experiment-type combination contains
    exactly one Baseline and one SCARLET experiment.
    """

    pair_counts = (
        summary
        .groupby(EXPERIMENT_PAIR_COLUMNS, dropna=False)["pipeline"]
        .agg(
            experiment_count="size",
            pipelines=lambda values: set(values),
        )
    )

    invalid = pair_counts[
        (pair_counts["experiment_count"] != 2)
        | (pair_counts["pipelines"] != EXPECTED_PIPELINES)
    ]

    if not invalid.empty:
        raise ValueError(
            "Invalid Baseline/SCARLET experiment pairing detected:\n"
            f"{invalid.to_string()}"
        )


# ---------------------------------------------------------------------------
# Target-source validation
# ---------------------------------------------------------------------------

def validate_target_sources(
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
) -> None:
    """
    Verify that every experiment contains exactly one target source.
    """

    target_measurements = measurements[
        measurements["is_target_source"]
    ].copy()

    if len(target_measurements) != len(summary):
        raise ValueError(
            "The number of target-source measurements does not match "
            "the number of experiments: "
            f"{len(target_measurements)} vs {len(summary)}."
        )

    counts = (
        target_measurements
        .groupby("experiment_id")
        .size()
    )

    invalid = counts[counts != 1]

    if not invalid.empty:
        raise ValueError(
            "The following experiments do not contain exactly "
            "one target source:\n"
            f"{invalid.to_string()}"
        )


# ---------------------------------------------------------------------------
# Target validity validation
# ---------------------------------------------------------------------------

def validate_target_measurement_quality(
    measurements: pd.DataFrame,
) -> None:
    """Verify that all target measurements are valid."""

    target_measurements = measurements[
        measurements["is_target_source"]
    ]

    invalid = target_measurements[
        ~target_measurements["valid"]
    ]

    if not invalid.empty:
        columns = [
            "experiment_id",
            "target_name",
            "pipeline",
            "quality_flag",
        ]

        raise ValueError(
            "Invalid target measurements detected:\n"
            f"{invalid[columns].to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# Pipeline validation
# ---------------------------------------------------------------------------

def validate_pipeline_pairing(
    measurements: pd.DataFrame,
) -> None:
    """
    Verify that each target/survey/experiment-type combination has
    both Baseline and SCARLET source measurements.
    """

    target_measurements = measurements[
        measurements["is_target_source"]
    ].copy()

    pipeline_groups = (
        target_measurements
        .groupby(
            EXPERIMENT_PAIR_COLUMNS,
            dropna=False,
        )["pipeline"]
        .agg(set)
    )

    invalid = pipeline_groups[
        pipeline_groups != EXPECTED_PIPELINES
    ]

    if not invalid.empty:
        raise ValueError(
            "Invalid target pipeline pairing detected:\n"
            f"{invalid.to_string()}"
        )


# ---------------------------------------------------------------------------
# Source-level uniqueness
# ---------------------------------------------------------------------------

def validate_source_measurement_uniqueness(
    measurements: pd.DataFrame,
) -> None:
    """
    Verify that each experiment/source pair occurs once.

    experiment_id is intentionally included because Baseline and SCARLET
    are separate experiments.
    """

    duplicates = measurements.duplicated(
        subset=[
            "experiment_id",
            "source_identifier",
        ]
    )

    duplicate_count = duplicates.sum()

    if duplicate_count:
        raise ValueError(
            "Found "
            f"{duplicate_count} duplicate experiment/source "
            "measurements."
        )


# ---------------------------------------------------------------------------
# Local-field completeness
# ---------------------------------------------------------------------------

def validate_local_field_completeness(
    measurements: pd.DataFrame,
) -> None:
    """
    Verify that Baseline and SCARLET contain the same local-field
    source identifiers for every paired experiment.

    Experiments are paired using target_name, survey and
    experiment_type rather than experiment_id because Baseline
    and SCARLET have different experiment IDs.
    """

    grouped = measurements.groupby(
        EXPERIMENT_PAIR_COLUMNS,
        dropna=False,
    )

    for pair_key, group in grouped:

        pipelines = set(group["pipeline"])

        if pipelines != EXPECTED_PIPELINES:
            raise ValueError(
                "Incomplete pipeline pair for "
                f"{pair_key}: {pipelines}"
            )

        baseline = group[
            group["pipeline"] == "Baseline"
        ]

        scarlet = group[
            group["pipeline"] == "SCARLET"
        ]

        baseline_sources = set(
            baseline["source_identifier"]
        )

        scarlet_sources = set(
            scarlet["source_identifier"]
        )

        baseline_only = sorted(
            baseline_sources - scarlet_sources
        )

        scarlet_only = sorted(
            scarlet_sources - baseline_sources
        )

        if baseline_only or scarlet_only:
            raise ValueError(
                "Local-field source mismatch for "
                f"{pair_key}.\n"
                f"Baseline-only sources: {baseline_only}\n"
                f"SCARLET-only sources: {scarlet_only}"
            )


# ---------------------------------------------------------------------------
# Detected-source-count validation
# ---------------------------------------------------------------------------

def validate_detected_source_counts(
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
) -> None:
    """
    Verify that detected_source_count in the experiment summary agrees
    with the number of unique measured sources for each experiment.
    """

    measured_counts = (
        measurements
        .groupby("experiment_id")["source_identifier"]
        .nunique()
    )

    comparison = summary[
        [
            "experiment_id",
            "detected_source_count",
        ]
    ].copy()

    comparison["measured_source_count"] = (
        comparison["experiment_id"]
        .map(measured_counts)
    )

    missing = comparison[
        comparison["measured_source_count"].isna()
    ]

    if not missing.empty:
        raise ValueError(
            "Experiments with no source measurements found:\n"
            f"{missing.to_string(index=False)}"
        )

    mismatch = comparison[
        comparison["detected_source_count"]
        != comparison["measured_source_count"]
    ]

    if not mismatch.empty:
        raise ValueError(
            "Detected source count mismatch detected:\n"
            f"{mismatch.to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# Summary/source consistency
# ---------------------------------------------------------------------------

def validate_target_identifier_consistency(
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
) -> None:
    """
    Verify that target source IDs agree between the summary and
    source-measurement datasets.
    """

    target_measurements = measurements[
        measurements["is_target_source"]
    ][
        [
            "experiment_id",
            "source_identifier",
        ]
    ].copy()

    merged = summary[
        [
            "experiment_id",
            "target_source_identifier",
        ]
    ].merge(
        target_measurements,
        on="experiment_id",
        how="left",
        validate="one_to_one",
    )

    missing = merged[
        merged["source_identifier"].isna()
    ]

    if not missing.empty:
        raise ValueError(
            "Target source identifier missing from source "
            "measurements:\n"
            f"{missing.to_string(index=False)}"
        )

    mismatch = merged[
        merged["target_source_identifier"]
        != merged["source_identifier"]
    ]

    if not mismatch.empty:
        raise ValueError(
            "Target source identifier mismatch detected:\n"
            f"{mismatch.to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# Summary/source measurement-count consistency
# ---------------------------------------------------------------------------

def validate_measurement_counts(
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
) -> None:
    """
    Verify that summary measurement counts agree with the actual
    source-level measurements.
    """

    actual_counts = (
        measurements
        .groupby("experiment_id")
        .agg(
            actual_valid_count=(
                "valid",
                "sum",
            ),
            actual_invalid_count=(
                "valid",
                lambda values: (~values).sum(),
            ),
        )
    )

    comparison = summary[
        [
            "experiment_id",
            "valid_measurement_count",
            "invalid_measurement_count",
        ]
    ].merge(
        actual_counts,
        on="experiment_id",
        how="left",
        validate="one_to_one",
    )

    comparison["actual_valid_count"] = (
        comparison["actual_valid_count"]
        .fillna(0)
    )

    comparison["actual_invalid_count"] = (
        comparison["actual_invalid_count"]
        .fillna(0)
    )

    valid_mismatch = comparison[
        comparison["valid_measurement_count"]
        != comparison["actual_valid_count"]
    ]

    if not valid_mismatch.empty:
        raise ValueError(
            "Valid measurement count mismatch detected:\n"
            f"{valid_mismatch.to_string(index=False)}"
        )

    invalid_mismatch = comparison[
        comparison["invalid_measurement_count"]
        != comparison["actual_invalid_count"]
    ]

    if not invalid_mismatch.empty:
        raise ValueError(
            "Invalid measurement count mismatch detected:\n"
            f"{invalid_mismatch.to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# Top-level input validation
# ---------------------------------------------------------------------------

def validate_input_data(
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
) -> None:
    """
    Run all structural and consistency validations required before
    constructing the analytical TARGET_PHOTOMETRY datasets.
    """

    validate_required_columns(
        summary,
        measurements,
    )

    validate_experiment_count(
        summary,
    )

    validate_experiment_ids(
        summary,
    )

    validate_experiment_pairing(
        summary,
    )

    validate_target_sources(
        summary,
        measurements,
    )

    validate_target_measurement_quality(
        measurements,
    )

    validate_pipeline_pairing(
        measurements,
    )

    validate_source_measurement_uniqueness(
        measurements,
    )

    validate_local_field_completeness(
        measurements,
    )

    validate_detected_source_counts(
        summary,
        measurements,
    )

    validate_target_identifier_consistency(
        summary,
        measurements,
    )

    validate_measurement_counts(
        summary,
        measurements,
    )