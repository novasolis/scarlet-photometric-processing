import pandas as pd


# ---------------------------------------------------------------------------
# Required source columns
# ---------------------------------------------------------------------------

LOCAL_FIELD_REQUIRED_COLUMNS = [
    "experiment_id",
    "target_name",
    "survey",
    "experiment_type",
    "pipeline",
    "source_identifier",
    "is_target_source",
]


EXPERIMENT_METADATA_COLUMNS = [
    "experiment_id",
    "target_name",
    "survey",
    "experiment_type",
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
    "aperture_radius",
    "annulus_inner_radius",
    "annulus_outer_radius",
]


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------


def extract_target_measurements(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the designated target source from each experiment/pipeline.

    Returns one row per experiment/pipeline.
    """

    missing_columns = [
        column
        for column in ["is_target_source"]
        if column not in measurements.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot extract target measurements. "
            f"Missing columns: {missing_columns}"
        )

    target = measurements[
        measurements["is_target_source"]
    ].copy()

    return target.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------


def validate_target_dataset(
    target: pd.DataFrame,
) -> None:
    """
    Validate the target-level analytical subset.

    No fixed experiment count is assumed. The validator instead checks
    that every experiment/pipeline contains exactly one target and that
    Baseline and SCARLET contain the same experiment set.
    """

    required_columns = [
        "experiment_id",
        "target_name",
        "survey",
        "experiment_type",
        "pipeline",
        "source_identifier",
        "is_target_source",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in target.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing target-dataset columns: "
            f"{missing_columns}"
        )

    if target.empty:
        raise ValueError("Target dataset is empty.")

    if not target["is_target_source"].all():
        raise ValueError(
            "Target dataset contains rows that are not marked as target sources."
        )

    duplicate_keys = target.duplicated(
        subset=["experiment_id", "pipeline"]
    )

    if duplicate_keys.any():
        raise ValueError(
            "Duplicate experiment/pipeline target measurements found."
        )

    pipelines = set(target["pipeline"].dropna().unique())
    expected_pipelines = {"Baseline", "SCARLET"}

    if pipelines != expected_pipelines:
        raise ValueError(
            "Unexpected pipelines in target dataset: "
            f"{pipelines}"
        )

    logical_keys = [
        "target_name",
        "survey",
        "experiment_type",
    ]

    baseline_experiments = set(
        map(
            tuple,
            target.loc[
                target["pipeline"] == "Baseline",
                logical_keys,
            ].itertuples(index=False, name=None),
        )
    )

    scarlet_experiments = set(
        map(
            tuple,
            target.loc[
                target["pipeline"] == "SCARLET",
                logical_keys,
            ].itertuples(index=False, name=None),
        )
    )

    if baseline_experiments != scarlet_experiments:
        raise ValueError(
            "Baseline and SCARLET target experiment sets do not match. "
            f"Missing in SCARLET: {sorted(baseline_experiments - scarlet_experiments)}; "
            f"Missing in Baseline: {sorted(scarlet_experiments - baseline_experiments)}"
        )


# ---------------------------------------------------------------------------
# Local-field extraction
# ---------------------------------------------------------------------------


def extract_local_field_measurements(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the complete studied local field for every experiment/pipeline.

    This intentionally retains the target and all neighbouring sources.
    """

    missing_columns = [
        column
        for column in LOCAL_FIELD_REQUIRED_COLUMNS
        if column not in measurements.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot extract local-field measurements. "
            f"Missing columns: {missing_columns}"
        )

    local_field = measurements.copy().reset_index(drop=True)

    return local_field


# ---------------------------------------------------------------------------
# Local-field validation
# ---------------------------------------------------------------------------


def validate_local_field_dataset(
    local_field: pd.DataFrame,
) -> None:
    """
    Validate the complete TARGET_PHOTOMETRY local-field dataset.

    The validator deliberately does not assume that every field contains
    exactly 30 successful measurements. The production pipeline may omit
    individual unsuccessful source measurements. What matters analytically
    is that source identity is unique and Baseline/SCARLET are structurally
    pairable.
    """

    missing_columns = [
        column
        for column in LOCAL_FIELD_REQUIRED_COLUMNS
        if column not in local_field.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required local-field columns: "
            f"{missing_columns}"
        )

    if local_field.empty:
        raise ValueError("Local-field dataset is empty.")

    duplicate_keys = local_field.duplicated(
        subset=[
            "experiment_id",
            "pipeline",
            "source_identifier",
        ]
    )

    if duplicate_keys.any():
        raise ValueError(
            "Duplicate experiment/pipeline/source measurements found."
        )

    pipelines = set(local_field["pipeline"].dropna().unique())
    expected_pipelines = {"Baseline", "SCARLET"}

    if pipelines != expected_pipelines:
        raise ValueError(
            "Unexpected pipelines in local-field dataset: "
            f"{pipelines}"
        )

    target_counts = (
        local_field[
            local_field["is_target_source"]
        ]
        .groupby(["experiment_id", "pipeline"])
        .size()
    )

    if target_counts.empty or not (target_counts == 1).all():
        raise ValueError(
            "Each experiment/pipeline must contain exactly one target source."
        )

    baseline = local_field[
        local_field["pipeline"] == "Baseline"
    ]
    scarlet = local_field[
        local_field["pipeline"] == "SCARLET"
    ]

    baseline_keys = set(
        zip(
            baseline["target_name"],
            baseline["survey"],
            baseline["experiment_type"],
        )
    )

    scarlet_keys = set(
        zip(
            scarlet["target_name"],
            scarlet["survey"],
            scarlet["experiment_type"],
        )
    )

    if baseline_keys != scarlet_keys:
        missing_in_scarlet = sorted(baseline_keys - scarlet_keys)
        missing_in_baseline = sorted(scarlet_keys - baseline_keys)
        raise ValueError(
            "Baseline and SCARLET experiment sets do not match. "
            f"Missing in SCARLET: {missing_in_scarlet}; "
            f"Missing in Baseline: {missing_in_baseline}"
        )

    # Every logical experiment/pipeline must have at least one source.
    source_counts = (
        local_field
        .groupby(["target_name", "survey", "experiment_type", "pipeline"])
        .size()
    )

    if (source_counts <= 0).any():
        raise ValueError(
            "Every experiment/pipeline must contain at least one source."
        )


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------


def attach_experiment_metadata(
    data: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach experiment-level metadata to target or local-field rows.

    The same function is used for both analytical layers. A target dataset
    is a one-row-per-experiment/pipeline subset, while a local-field dataset
    contains multiple source rows per experiment/pipeline; therefore the
    merge is validated as many-to-one.
    """

    missing_summary_columns = [
        column
        for column in EXPERIMENT_METADATA_COLUMNS
        if column not in summary.columns
    ]

    if missing_summary_columns:
        raise ValueError(
            "Missing experiment-summary metadata columns: "
            f"{missing_summary_columns}"
        )

    metadata = summary[EXPERIMENT_METADATA_COLUMNS].copy()

    if metadata["experiment_id"].duplicated().any():
        raise ValueError(
            "Experiment summary contains duplicate experiment IDs."
        )

    merge_keys = [
        "experiment_id",
        "target_name",
        "survey",
        "experiment_type",
    ]

    missing_data_keys = [
        column
        for column in merge_keys
        if column not in data.columns
    ]

    if missing_data_keys:
        raise ValueError(
            "Cannot attach experiment metadata. "
            f"Missing data columns: {missing_data_keys}"
        )

    enriched = data.merge(
        metadata,
        on=merge_keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_summary"),
    )

    return enriched


# ---------------------------------------------------------------------------
# Enriched dataset validation
# ---------------------------------------------------------------------------


def validate_enriched_target_dataset(
    target: pd.DataFrame,
) -> None:
    """Validate experiment metadata attached to target rows."""

    validate_target_dataset(target)
    validate_enriched_local_field_dataset(target)


def validate_enriched_local_field_dataset(
    local_field: pd.DataFrame,
) -> None:
    """
    Validate experiment metadata attached to local-field rows.
    """

    required_metadata = [
        "detected_source_count",
        "target_source_identifier",
        "crowded_source_count",
        "crowding_fraction",
        "crowding_radius",
        "aperture_radius",
        "annulus_inner_radius",
        "annulus_outer_radius",
    ]

    missing_columns = [
        column
        for column in required_metadata
        if column not in local_field.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing local-field metadata columns: "
            f"{missing_columns}"
        )

    null_columns = [
        column
        for column in required_metadata
        if local_field[column].isna().any()
    ]

    if null_columns:
        raise ValueError(
            "Missing experiment metadata detected in: "
            f"{null_columns}"
        )

    # The target identifier recorded in the experiment summary must agree
    # with the actual designated target source in each experiment/pipeline.
    # The target identifier recorded in the experiment summary
# must agree with the actual designated target source.

    target_rows = local_field[
        local_field["is_target_source"]
    ]

    identifier_match = (
        target_rows["source_identifier"].astype(str)
        == target_rows["target_source_identifier"].astype(str)
    )

    if not bool(identifier_match.all()):
        raise ValueError(
            "Target source identifier mismatch detected."
        )


# ---------------------------------------------------------------------------
# Backward-compatible helper
# ---------------------------------------------------------------------------


def attach_experiment_metadata_to_local_field(
    local_field: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compatibility wrapper for the consolidated metadata function."""

    return attach_experiment_metadata(local_field, summary)
