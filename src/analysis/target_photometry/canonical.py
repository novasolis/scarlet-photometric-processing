import pandas as pd


# ---------------------------------------------------------------------------
# Canonical TARGET_PHOTOMETRY analytical schema
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    # Experiment identity
    "experiment_id",
    "target_name",
    "survey",
    "experiment_type",
    "pipeline",

    # Source identity
    "source_identifier",
    "is_target_source",

    # Photometry
    "net_flux",
    "instrumental_magnitude",

    # Measurement quality
    "valid",
    "quality_flag",

    # Background / aperture information
    "background_level",
    "background_flux",
    "background_fraction",
    "aperture_coverage",
    "annulus_coverage",

    # Experiment-level environment
    "detected_source_count",
    "crowded_source_count",
    "crowding_fraction",
    "crowding_radius",

    # Aperture configuration
    "aperture_radius",
    "annulus_inner_radius",
    "annulus_outer_radius",

    # Experiment-level measurement completeness
    "valid_measurement_count",
    "valid_measurement_fraction",
    "invalid_measurement_count",
    "missing_measurement_count",
    "non_positive_flux_count",
]


# ---------------------------------------------------------------------------
# Canonical dataset construction
# ---------------------------------------------------------------------------

def build_canonical_dataset(
    local_field: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical source-level analytical dataset.

    The canonical dataset contains every source processed within each
    TARGET_PHOTOMETRY experiment and pipeline, including the designated
    target source and neighbouring sources.

    Invalid or incomplete source measurements are retained so that
    measurement completeness and source-level quality can be analysed.
    """

    missing_columns = [
        column
        for column in CANONICAL_COLUMNS
        if column not in local_field.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot build canonical dataset. Missing columns: "
            f"{missing_columns}"
        )

    canonical = local_field[
        CANONICAL_COLUMNS
    ].copy()

    return canonical.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Canonical dataset validation
# ---------------------------------------------------------------------------

def validate_canonical_dataset(
    canonical: pd.DataFrame,
) -> None:
    """
    Validate the complete source-level canonical dataset.

    Validation is deliberately source-level rather than target-level.
    Invalid source measurements may contain missing photometric values,
    but valid measurements must contain the required photometric outputs.
    """

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    if list(canonical.columns) != CANONICAL_COLUMNS:
        raise ValueError(
            "Canonical column order does not match the defined schema."
        )

    if canonical.empty:
        raise ValueError(
            "Canonical dataset is empty."
        )

    # ------------------------------------------------------------------
    # Source uniqueness
    # ------------------------------------------------------------------

    duplicate_keys = canonical.duplicated(
        subset=[
            "experiment_id",
            "pipeline",
            "source_identifier",
        ]
    )

    if duplicate_keys.any():
        raise ValueError(
            "Duplicate experiment/pipeline/source rows found "
            "in canonical dataset."
        )

    # ------------------------------------------------------------------
    # Pipeline validation
    # ------------------------------------------------------------------

    pipelines = set(
        canonical["pipeline"]
        .dropna()
        .unique()
    )

    expected_pipelines = {
        "Baseline",
        "SCARLET",
    }

    if pipelines != expected_pipelines:
        raise ValueError(
            "Unexpected pipelines in canonical dataset: "
            f"{pipelines}"
        )

    # ------------------------------------------------------------------
    # Required identity fields
    # ------------------------------------------------------------------

    identity_columns = [
        "experiment_id",
        "target_name",
        "survey",
        "experiment_type",
        "pipeline",
        "source_identifier",
        "is_target_source",
    ]

    missing_identity = [
        column
        for column in identity_columns
        if canonical[column].isna().any()
    ]

    if missing_identity:
        raise ValueError(
            "Missing identity values found in canonical dataset: "
            f"{missing_identity}"
        )

    # ------------------------------------------------------------------
    # Measurement validity
    # ------------------------------------------------------------------

    if canonical["valid"].isna().any():
        raise ValueError(
            "Missing validity flags found in canonical dataset."
        )

    valid_rows = canonical[
        canonical["valid"]
    ]

    # Valid measurements must contain photometric values.

    missing_valid_flux = valid_rows[
        valid_rows["net_flux"].isna()
    ]

    if not missing_valid_flux.empty:
        columns = [
            "experiment_id",
            "pipeline",
            "source_identifier",
            "quality_flag",
        ]

        raise ValueError(
            "Valid source measurements contain missing net-flux values:\n"
            f"{missing_valid_flux[columns].to_string(index=False)}"
        )

    missing_valid_magnitude = valid_rows[
        valid_rows["instrumental_magnitude"].isna()
    ]

    if not missing_valid_magnitude.empty:
        columns = [
            "experiment_id",
            "pipeline",
            "source_identifier",
            "quality_flag",
        ]

        raise ValueError(
            "Valid source measurements contain missing "
            "instrumental-magnitude values:\n"
            f"{missing_valid_magnitude[columns].to_string(index=False)}"
        )

    # ------------------------------------------------------------------
    # Target-source validation
    # ------------------------------------------------------------------

    target_rows = canonical[
        canonical["is_target_source"]
    ]

    if target_rows.empty:
        raise ValueError(
            "No target sources found in canonical dataset."
        )

    target_counts = (
        target_rows
        .groupby(
            [
                "experiment_id",
                "pipeline",
            ]
        )
        .size()
    )

    if not (
        target_counts == 1
    ).all():
        raise ValueError(
            "Each experiment/pipeline must contain exactly "
            "one target source."
        )

    # ------------------------------------------------------------------
    # Experiment/pipeline completeness
    # ------------------------------------------------------------------

    experiment_pipeline_counts = (
        canonical
        .groupby(
            [
                "experiment_id",
                "pipeline",
            ]
        )
        .size()
    )

    if (
        experiment_pipeline_counts <= 0
    ).any():
        raise ValueError(
            "Every experiment/pipeline must contain at least "
            "one source measurement."
        )

    # ------------------------------------------------------------------
    # Target validity
    # ------------------------------------------------------------------

    invalid_targets = target_rows[
        ~target_rows["valid"]
    ]

    if not invalid_targets.empty:
        columns = [
            "experiment_id",
            "target_name",
            "pipeline",
            "source_identifier",
            "quality_flag",
        ]

        raise ValueError(
            "Invalid target measurements detected:\n"
            f"{invalid_targets[columns].to_string(index=False)}"
        )