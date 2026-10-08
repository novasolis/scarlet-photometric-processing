import pandas as pd


# ---------------------------------------------------------------------------
# Baseline / SCARLET source-level pairing
# ---------------------------------------------------------------------------

PAIR_KEYS = [
    "target_name",
    "survey",
    "experiment_type",
    "source_identifier",
]


PAIR_COLUMNS = [
    "experiment_id",
    "is_target_source",
    "net_flux",
    "instrumental_magnitude",
    "valid",
    "quality_flag",
    "background_level",
    "background_flux",
    "background_fraction",
    "aperture_coverage",
    "annulus_coverage",
    "detected_source_count",
    "crowded_source_count",
    "crowding_fraction",
    "crowding_radius",
    "aperture_radius",
    "annulus_inner_radius",
    "annulus_outer_radius",
]


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def build_paired_dataset(
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pair Baseline and SCARLET measurements for every studied source.

    One row represents one source in one logical experiment.

    The target is retained as a source-level flag rather than being treated
    as the only analytical observation.

    Sources with invalid or incomplete measurements are retained in the
    paired dataset. The ``paired_valid`` flag identifies sources for which
    both Baseline and SCARLET measurements are analytically usable.
    """

    required_columns = [
        *PAIR_KEYS,
        "pipeline",
        *PAIR_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in canonical.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot build paired dataset. Missing columns: "
            f"{missing_columns}"
        )

    # ------------------------------------------------------------------
    # Split pipelines
    # ------------------------------------------------------------------

    baseline = canonical[
        canonical["pipeline"] == "Baseline"
    ].copy()

    scarlet = canonical[
        canonical["pipeline"] == "SCARLET"
    ].copy()

    if baseline.empty:
        raise ValueError(
            "No Baseline measurements found."
        )

    if scarlet.empty:
        raise ValueError(
            "No SCARLET measurements found."
        )

    # ------------------------------------------------------------------
    # Validate source-level uniqueness within each pipeline
    # ------------------------------------------------------------------

    baseline_duplicates = baseline.duplicated(
        subset=PAIR_KEYS
    )

    if baseline_duplicates.any():
        raise ValueError(
            "Duplicate Baseline source keys found before pairing."
        )

    scarlet_duplicates = scarlet.duplicated(
        subset=PAIR_KEYS
    )

    if scarlet_duplicates.any():
        raise ValueError(
            "Duplicate SCARLET source keys found before pairing."
        )

    # ------------------------------------------------------------------
    # Compare source populations
    # ------------------------------------------------------------------

    baseline_keys = set(
        map(
            tuple,
            baseline[PAIR_KEYS].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    scarlet_keys = set(
        map(
            tuple,
            scarlet[PAIR_KEYS].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    if baseline_keys != scarlet_keys:
        missing_in_scarlet = sorted(
            baseline_keys - scarlet_keys
        )

        missing_in_baseline = sorted(
            scarlet_keys - baseline_keys
        )

        raise ValueError(
            "Baseline and SCARLET source sets do not match. "
            f"Missing in SCARLET: {missing_in_scarlet}; "
            f"Missing in Baseline: {missing_in_baseline}"
        )

    # ------------------------------------------------------------------
    # Select pairing columns
    # ------------------------------------------------------------------

    baseline_columns = PAIR_KEYS + [
        column
        for column in PAIR_COLUMNS
    ]

    scarlet_columns = PAIR_KEYS + [
        column
        for column in PAIR_COLUMNS
    ]

    baseline = baseline[baseline_columns]
    scarlet = scarlet[scarlet_columns]

    # ------------------------------------------------------------------
    # Pair Baseline and SCARLET source measurements
    # ------------------------------------------------------------------

    paired = baseline.merge(
        scarlet,
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_baseline",
            "_scarlet",
        ),
    )

    # ------------------------------------------------------------------
    # Verify target-source consistency
    # ------------------------------------------------------------------

    paired["is_target_source"] = (
        paired["is_target_source_baseline"]
        .fillna(False)
        .astype(bool)
    )

    if not (
        paired["is_target_source_baseline"]
        == paired["is_target_source_scarlet"]
    ).all():
        raise ValueError(
            "Baseline and SCARLET target-source flags disagree."
        )

    # ------------------------------------------------------------------
    # Source role
    # ------------------------------------------------------------------

    paired["source_role"] = paired[
        "is_target_source"
    ].map(
        {
            True: "Target",
            False: "Neighbour",
        }
    )

    # ------------------------------------------------------------------
    # Paired analytical validity
    # ------------------------------------------------------------------
    #
    # A source is analytically pair-valid only when both pipelines:
    #
    #   1. mark the measurement as valid
    #   2. contain a net flux
    #   3. contain an instrumental magnitude
    #
    # Invalid/incomplete sources remain in the paired dataset.

    paired["paired_valid"] = (
        paired["valid_baseline"].fillna(False).astype(bool)
        & paired["valid_scarlet"].fillna(False).astype(bool)
        & paired["net_flux_baseline"].notna()
        & paired["net_flux_scarlet"].notna()
        & paired["instrumental_magnitude_baseline"].notna()
        & paired["instrumental_magnitude_scarlet"].notna()
    )

    # ------------------------------------------------------------------
    # Pair-quality classification
    # ------------------------------------------------------------------

    paired["pair_quality_flag"] = "VALID"

    paired.loc[
        ~paired["paired_valid"],
        "pair_quality_flag",
    ] = "INCOMPLETE"

    # ------------------------------------------------------------------
    # Stable ordering
    # ------------------------------------------------------------------

    paired = paired.sort_values(
        by=[
            "survey",
            "target_name",
            "source_identifier",
        ]
    ).reset_index(drop=True)

    return paired


# ---------------------------------------------------------------------------
# Pair validation
# ---------------------------------------------------------------------------

def validate_paired_dataset(
    paired: pd.DataFrame,
) -> None:
    """
    Validate the complete source-level paired dataset.

    The paired dataset may contain incomplete or invalid measurements.
    Such rows are retained for source-level quality analysis and are
    identified using ``paired_valid``.

    Only sources with paired_valid=True are suitable for comparative
    photometric metrics.
    """

    required_columns = [
        *PAIR_KEYS,
        "experiment_id_baseline",
        "experiment_id_scarlet",
        "is_target_source",
        "is_target_source_baseline",
        "is_target_source_scarlet",
        "source_role",
        "valid_baseline",
        "valid_scarlet",
        "net_flux_baseline",
        "net_flux_scarlet",
        "instrumental_magnitude_baseline",
        "instrumental_magnitude_scarlet",
        "paired_valid",
        "pair_quality_flag",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in paired.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing paired-dataset columns: "
            f"{missing_columns}"
        )

    if paired.empty:
        raise ValueError(
            "Paired dataset is empty."
        )

    # ------------------------------------------------------------------
    # Source uniqueness
    # ------------------------------------------------------------------

    duplicate_keys = paired.duplicated(
        subset=PAIR_KEYS
    )

    if duplicate_keys.any():
        raise ValueError(
            "Duplicate source pairs found."
        )

    # ------------------------------------------------------------------
    # Target-source consistency
    # ------------------------------------------------------------------

    if not (
        paired["is_target_source_baseline"]
        == paired["is_target_source_scarlet"]
    ).all():
        raise ValueError(
            "Baseline and SCARLET target-source flags disagree."
        )

    if not (
        paired["is_target_source"]
        == paired["is_target_source_baseline"]
    ).all():
        raise ValueError(
            "Consolidated target-source flag does not agree "
            "with Baseline target-source flag."
        )

    # ------------------------------------------------------------------
    # Source-role consistency
    # ------------------------------------------------------------------

    expected_roles = paired[
        "is_target_source"
    ].map(
        {
            True: "Target",
            False: "Neighbour",
        }
    )

    if not (
        paired["source_role"]
        == expected_roles
    ).all():
        raise ValueError(
            "Source-role values are inconsistent with "
            "target-source flags."
        )

    # ------------------------------------------------------------------
    # Every logical experiment must contain exactly one target
    # ------------------------------------------------------------------

    target_counts = (
        paired[
            paired["is_target_source"]
        ]
        .groupby(
            [
                "target_name",
                "survey",
                "experiment_type",
            ]
        )
        .size()
    )

    if target_counts.empty:
        raise ValueError(
            "No paired target sources found."
        )

    if not (
        target_counts == 1
    ).all():
        raise ValueError(
            "Each logical experiment must contain exactly "
            "one paired target."
        )

    # ------------------------------------------------------------------
    # Paired-valid consistency
    # ------------------------------------------------------------------

    expected_paired_valid = (
        paired["valid_baseline"].fillna(False).astype(bool)
        & paired["valid_scarlet"].fillna(False).astype(bool)
        & paired["net_flux_baseline"].notna()
        & paired["net_flux_scarlet"].notna()
        & paired["instrumental_magnitude_baseline"].notna()
        & paired["instrumental_magnitude_scarlet"].notna()
    )

    if not (
        paired["paired_valid"]
        == expected_paired_valid
    ).all():
        raise ValueError(
            "paired_valid flags are inconsistent with "
            "the underlying source measurements."
        )

    # ------------------------------------------------------------------
    # Pair-quality consistency
    # ------------------------------------------------------------------

    expected_quality = pd.Series(
        "INCOMPLETE",
        index=paired.index,
    )

    expected_quality.loc[
        paired["paired_valid"]
    ] = "VALID"

    if not (
        paired["pair_quality_flag"]
        == expected_quality
    ).all():
        raise ValueError(
            "pair_quality_flag values are inconsistent "
            "with paired_valid."
        )

    # ------------------------------------------------------------------
    # Valid paired measurements must contain all photometric values
    # ------------------------------------------------------------------

    valid_pairs = paired[
        paired["paired_valid"]
    ]

    photometric_columns = [
        "net_flux_baseline",
        "net_flux_scarlet",
        "instrumental_magnitude_baseline",
        "instrumental_magnitude_scarlet",
    ]

    for column in photometric_columns:
        if valid_pairs[column].isna().any():
            raise ValueError(
                "paired_valid=True but missing values found "
                f"in paired column '{column}'."
            )

    # ------------------------------------------------------------------
    # Validity flags must not be missing
    # ------------------------------------------------------------------

    for column in [
        "valid_baseline",
        "valid_scarlet",
    ]:
        if paired[column].isna().any():
            raise ValueError(
                f"Missing validity flags found in '{column}'."
            )

    # ------------------------------------------------------------------
    # Experiment IDs must exist
    # ------------------------------------------------------------------

    for column in [
        "experiment_id_baseline",
        "experiment_id_scarlet",
    ]:
        if paired[column].isna().any():
            raise ValueError(
                f"Missing experiment IDs found in '{column}'."
            )

    # ------------------------------------------------------------------
    # Pipeline-specific experiment IDs must be unique within each
    # logical experiment/source pair
    # ------------------------------------------------------------------

    if (
        paired[
            "experiment_id_baseline"
        ].duplicated(
            keep=False
        )
        & paired[
            "source_identifier"
        ].isna()
    ).any():
        raise ValueError(
            "Invalid source identifiers found in paired dataset."
        )