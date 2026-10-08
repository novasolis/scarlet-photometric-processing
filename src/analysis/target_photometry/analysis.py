from pathlib import Path
from typing import Any
import shutil

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    spearmanr = None

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .io import (
    load_experiment_summary,
    load_source_measurements,
)

from .validation import (
    validate_input_data,
)

from .dataset import (
    extract_local_field_measurements,
    validate_local_field_dataset,
    attach_experiment_metadata,
    validate_enriched_local_field_dataset,
)

from .canonical import (
    build_canonical_dataset,
    validate_canonical_dataset,
)

from .pairing import (
    build_paired_dataset,
    validate_paired_dataset,
)

from .metrics import (
    calculate_paired_metrics,
    validate_paired_metrics,
)

from .statistics import (
    build_descriptive_statistics,
    validate_descriptive_statistics,
)


# ===========================================================================
# Analysis output directories
# ===========================================================================

ANALYSIS_DIR = Path(
    "results/analysis/target_photometry"
)

FIGURES_TABLES_DIR = Path(
    "results/figures_and_tables/target_photometry"
)

TARGET_RESULTS_DIR = (
    FIGURES_TABLES_DIR
    / "5_3_1_target_photometric_measurements"
)

CROSS_DATASET_RESULTS_DIR = (
    FIGURES_TABLES_DIR
    / "5_5_1_cross_dataset"
)


# ===========================================================================
# Helpers
# ===========================================================================

def _require_columns(
    data: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> None:
    """Verify that an analytical dataset contains required columns."""

    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def _finite_series(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a numeric series with non-finite values converted to NaN."""

    return pd.to_numeric(
        data[column],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _reset_output_directory(
    directory: Path,
) -> None:
    """
    Remove all generated outputs from a directory and recreate it.

    This prevents stale thesis artefacts from surviving after an output
    has been renamed, removed, or replaced.
    """

    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ===========================================================================
# Target-level analytical dataset
# ===========================================================================

def build_target_analysis_dataset(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the target-only analytical view from the complete paired metrics.

    The source-level production population remains unchanged. This function
    creates a target-level view containing one paired row per logical target
    experiment.
    """

    _require_columns(
        metrics,
        [
            "target_name",
            "survey",
            "experiment_type",
            "source_identifier",
            "is_target_source",
            "source_role",
            "net_flux_baseline",
            "net_flux_scarlet",
            "instrumental_magnitude_baseline",
            "instrumental_magnitude_scarlet",
            "flux_difference",
            "flux_relative_difference_percent",
            "absolute_flux_relative_difference_percent",
            "magnitude_difference",
            "absolute_magnitude_difference",
        ],
        "Paired metrics",
    )

    target = metrics[
        metrics["is_target_source"]
    ].copy()

    target = target.sort_values(
        by=["survey", "target_name"]
    ).reset_index(drop=True)

    background_pairs = [
        (
            "background_level_baseline",
            "background_level_scarlet",
            "background_level_difference",
        ),
        (
            "background_flux_baseline",
            "background_flux_scarlet",
            "background_flux_difference",
        ),
        (
            "background_fraction_baseline",
            "background_fraction_scarlet",
            "background_fraction_difference",
        ),
    ]

    for (
        baseline_column,
        scarlet_column,
        difference_column,
    ) in background_pairs:

        if (
            baseline_column in target.columns
            and scarlet_column in target.columns
        ):
            baseline = _finite_series(
                target,
                baseline_column,
            )

            scarlet = _finite_series(
                target,
                scarlet_column,
            )

            target[difference_column] = (
                scarlet - baseline
            )

    if (
        "valid_baseline" in target.columns
        and "valid_scarlet" in target.columns
    ):
        target["validity_consistent"] = (
            target["valid_baseline"]
            .fillna(False)
            .astype(bool)
            ==
            target["valid_scarlet"]
            .fillna(False)
            .astype(bool)
        )

    if (
        "quality_flag_baseline" in target.columns
        and "quality_flag_scarlet" in target.columns
    ):
        target["quality_flag_consistent"] = (
            target["quality_flag_baseline"]
            ==
            target["quality_flag_scarlet"]
        )

    return target


# ===========================================================================
# Target-level descriptive summary
# ===========================================================================

def build_target_summary(
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Build one-row-per-target analytical summary."""

    required = [
        "target_name",
        "survey",
        "net_flux_baseline",
        "net_flux_scarlet",
        "flux_difference",
        "flux_relative_difference_percent",
        "absolute_flux_relative_difference_percent",
        "instrumental_magnitude_baseline",
        "instrumental_magnitude_scarlet",
        "magnitude_difference",
        "absolute_magnitude_difference",
    ]

    _require_columns(
        target,
        required,
        "Target analytical dataset",
    )

    columns = [
        column
        for column in [
            "target_name",
            "survey",
            "experiment_type",
            "source_identifier",
            "detected_source_count_baseline",
            "crowded_source_count_baseline",
            "crowding_fraction_baseline",
            "crowding_radius_baseline",
            "net_flux_baseline",
            "net_flux_scarlet",
            "flux_difference",
            "flux_relative_difference_percent",
            "absolute_flux_relative_difference_percent",
            "instrumental_magnitude_baseline",
            "instrumental_magnitude_scarlet",
            "magnitude_difference",
            "absolute_magnitude_difference",
            "background_level_baseline",
            "background_level_scarlet",
            "background_level_difference",
            "background_flux_baseline",
            "background_flux_scarlet",
            "background_flux_difference",
            "background_fraction_baseline",
            "background_fraction_scarlet",
            "background_fraction_difference",
            "valid_baseline",
            "valid_scarlet",
            "validity_consistent",
            "quality_flag_baseline",
            "quality_flag_scarlet",
            "quality_flag_consistent",
        ]
        if column in target.columns
    ]

    summary = target[columns].copy()

    summary["flux_change_direction"] = np.select(
        [
            summary["flux_difference"] > 0,
            summary["flux_difference"] < 0,
        ],
        [
            "SCARLET higher",
            "SCARLET lower",
        ],
        default="No change",
    )

    summary["magnitude_change_direction"] = np.select(
        [
            summary["magnitude_difference"] > 0,
            summary["magnitude_difference"] < 0,
        ],
        [
            "SCARLET higher",
            "SCARLET lower",
        ],
        default="No change",
    )

    return summary


# ===========================================================================
# Field-level analytical summary
# ===========================================================================

def build_field_summary(
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per logical target field."""

    _require_columns(
        target,
        [
            "target_name",
            "survey",
            "flux_relative_difference_percent",
            "absolute_flux_relative_difference_percent",
            "magnitude_difference",
            "absolute_magnitude_difference",
        ],
        "Target analytical dataset",
    )

    group_columns = [
        column
        for column in [
            "target_name",
            "survey",
            "experiment_type",
        ]
        if column in target.columns
    ]

    aggregation = {}

    for column in [
        "detected_source_count_baseline",
        "crowded_source_count_baseline",
        "crowding_fraction_baseline",
        "crowding_radius_baseline",
        "flux_relative_difference_percent",
        "absolute_flux_relative_difference_percent",
        "magnitude_difference",
        "absolute_magnitude_difference",
        "background_level_difference",
        "background_flux_difference",
        "background_fraction_difference",
    ]:
        if column in target.columns:
            aggregation[column] = "first"

    return (
        target
        .groupby(
            group_columns,
            as_index=False,
        )
        .agg(aggregation)
    )


# ===========================================================================
# Source-population summary
# ===========================================================================

def build_source_population_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact source-level summary for target, neighbour and survey
    populations.
    """

    _require_columns(
        metrics,
        [
            "survey",
            "is_target_source",
            "paired_valid",
            "flux_relative_difference_percent",
            "absolute_magnitude_difference",
        ],
        "Paired metrics",
    )

    groups = {
        "ALL": metrics,

        "TARGET": metrics[
            metrics["is_target_source"]
        ],

        "NEIGHBOUR": metrics[
            ~metrics["is_target_source"]
        ],

        "SDSS": metrics[
            metrics["survey"] == "SDSS"
        ],

        "TESS": metrics[
            metrics["survey"] == "TESS"
        ],

        "SDSS_TARGET": metrics[
            (metrics["survey"] == "SDSS")
            & metrics["is_target_source"]
        ],

        "SDSS_NEIGHBOUR": metrics[
            (metrics["survey"] == "SDSS")
            & ~metrics["is_target_source"]
        ],

        "TESS_TARGET": metrics[
            (metrics["survey"] == "TESS")
            & metrics["is_target_source"]
        ],

        "TESS_NEIGHBOUR": metrics[
            (metrics["survey"] == "TESS")
            & ~metrics["is_target_source"]
        ],
    }

    rows = []

    for group_name, group in groups.items():

        if group.empty:
            continue

        valid_flux = group[
            "flux_relative_difference_percent"
        ].notna()

        valid_magnitude = group[
            "absolute_magnitude_difference"
        ].notna()

        paired_valid = (
            group["paired_valid"]
            .fillna(False)
            .astype(bool)
        )

        rows.append(
            {
                "group": group_name,
                "source_count": len(group),
                "paired_valid_count": int(
                    paired_valid.sum()
                ),
                "paired_valid_fraction": (
                    paired_valid.mean()
                ),
                "finite_flux_change_count": int(
                    valid_flux.sum()
                ),
                "finite_magnitude_change_count": int(
                    valid_magnitude.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def _derive_source_background_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return source-level metrics with derived SCARLET-minus-Baseline
    background differences.

    ``calculate_paired_metrics`` does not currently materialise the three
    background difference columns for source-level rows, although it retains
    the paired Baseline and SCARLET background measurements.  Target-level
    analysis already derives these differences explicitly.  The all-source
    outputs use the same mechanism here so that the existing production
    metrics remain untouched.
    """

    table = metrics.copy()

    background_pairs = [
        (
            "background_level_baseline",
            "background_level_scarlet",
            "background_level_difference",
        ),
        (
            "background_flux_baseline",
            "background_flux_scarlet",
            "background_flux_difference",
        ),
        (
            "background_fraction_baseline",
            "background_fraction_scarlet",
            "background_fraction_difference",
        ),
    ]

    for baseline_column, scarlet_column, difference_column in background_pairs:
        if baseline_column in table.columns and scarlet_column in table.columns:
            baseline = _finite_series(table, baseline_column)
            scarlet = _finite_series(table, scarlet_column)
            table[difference_column] = scarlet - baseline

    return table


def build_all_source_photometric_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build the source-level Section 5.3.1 output for every LocalField source.

    This is deliberately separate from the target-level thesis table.  Every
    paired source is retained, including the target and neighbouring sources,
    so the richer TARGET_PHOTOMETRY source information is not discarded.
    No accuracy/improvement claim is made: the reported quantities describe
    the measured Baseline-to-SCARLET response.
    """

    metrics = _derive_source_background_differences(metrics)

    _require_columns(
        metrics,
        [
            "target_name",
            "survey",
            "source_identifier",
            "is_target_source",
            "source_role",
            "flux_relative_difference_percent",
            "absolute_flux_relative_difference_percent",
            "magnitude_difference",
            "absolute_magnitude_difference",
        ],
        "Paired metrics",
    )

    table = metrics.copy()

    if "paired_valid" in table.columns:
        table["Measurement quality"] = np.where(
            table["paired_valid"].fillna(False).astype(bool),
            "Valid paired measurement",
            "Requires review",
        )
    else:
        table["Measurement quality"] = "Requires review"

    table["Source population"] = np.where(
        table["is_target_source"].fillna(False).astype(bool),
        "Target",
        "Neighbour",
    )

    rename = {
        "target_name": "Target field",
        "survey": "Survey",
        "source_identifier": "Source identifier",
        "source_role": "Source role",
        "flux_relative_difference_percent": "Flux change (%)",
        "absolute_flux_relative_difference_percent": "Absolute flux change (%)",
        "magnitude_difference": "Magnitude change (mag)",
        "absolute_magnitude_difference": "Absolute magnitude change (mag)",
        "net_flux_baseline": "Baseline net flux",
        "net_flux_scarlet": "SCARLET net flux",
        "flux_difference": "Flux difference",
        "instrumental_magnitude_baseline": "Baseline instrumental magnitude",
        "instrumental_magnitude_scarlet": "SCARLET instrumental magnitude",
        "background_level_baseline": "Baseline background",
        "background_level_scarlet": "SCARLET background",
        "background_level_difference": "Background change",
        "background_flux_baseline": "Baseline background flux",
        "background_flux_scarlet": "SCARLET background flux",
        "background_flux_difference": "Background flux change",
        "background_fraction_baseline": "Baseline background fraction",
        "background_fraction_scarlet": "SCARLET background fraction",
        "background_fraction_difference": "Background fraction change",
    }
    table = table.rename(columns=rename)

    preferred = [
        "Target field",
        "Survey",
        "Source identifier",
        "Source population",
        "Source role",
        "Flux change (%)",
        "Absolute flux change (%)",
        "Magnitude change (mag)",
        "Absolute magnitude change (mag)",
        "Background change",
        "Baseline net flux",
        "SCARLET net flux",
        "Flux difference",
        "Baseline instrumental magnitude",
        "SCARLET instrumental magnitude",
        "Measurement quality",
    ]

    table = table[[c for c in preferred if c in table.columns]].copy()

    table["_survey_order"] = table["Survey"].map({"SDSS": 0, "TESS": 1}).fillna(99)
    table["_population_order"] = table["Source population"].map({"Target": 0, "Neighbour": 1}).fillna(99)

    table = (
        table.sort_values(
            by=["_survey_order", "Target field", "_population_order", "Source identifier"],
            kind="stable",
        )
        .drop(columns=["_survey_order", "_population_order"])
        .reset_index(drop=True)
    )

    decimal_places = {
        "Flux change (%)": 4,
        "Absolute flux change (%)": 4,
        "Magnitude change (mag)": 6,
        "Absolute magnitude change (mag)": 6,
        "Background change": 4,
    }
    for column, decimals in decimal_places.items():
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(decimals)

    return table


def build_all_source_photometric_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact thesis-ready summary of all source-level responses."""

    metrics = _derive_source_background_differences(metrics)

    _require_columns(
        metrics,
        [
            "survey",
            "is_target_source",
            "flux_relative_difference_percent",
            "absolute_flux_relative_difference_percent",
            "absolute_magnitude_difference",
            "background_level_difference",
        ],
        "Paired metrics",
    )

    rows = []
    group_specs = [
        ("SDSS", "SDSS", None),
        ("SDSS Target", "SDSS", True),
        ("SDSS Neighbour", "SDSS", False),
        ("TESS", "TESS", None),
        ("TESS Target", "TESS", True),
        ("TESS Neighbour", "TESS", False),
        ("All sources", None, None),
    ]

    for label, survey, target_flag in group_specs:
        group = metrics.copy()
        if survey is not None:
            group = group[group["survey"] == survey]
        if target_flag is not None:
            group = group[group["is_target_source"].fillna(False).astype(bool) == target_flag]
        if group.empty:
            continue

        flux = _finite_series(group, "flux_relative_difference_percent")
        abs_flux = _finite_series(group, "absolute_flux_relative_difference_percent")
        abs_mag = _finite_series(group, "absolute_magnitude_difference")
        background = _finite_series(group, "background_level_difference")

        paired = (
            group["paired_valid"].fillna(False).astype(bool)
            if "paired_valid" in group.columns
            else pd.Series(False, index=group.index)
        )

        rows.append({
            "Source population": label,
            "Source count": len(group),
            "Paired valid (n)": int(paired.sum()),
            "Median flux change (%)": flux.median(),
            "Median absolute flux change (%)": abs_flux.median(),
            "Median absolute magnitude change (mag)": abs_mag.median(),
            "Median background change": background.median(),
        })

    summary = pd.DataFrame(rows)
    for column, decimals in {
        "Median flux change (%)": 4,
        "Median absolute flux change (%)": 4,
        "Median absolute magnitude change (mag)": 6,
        "Median background change": 4,
    }.items():
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce").round(decimals)

    return summary


# ===========================================================================
# Section 5.3.1 thesis table
# ===========================================================================

def build_target_photometric_thesis_table(
    target_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the compact thesis-ready target-level comparison table.

    MAE is intentionally not reported because TARGET_PHOTOMETRY does not
    provide an independent ground-truth flux or magnitude.
    """

    _require_columns(
        target_summary,
        [
            "target_name",
            "survey",
            "net_flux_baseline",
            "net_flux_scarlet",
            "flux_relative_difference_percent",
            "absolute_flux_relative_difference_percent",
            "instrumental_magnitude_baseline",
            "instrumental_magnitude_scarlet",
            "magnitude_difference",
            "absolute_magnitude_difference",
        ],
        "Target summary",
    )

    table = target_summary.copy()

    valid_baseline = (
        table["valid_baseline"]
        .fillna(False)
        .astype(bool)
        if "valid_baseline" in table.columns
        else pd.Series(
            False,
            index=table.index,
        )
    )

    valid_scarlet = (
        table["valid_scarlet"]
        .fillna(False)
        .astype(bool)
        if "valid_scarlet" in table.columns
        else pd.Series(
            False,
            index=table.index,
        )
    )

    table["Measurement quality"] = np.where(
        valid_baseline & valid_scarlet,
        "Valid in both",
        "Requires review",
    )

    table = table.rename(
        columns={
            "target_name":
                "Target",

            "survey":
                "Survey",

            "net_flux_baseline":
                "Baseline net flux",

            "net_flux_scarlet":
                "SCARLET net flux",

            "flux_relative_difference_percent":
                "Flux change (%)",

            "absolute_flux_relative_difference_percent":
                "Absolute flux change (%)",

            "instrumental_magnitude_baseline":
                "Baseline instrumental magnitude",

            "instrumental_magnitude_scarlet":
                "SCARLET instrumental magnitude",

            "magnitude_difference":
                "Magnitude change",

            "absolute_magnitude_difference":
                "Absolute magnitude change",

            "background_level_baseline":
                "Baseline background",

            "background_level_scarlet":
                "SCARLET background",

            "background_level_difference":
                "Background change",
        }
    )

    preferred_columns = [
        "Target",
        "Survey",
        "Flux change (%)",
        "Absolute flux change (%)",
        "Absolute magnitude change",
        "Background change",
        "Measurement quality",
    ]

    table = table[
        [
            column
            for column in preferred_columns
            if column in table.columns
        ]
    ].copy()

    table["_survey_order"] = (
        table["Survey"]
        .map(
            {
                "SDSS": 0,
                "TESS": 1,
            }
        )
        .fillna(99)
    )

    table = (
        table
        .sort_values(
            by=[
                "_survey_order",
                "Target",
            ],
            kind="stable",
        )
        .drop(
            columns="_survey_order"
        )
        .reset_index(
            drop=True
        )
    )

    decimal_places = {
        "Flux change (%)": 4,
        "Absolute flux change (%)": 4,
        "Absolute magnitude change": 5,
        "Background change": 3,
    }

    for column, decimals in decimal_places.items():

        if column in table.columns:

            table[column] = (
                pd.to_numeric(
                    table[column],
                    errors="coerce",
                )
                .round(decimals)
            )

    return table



def build_target_photometric_survey_summary(
    target_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a concise SDSS/TESS summary of target-level photometric effects.

    This table is intended as a higher-level companion to the target-by-target
    Section 5.3.1 table. It summarises the distribution of paired changes
    within each observational environment without replacing the target-level
    results. Medians are used because the number of targets is small and the
    target-level changes are heterogeneous.

    No MAE is reported because TARGET_PHOTOMETRY contains no independent
    ground-truth photometric measurement.
    """

    _require_columns(
        target_summary,
        [
            "survey",
            "flux_relative_difference_percent",
            "absolute_flux_relative_difference_percent",
            "absolute_magnitude_difference",
            "background_level_difference",
        ],
        "Target summary",
    )

    rows = []

    for survey in ["SDSS", "TESS"]:
        group = target_summary[target_summary["survey"] == survey].copy()

        flux_change = _finite_series(
            group,
            "flux_relative_difference_percent",
        )
        absolute_flux_change = _finite_series(
            group,
            "absolute_flux_relative_difference_percent",
        )
        absolute_magnitude_change = _finite_series(
            group,
            "absolute_magnitude_difference",
        )
        background_change = _finite_series(
            group,
            "background_level_difference",
        )

        rows.append(
            {
                "Survey": survey,
                "Target count": len(group),
                "Finite flux changes (n)": int(flux_change.notna().sum()),
                "Flux increase (n)": int((flux_change > 0).sum()),
                "Flux decrease (n)": int((flux_change < 0).sum()),
                "Median flux change (%)": flux_change.median(),
                "Median absolute flux change (%)": absolute_flux_change.median(),
                "Median absolute magnitude change (mag)": absolute_magnitude_change.median(),
                "Median background change": background_change.median(),
            }
        )

    table = pd.DataFrame(rows)

    decimal_places = {
        "Median flux change (%)": 4,
        "Median absolute flux change (%)": 4,
        "Median absolute magnitude change (mag)": 5,
        "Median background change": 3,
    }

    for column, decimals in decimal_places.items():
        if column in table.columns:
            table[column] = pd.to_numeric(
                table[column],
                errors="coerce",
            ).round(decimals)

    return table


def save_target_photometric_survey_summary(
    table: pd.DataFrame,
) -> None:
    """Save the concise SDSS/TESS Section 5.3.1 summary table as CSV."""

    TARGET_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        TARGET_RESULTS_DIR
        / "table_5_3_1_survey_summary.csv"
    )

    table.to_csv(
        csv_path,
        index=False,
    )

def save_all_source_photometric_outputs(
    source_table: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> None:
    """Save the all-source Section 5.3.1 supporting outputs."""

    TARGET_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_table.to_csv(
        TARGET_RESULTS_DIR / "table_5_3_1_all_source_photometric_measurements.csv",
        index=False,
    )

    source_summary.to_csv(
        TARGET_RESULTS_DIR / "table_5_3_1_all_source_summary.csv",
        index=False,
    )


def save_target_photometric_thesis_table(
    table: pd.DataFrame,
) -> None:
    """Save the final Section 5.3.1 thesis table as CSV only."""

    TARGET_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        TARGET_RESULTS_DIR
        / "table_5_3_1_target_photometric_measurements.csv"
    )

    table.to_csv(
        csv_path,
        index=False,
    )


# ===========================================================================
# Section 5.5.1 cross-dataset evaluation
# ===========================================================================

def build_cross_dataset_summary(
    target_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the concise SDSS/TESS comparison for Section 5.5.1.

    The comparison is based on the same target-level Baseline/SCARLET
    measurements used in Section 5.3.1. Medians are used because the two
    survey groups are small (n=6 and n=7) and target-level effects are
    heterogeneous.

    ``baseline crowding fraction`` describes the target's local baseline
    source contribution, while the remaining quantities describe the observed
    paired SCARLET response. Precision is deliberately not represented by a
    single value here because this static experiment provides one measurement
    per target; time-series precision is assessed separately.
    """

    _require_columns(
        target_summary,
        [
            "survey",
            "crowding_fraction_baseline",
            "absolute_flux_relative_difference_percent",
            "absolute_magnitude_difference",
            "background_level_difference",
        ],
        "Target summary",
    )

    rows = []

    for survey in ["SDSS", "TESS"]:
        group = target_summary[target_summary["survey"] == survey].copy()

        crowding = _finite_series(
            group,
            "crowding_fraction_baseline",
        )
        abs_flux = _finite_series(
            group,
            "absolute_flux_relative_difference_percent",
        )
        abs_mag = _finite_series(
            group,
            "absolute_magnitude_difference",
        )
        background = _finite_series(
            group,
            "background_level_difference",
        )
        signed_flux = _finite_series(
            group,
            "flux_relative_difference_percent",
        )

        rows.append(
            {
                "Survey": survey,
                "Target count": int(len(group)),
                "Median baseline crowding fraction": crowding.median(),
                "Median absolute flux change (%)": abs_flux.median(),
                "Median absolute magnitude change (mag)": abs_mag.median(),
                "Median background change": background.median(),
                "Flux increase (n)": int((signed_flux > 0).sum()),
                "Flux decrease (n)": int((signed_flux < 0).sum()),
            }
        )

    table = pd.DataFrame(rows)

    decimal_places = {
        "Median baseline crowding fraction": 4,
        "Median absolute flux change (%)": 4,
        "Median absolute magnitude change (mag)": 5,
        "Median background change": 3,
    }

    for column, decimals in decimal_places.items():
        if column in table.columns:
            table[column] = pd.to_numeric(
                table[column],
                errors="coerce",
            ).round(decimals)

    return table


def save_cross_dataset_summary(
    table: pd.DataFrame,
) -> None:
    """Save the Section 5.5.1 cross-dataset comparison table."""

    CROSS_DATASET_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        CROSS_DATASET_RESULTS_DIR
        / "table_5_5_1_cross_dataset_summary.csv",
        index=False,
    )


def save_cross_dataset_figure(
    target_summary: pd.DataFrame,
) -> None:
    """Generate the thesis-ready Section 5.5.1 SDSS/TESS comparison figure.

    Four target-level panels are shown:

        (a) baseline crowding fraction;
        (b) absolute SCARLET-induced flux change;
        (c) absolute SCARLET-induced instrumental-magnitude change; and
        (d) signed SCARLET-induced background-level change.

    Individual targets are retained to show within-survey variation, with a
    median marker for each observational environment. This avoids reducing
    the comparison to only two summary points.
    """

    _require_columns(
        target_summary,
        [
            "survey",
            "crowding_fraction_baseline",
            "absolute_flux_relative_difference_percent",
            "absolute_magnitude_difference",
            "background_level_difference",
        ],
        "Target summary",
    )

    plot_data = target_summary.copy()
    for column in [
        "crowding_fraction_baseline",
        "absolute_flux_relative_difference_percent",
        "absolute_magnitude_difference",
        "background_level_difference",
    ]:
        plot_data[column] = _finite_series(plot_data, column)

    plot_data = plot_data[plot_data["survey"].isin(["SDSS", "TESS"])].copy()
    if plot_data.empty:
        raise ValueError("No SDSS/TESS target data available for Section 5.5.1 figure.")

    survey_order = ["SDSS", "TESS"]
    x_base = {"SDSS": 0, "TESS": 1}
    marker_map = {"SDSS": "o", "TESS": "s"}

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 7.8),
        constrained_layout=True,
    )
    axes = axes.ravel()

    panels = [
        (
            "crowding_fraction_baseline",
            "Baseline crowding fraction",
            "(a) Target-local crowding",
        ),
        (
            "absolute_flux_relative_difference_percent",
            "Absolute flux change (%)",
            "(b) SCARLET-induced flux change",
        ),
        (
            "absolute_magnitude_difference",
            "Absolute magnitude change (mag)",
            "(c) SCARLET-induced magnitude change",
        ),
        (
            "background_level_difference",
            "Background change",
            "(d) SCARLET-induced background change",
        ),
    ]

    for ax, (column, ylabel, title) in zip(axes, panels):
        for survey in survey_order:
            values = plot_data.loc[
                plot_data["survey"] == survey,
                column,
            ].dropna().to_numpy()

            if len(values) == 0:
                continue

            # Small deterministic offsets make all individual target values
            # visible without introducing a boxplot that could overstate the
            # meaning of such small samples.
            offsets = np.linspace(
                -0.13,
                0.13,
                len(values),
            ) if len(values) > 1 else np.array([0.0])

            x = x_base[survey] + offsets
            ax.scatter(
                x,
                values,
                marker=marker_map[survey],
                s=42,
                alpha=0.85,
                label=survey,
                zorder=3,
            )

            median_value = float(np.nanmedian(values))
            ax.plot(
                [x_base[survey] - 0.16, x_base[survey] + 0.16],
                [median_value, median_value],
                linewidth=2.2,
                zorder=4,
            )

        if column == "background_level_difference":
            ax.axhline(
                0.0,
                linewidth=0.9,
                linestyle="--",
                zorder=1,
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SDSS", "TESS"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.7,
            alpha=0.6,
        )
        ax.set_axisbelow(True)

    axes[0].legend(
        title="Observational environment",
        frameon=False,
        loc="best",
    )

    figure.suptitle(
        "Cross-dataset comparison of target photometric effects",
        fontsize=15,
        fontweight="bold",
    )

    output_path = (
        CROSS_DATASET_RESULTS_DIR
        / "figure_5_5_1_cross_dataset_comparison.png"
    )

    CROSS_DATASET_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# ===========================================================================
# Section 5.3.1 thesis figure
# ===========================================================================

def save_target_photometric_figure(
    target_summary: pd.DataFrame,
) -> None:
    """
    Generate the thesis-ready Section 5.3.1 target-level comparison figure.

    Three complementary effects of SCARLET preprocessing are shown:

        (a) signed relative target net-flux change;
        (b) signed background-level change; and
        (c) absolute instrumental-magnitude change.

    SDSS and TESS targets are distinguished by marker shape. Target labels
    appear only on the shared bottom axis at approximately 45 degrees.
    The panels use generous vertical spacing so that panel titles, axes and
    tick labels remain visually separate in the exported figure.
    """

    _require_columns(
        target_summary,
        [
            "target_name",
            "survey",
            "flux_relative_difference_percent",
            "background_level_difference",
            "absolute_magnitude_difference",
        ],
        "Target summary",
    )

    plot_data = target_summary.copy()

    for column in [
        "flux_relative_difference_percent",
        "background_level_difference",
        "absolute_magnitude_difference",
    ]:
        plot_data[column] = _finite_series(plot_data, column)

    # Deterministic thesis ordering: SDSS first, then TESS, preserving the
    # existing target-name ordering within each observational environment.
    plot_data["_survey_order"] = (
        plot_data["survey"]
        .map({"SDSS": 0, "TESS": 1})
        .fillna(99)
    )

    plot_data = (
        plot_data
        .sort_values(
            by=["_survey_order", "target_name"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "Cannot generate target photometric figure from an empty dataset."
        )

    labels = plot_data["target_name"].astype(str).tolist()
    x_positions = np.arange(len(plot_data))

    # A taller canvas and explicit vertical spacing are intentional here.
    # Earlier versions compressed the three panels too tightly, causing the
    # panel titles to visually collide with the neighbouring axes.
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11.8, 9.6),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.0, 1.0, 1.0],
            "hspace": 0.28,
        },
    )

    marker_map = {
        "SDSS": "o",
        "TESS": "s",
    }

    def plot_panel(
        ax: Any,
        column: str,
        ylabel: str,
        title: str,
        show_legend: bool = False,
    ) -> None:
        """Plot one target-level effect panel."""

        ax.axhline(
            0.0,
            linewidth=0.9,
            linestyle="--",
            zorder=1,
        )

        for survey, marker in marker_map.items():
            mask = plot_data["survey"].eq(survey)

            if not mask.any():
                continue

            ax.scatter(
                x_positions[mask.to_numpy()],
                plot_data.loc[mask, column].to_numpy(),
                marker=marker,
                s=42,
                label=survey,
                zorder=3,
            )

        ax.set_ylabel(
            ylabel,
            fontsize=9,
            labelpad=6,
        )

        # Titles sit clearly above each axes rather than directly against the
        # previous panel's lower boundary.
        ax.set_title(
            title,
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )

        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.6,
            alpha=0.65,
        )

        ax.tick_params(
            axis="both",
            labelsize=8.5,
            pad=3,
        )

        if show_legend:
            ax.legend(
                title="Observational environment",
                frameon=False,
                loc="upper right",
                fontsize=8.5,
                title_fontsize=8.5,
                borderaxespad=0.3,
                handletextpad=0.5,
            )

    plot_panel(
        axes[0],
        "flux_relative_difference_percent",
        "Relative flux change (%)",
        "(a) Target net-flux change: SCARLET relative to Baseline",
        show_legend=True,
    )

    plot_panel(
        axes[1],
        "background_level_difference",
        "Background change (instrument units)",
        "(b) Background-level change: SCARLET relative to Baseline",
    )

    plot_panel(
        axes[2],
        "absolute_magnitude_difference",
        "Absolute magnitude change (mag)",
        "(c) Absolute change in instrumental magnitude",
    )

    # Only the bottom shared axis carries target labels.  This avoids the
    # repeated labels and vertical crowding seen in earlier versions.
    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=8.5,
    )
    axes[-1].set_xlabel(
        "Target",
        fontsize=9,
        labelpad=10,
    )

    for ax in axes[:-1]:
        ax.tick_params(
            axis="x",
            labelbottom=False,
        )

    # Visually separate the SDSS and TESS groups in every panel.
    sdss_count = int(
        plot_data["survey"].eq("SDSS").sum()
    )

    if 0 < sdss_count < len(plot_data):
        separator = sdss_count - 0.5
        for ax in axes:
            ax.axvline(
                separator,
                linewidth=0.8,
                linestyle=":",
                alpha=0.8,
                zorder=1,
            )

    figure.suptitle(
        "Target-level effect of SCARLET preprocessing on aperture photometry",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    figure.text(
        0.5,
        0.018,
        "Positive values indicate an increase relative to Baseline; "
        "negative values indicate a decrease.",
        ha="center",
        fontsize=8.5,
    )

    # Explicit margins prevent the 45-degree labels and explanatory note from
    # being compressed into the bottom of the plotting area.
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.925,
        bottom=0.205,
        hspace=0.28,
    )

    png_path = (
        TARGET_RESULTS_DIR
        / "figure_5_3_1_target_photometric_comparison.png"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)


def save_all_source_photometric_figure(
    metrics: pd.DataFrame,
) -> None:
    """Generate a simple, thesis-ready visualisation of all paired sources.

    The figure complements the target-level analysis by showing every valid
    paired LocalField source measurement.  Neighbouring sources are plotted
    first as small, semi-transparent triangles; target sources are plotted last
    as larger circles so that the target response remains visually identifiable
    even when many sources have similar values.

    The x-axis is the deterministic source order within survey and target
    field.  The vertical separator distinguishes SDSS from TESS.  No points
    are clipped, summarised, or replaced by population statistics.
    """

    metrics = _derive_source_background_differences(metrics)

    _require_columns(
        metrics,
        [
            "target_name",
            "survey",
            "source_identifier",
            "is_target_source",
            "flux_relative_difference_percent",
            "background_level_difference",
            "absolute_magnitude_difference",
        ],
        "Paired metrics",
    )

    plot_data = metrics.copy()
    plot_data["is_target_source"] = (
        plot_data["is_target_source"].fillna(False).astype(bool)
    )

    metric_columns = [
        "flux_relative_difference_percent",
        "background_level_difference",
        "absolute_magnitude_difference",
    ]
    for column in metric_columns:
        plot_data[column] = pd.to_numeric(
            plot_data[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    # Deterministic ordering: survey -> target field -> source identifier.
    # Within each field the target is retained at its natural source position;
    # no value-based sorting is used because it could visually imply an order
    # that does not exist in the observations.
    plot_data["_survey_order"] = (
        plot_data["survey"].map({"SDSS": 0, "TESS": 1}).fillna(99)
    )
    plot_data = (
        plot_data.sort_values(
            by=[
                "_survey_order",
                "target_name",
                "source_identifier",
                "is_target_source",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "Cannot generate all-source photometric figure from an empty dataset."
        )

    x = np.arange(len(plot_data), dtype=float)
    sdss_count = int((plot_data["survey"] == "SDSS").sum())

    # Survey colours are deliberately shared across all three panels.  Marker
    # shape identifies source role, making the target measurements visible
    # independently of colour.
    survey_colours = {
        "SDSS": "#1f77b4",
        "TESS": "#ff7f0e",
    }

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11.8, 9.0),
        sharex=True,
        gridspec_kw={"hspace": 0.30},
    )

    metric_specs = [
        (
            "flux_relative_difference_percent",
            "Relative flux change (%)",
            "(a) Source net-flux change: SCARLET relative to Baseline",
        ),
        (
            "background_level_difference",
            "Background change (instrument units)",
            "(b) Source background-level change: SCARLET relative to Baseline",
        ),
        (
            "absolute_magnitude_difference",
            "Absolute magnitude change (mag)",
            "(c) Absolute change in source instrumental magnitude",
        ),
    ]

    for ax, (column, ylabel, title) in zip(axes, metric_specs):
        # Plot neighbours first so that the much smaller target population is
        # never hidden by the dense neighbour population.
        for survey in ("SDSS", "TESS"):
            survey_mask = plot_data["survey"].eq(survey)
            neighbour_mask = survey_mask & ~plot_data["is_target_source"]
            target_mask = survey_mask & plot_data["is_target_source"]

            neighbour = plot_data.loc[neighbour_mask]
            target = plot_data.loc[target_mask]

            if not neighbour.empty:
                ax.scatter(
                    neighbour.index.to_numpy(dtype=float),
                    neighbour[column].to_numpy(dtype=float),
                    marker="^",
                    s=18,
                    alpha=0.55,
                    color=survey_colours[survey],
                    edgecolors="none",
                    linewidths=0,
                    zorder=3,
                )

            if not target.empty:
                ax.scatter(
                    target.index.to_numpy(dtype=float),
                    target[column].to_numpy(dtype=float),
                    marker="o",
                    s=48,
                    alpha=1.0,
                    color=survey_colours[survey],
                    edgecolors="black",
                    linewidths=0.75,
                    zorder=5,
                )

        ax.axhline(
            0.0,
            color="black",
            linewidth=0.8,
            linestyle="--",
            alpha=0.65,
            zorder=1,
        )

        if 0 < sdss_count < len(plot_data):
            ax.axvline(
                sdss_count - 0.5,
                color="black",
                linewidth=0.8,
                linestyle=":",
                alpha=0.65,
                zorder=1,
            )

        ax.set_ylabel(ylabel, fontsize=9.5, labelpad=7)
        ax.set_title(
            title,
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=7,
        )
        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.6,
            alpha=0.60,
        )
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.margins(x=0.005)

        # Let matplotlib derive the limits from all finite observations.  This
        # preserves the actual scale and ensures no source response is clipped.
        finite = plot_data[column].dropna()
        if finite.empty:
            ax.set_ylim(-1.0, 1.0)
        else:
            ymin = float(finite.min())
            ymax = float(finite.max())
            span = ymax - ymin
            padding = 0.08 * span if span > 0 else max(abs(ymax) * 0.1, 1.0)
            ax.set_ylim(ymin - padding, ymax + padding)

    axes[-1].set_xlabel(
        "Source (ordered within survey and target field; all paired LocalField sources)",
        fontsize=9.5,
        labelpad=8,
    )
    axes[-1].set_xticks([])

    # Keep the legend in its own figure-level strip.  This prevents it from
    # colliding with panel titles or plotted observations.
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor=survey_colours["SDSS"],
            markeredgecolor="black", markeredgewidth=0.7,
            markersize=6.5, label="SDSS target",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="None",
            markerfacecolor=survey_colours["SDSS"],
            markeredgecolor="none",
            markersize=5.5, label="SDSS neighbour",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor=survey_colours["TESS"],
            markeredgecolor="black", markeredgewidth=0.7,
            markersize=6.5, label="TESS target",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="None",
            markerfacecolor=survey_colours["TESS"],
            markeredgecolor="none",
            markersize=5.5, label="TESS neighbour",
        ),
    ]

    figure.legend(
        handles=legend_handles,
        title="Source population",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        title_fontsize=8.5,
        handletextpad=0.35,
        columnspacing=1.2,
    )

    figure.suptitle(
        "Source-level changes in aperture-photometric measurements following SCARLET preprocessing",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    figure.text(
        0.5,
        0.018,
        "Circles = target sources; triangles = neighbouring sources. "
        "All finite paired LocalField source measurements are plotted.",
        ha="center",
        fontsize=8.2,
    )

    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.875,
        bottom=0.085,
        hspace=0.30,
    )

    TARGET_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        TARGET_RESULTS_DIR / "figure_5_3_1_all_source_photometric_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

# ===========================================================================
# Section 5.3.1 supporting summary
# ===========================================================================

def build_target_effect_summary(
    target_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact summary supporting the written discussion of Sections
    5.3.1 and 5.3.3.
    """

    flux_change = pd.to_numeric(
        target_summary[
            "flux_relative_difference_percent"
        ],
        errors="coerce",
    )

    abs_flux_change = pd.to_numeric(
        target_summary[
            "absolute_flux_relative_difference_percent"
        ],
        errors="coerce",
    )

    magnitude_change = pd.to_numeric(
        target_summary[
            "magnitude_difference"
        ],
        errors="coerce",
    )

    absolute_magnitude_change = pd.to_numeric(
        target_summary[
            "absolute_magnitude_difference"
        ],
        errors="coerce",
    )

    valid_flux = flux_change.notna()
    valid_magnitude = magnitude_change.notna()

    rows = [
        {
            "metric":
                "Target count",
            "value":
                len(target_summary),
        },
        {
            "metric":
                "Targets with finite flux change",
            "value":
                int(valid_flux.sum()),
        },
        {
            "metric":
                "Targets with SCARLET higher flux",
            "value":
                int((flux_change > 0).sum()),
        },
        {
            "metric":
                "Targets with SCARLET lower flux",
            "value":
                int((flux_change < 0).sum()),
        },
        {
            "metric":
                "Median absolute flux change (%)",
            "value":
                abs_flux_change.median(),
        },
        {
            "metric":
                "Mean absolute flux change (%)",
            "value":
                abs_flux_change.mean(),
        },
        {
            "metric":
                "Targets with finite magnitude change",
            "value":
                int(valid_magnitude.sum()),
        },
        {
            "metric":
                "Median absolute magnitude change (mag)",
            "value":
                absolute_magnitude_change.median(),
        },
        {
            "metric":
                "Mean absolute magnitude change (mag)",
            "value":
                absolute_magnitude_change.mean(),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ===========================================================================
# Target-local crowding analysis for Sections 5.4.1–5.4.3
# ===========================================================================

ADDITIONAL_RESULTS_DIR = (
    FIGURES_TABLES_DIR / "5_4_deblending_and_crowding"
)
LOCAL_CROWDING_RADIUS_PIXELS = 10.0


def _spearman_result(x: pd.Series, y: pd.Series) -> dict:
    """Return a descriptive Spearman association for two variables."""
    pair = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    if n < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return {"n": n, "spearman_rho": np.nan, "p_value": np.nan}
    if spearmanr is None:
        return {
            "n": n,
            "spearman_rho": pair["x"].corr(pair["y"], method="spearman"),
            "p_value": np.nan,
        }
    result: Any = spearmanr(pair["x"], pair["y"])
    rho, p_value = result
    return {"n": n, "spearman_rho": float(rho), "p_value": float(p_value)}


def build_target_local_crowding_dataset(
    measurements: pd.DataFrame,
    target_summary: pd.DataFrame,
    radius_pixels: float = LOCAL_CROWDING_RADIUS_PIXELS,
) -> pd.DataFrame:
    """
    Build a target-centred crowding dataset.

    For each Baseline target experiment, neighbours are restricted to 10 px
    from the designated target. The primary explanatory variable is
    sum(F_neighbours) / F_target using positive, valid Baseline net fluxes.
    """
    _require_columns(
        measurements,
        [
            "experiment_id", "target_name", "survey", "pipeline",
            "source_identifier", "is_target_source", "x_centroid",
            "y_centroid", "net_flux", "background_fraction", "valid",
        ],
        "Source measurements",
    )
    _require_columns(
        target_summary,
        [
            "target_name", "survey", "net_flux_baseline",
            "flux_relative_difference_percent",
            "absolute_flux_relative_difference_percent",
            "absolute_magnitude_difference", "background_level_difference",
        ],
        "Target summary",
    )

    baseline = measurements[measurements["pipeline"].eq("Baseline")].copy()
    rows = []

    for experiment_id, group in baseline.groupby("experiment_id", sort=False):
        target_rows = group[group["is_target_source"].astype(bool)]
        if len(target_rows) != 1:
            raise ValueError(
                f"Expected one target in Baseline experiment {experiment_id}; "
                f"found {len(target_rows)}."
            )
        target = target_rows.iloc[0]
        tx = pd.to_numeric(target["x_centroid"], errors="coerce")
        ty = pd.to_numeric(target["y_centroid"], errors="coerce")
        tf = pd.to_numeric(target["net_flux"], errors="coerce")
        if not (np.isfinite(tx) and np.isfinite(ty) and np.isfinite(tf) and tf > 0):
            raise ValueError(f"Invalid target coordinates or flux for {target['target_name']}")

        neighbours = group[~group["is_target_source"].astype(bool)].copy()
        neighbours["distance"] = np.hypot(
            pd.to_numeric(neighbours["x_centroid"], errors="coerce") - tx,
            pd.to_numeric(neighbours["y_centroid"], errors="coerce") - ty,
        )
        neighbours["flux"] = pd.to_numeric(neighbours["net_flux"], errors="coerce")
        local = neighbours[
            (neighbours["distance"] <= radius_pixels)
            & neighbours["valid"].astype(bool)
            & np.isfinite(neighbours["flux"])
            & (neighbours["flux"] > 0)
        ].sort_values("distance", kind="stable")

        name = str(target["target_name"])
        survey = str(target["survey"])
        effect = target_summary[
            target_summary["target_name"].astype(str).eq(name)
            & target_summary["survey"].astype(str).eq(survey)
        ]
        if len(effect) != 1:
            raise ValueError(f"Expected one target result for {name}; found {len(effect)}")
        effect = effect.iloc[0]

        if local.empty:
            nearest_distance = np.nan
            nearest_ratio = np.nan
            total_ratio = 0.0
            neighbour_ids = ""
        else:
            nearest = local.iloc[0]
            nearest_distance = float(nearest["distance"])
            nearest_ratio = float(nearest["flux"] / tf)
            total_ratio = float(local["flux"].sum() / tf)
            neighbour_ids = ";".join(local["source_identifier"].astype(str))

        rows.append({
            "target_name": name,
            "survey": survey,
            "experiment_id": str(experiment_id),
            "target_source_identifier": str(target["source_identifier"]),
            "local_radius_pixels": float(radius_pixels),
            "neighbour_count_within_10px": int(len(local)),
            "nearest_neighbour_distance_pixels": nearest_distance,
            "nearest_neighbour_flux_ratio": nearest_ratio,
            "local_neighbour_to_target_flux_ratio": total_ratio,
            "baseline_target_net_flux": float(tf),
            "baseline_target_background_fraction": float(
                pd.to_numeric(target["background_fraction"], errors="coerce")
            ),
            "local_neighbour_source_ids": neighbour_ids,
            "flux_change_percent": float(effect["flux_relative_difference_percent"]),
            "absolute_flux_change_percent": float(effect["absolute_flux_relative_difference_percent"]),
            "absolute_magnitude_change_mag": float(effect["absolute_magnitude_difference"]),
            "background_change": float(effect["background_level_difference"]),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No target-local crowding records were generated.")
    return result.sort_values(["survey", "target_name"], kind="stable").reset_index(drop=True)


def build_target_local_crowding_table(local: pd.DataFrame) -> pd.DataFrame:
    """Build the thesis table for Section 5.4.1."""
    columns = [
        "target_name", "survey", "neighbour_count_within_10px",
        "nearest_neighbour_distance_pixels", "nearest_neighbour_flux_ratio",
        "local_neighbour_to_target_flux_ratio", "baseline_target_net_flux",
        "baseline_target_background_fraction",
    ]
    _require_columns(local, columns, "Target-local crowding dataset")
    table = local[columns].rename(columns={
        "target_name": "Target",
        "survey": "Survey",
        "neighbour_count_within_10px": "Neighbours within 10 px (n)",
        "nearest_neighbour_distance_pixels": "Nearest-neighbour distance (px)",
        "nearest_neighbour_flux_ratio": "Nearest-neighbour flux / target flux",
        "local_neighbour_to_target_flux_ratio": "Local neighbour flux / target flux",
        "baseline_target_net_flux": "Baseline target net flux",
        "baseline_target_background_fraction": "Baseline background fraction",
    }).copy()
    table["_order"] = table["Survey"].map({"SDSS": 0, "TESS": 1}).fillna(99)
    table = table.sort_values(["_order", "Target"], kind="stable").drop(columns="_order").reset_index(drop=True)
    for c, d in {
        "Nearest-neighbour distance (px)": 3,
        "Nearest-neighbour flux / target flux": 4,
        "Local neighbour flux / target flux": 4,
        "Baseline target net flux": 3,
        "Baseline background fraction": 4,
    }.items():
        table[c] = pd.to_numeric(table[c], errors="coerce").round(d)
    return table


def build_survey_local_crowding_table(local: pd.DataFrame) -> pd.DataFrame:
    """Build a concise survey-level summary of target-local crowding.

    This table is intended for the main thesis text, while the corresponding
    target-level table is retained for the Appendix. Medians are used because
    local neighbour-to-target flux ratios can be strongly skewed by individual
    bright neighbours.
    """

    required = [
        "survey",
        "neighbour_count_within_10px",
        "nearest_neighbour_distance_pixels",
        "nearest_neighbour_flux_ratio",
        "local_neighbour_to_target_flux_ratio",
        "baseline_target_net_flux",
        "baseline_target_background_fraction",
    ]
    _require_columns(local, required, "Target-local crowding dataset")

    rows = []
    for survey in ["SDSS", "TESS"]:
        group = local[local["survey"].eq(survey)]
        if group.empty:
            continue

        rows.append({
            "Survey": survey,
            "Target count": int(len(group)),
            "Median neighbours within 10 px (n)": group[
                "neighbour_count_within_10px"
            ].median(),
            "Median nearest-neighbour distance (px)": _finite_series(
                group, "nearest_neighbour_distance_pixels"
            ).median(),
            "Median nearest-neighbour flux / target flux": _finite_series(
                group, "nearest_neighbour_flux_ratio"
            ).median(),
            "Median local neighbour flux / target flux": _finite_series(
                group, "local_neighbour_to_target_flux_ratio"
            ).median(),
            "Median baseline target net flux": _finite_series(
                group, "baseline_target_net_flux"
            ).median(),
            "Median baseline background fraction": _finite_series(
                group, "baseline_target_background_fraction"
            ).median(),
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    for c, d in {
        "Median neighbours within 10 px (n)": 2,
        "Median nearest-neighbour distance (px)": 3,
        "Median nearest-neighbour flux / target flux": 4,
        "Median local neighbour flux / target flux": 4,
        "Median baseline target net flux": 3,
        "Median baseline background fraction": 4,
    }.items():
        table[c] = pd.to_numeric(table[c], errors="coerce").round(d)

    return table


def build_target_local_crowding_associations(local: pd.DataFrame) -> pd.DataFrame:
    """Test local neighbour contribution against the primary photometric effect.

    The primary statistical outcome is absolute target flux change. Absolute
    magnitude change is also plotted, but it is not reported as a second
    Spearman test because it is a monotonic transformation of the same
    absolute fractional flux change and therefore produces the same rank
    ordering, rho, and p-value. Reporting both would duplicate the same
    statistical test.
    """
    _require_columns(
        local,
        ["survey", "local_neighbour_to_target_flux_ratio",
         "absolute_flux_change_percent", "absolute_magnitude_change_mag"],
        "Target-local crowding dataset",
    )

    groups = {
        "All targets": local,
        "SDSS targets": local[local["survey"].eq("SDSS")],
        "TESS targets": local[local["survey"].eq("TESS")],
    }

    rows = []
    for group_name, group in groups.items():
        result = _spearman_result(
            group["local_neighbour_to_target_flux_ratio"],
            group["absolute_flux_change_percent"],
        )
        rows.append({
            "Group": group_name,
            "Predictor": "Local neighbour flux / target flux",
            "Outcome": "Absolute flux change (%)",
            "n": result["n"],
            "Spearman rho": result["spearman_rho"],
            "p-value": result["p_value"],
        })

    table = pd.DataFrame(rows)
    table["Spearman rho"] = table["Spearman rho"].round(4)
    table["p-value"] = table["p-value"].round(4)
    return table


def save_target_local_crowding_profile_figure(local: pd.DataFrame) -> None:
    """Generate the Section 5.4.1 target-local crowding profile."""
    data = local.copy()
    for c in ["local_neighbour_to_target_flux_ratio", "nearest_neighbour_distance_pixels"]:
        data[c] = _finite_series(data, c)
    data = data.dropna(subset=["local_neighbour_to_target_flux_ratio", "nearest_neighbour_distance_pixels"])
    data = data.sort_values("local_neighbour_to_target_flux_ratio", kind="stable").reset_index(drop=True)
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    x = np.arange(len(data))
    for survey, marker in {"SDSS": "o", "TESS": "s"}.items():
        mask = data["survey"].eq(survey)
        if mask.any():
            axes[0].scatter(x[mask], data.loc[mask, "local_neighbour_to_target_flux_ratio"], marker=marker, s=52, label=survey, zorder=3)
            axes[1].scatter(x[mask], data.loc[mask, "nearest_neighbour_distance_pixels"], marker=marker, s=52, label=survey, zorder=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Local neighbour flux / target flux")
    axes[1].set_ylabel("Nearest-neighbour distance (px)")
    labels = data["target_name"].astype(str).tolist()
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_xlabel("Target")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    axes[0].set_title("(a) Local neighbour-to-target flux contribution", loc="left", fontweight="bold", pad=8)
    axes[1].set_title("(b) Nearest-neighbour separation", loc="left", fontweight="bold", pad=8)
    axes[0].legend(title="Observational environment", frameon=False, fontsize=8.5, title_fontsize=8.5)
    fig.suptitle("Target-local crowding characteristics", fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.30, wspace=0.28)
    ADDITIONAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ADDITIONAL_RESULTS_DIR / "figure_5_4_1_target_local_crowding_profile.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_target_local_crowding_effect_figure(local: pd.DataFrame) -> None:
    """Generate the key Section 5.4.2 local-crowding/effect figure."""
    data = local.copy()
    for c in ["local_neighbour_to_target_flux_ratio", "absolute_flux_change_percent", "absolute_magnitude_change_mag"]:
        data[c] = _finite_series(data, c)
    data = data[(data["local_neighbour_to_target_flux_ratio"] > 0)].dropna(subset=["local_neighbour_to_target_flux_ratio", "absolute_flux_change_percent", "absolute_magnitude_change_mag"])
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    for survey, marker in {"SDSS": "o", "TESS": "s"}.items():
        mask = data["survey"].eq(survey)
        if mask.any():
            axes[0].scatter(data.loc[mask, "local_neighbour_to_target_flux_ratio"], data.loc[mask, "absolute_flux_change_percent"], marker=marker, s=52, label=survey, zorder=3)
            axes[1].scatter(data.loc[mask, "local_neighbour_to_target_flux_ratio"], data.loc[mask, "absolute_magnitude_change_mag"], marker=marker, s=52, label=survey, zorder=3)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Local neighbour flux / target flux")
        ax.grid(axis="both", linestyle=":", linewidth=0.6, alpha=0.65)
    axes[0].set_ylabel("Absolute target flux change (%)")
    axes[1].set_ylabel("Absolute target magnitude change (mag)")
    axes[0].set_title("(a) Local neighbour contribution vs flux change", loc="left", fontweight="bold", pad=8)
    axes[1].set_title("(b) Local neighbour contribution vs magnitude change", loc="left", fontweight="bold", pad=8)
    axes[0].legend(title="Observational environment", frameon=False, fontsize=8.5, title_fontsize=8.5)
    fig.suptitle("Target-local neighbour contribution and SCARLET photometric effect", fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.18, wspace=0.28)
    ADDITIONAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ADDITIONAL_RESULTS_DIR / "figure_5_4_2_local_neighbour_ratio_vs_photometric_effect.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_representative_deblending_cases(local: pd.DataFrame) -> pd.DataFrame:
    """Build the three pre-selected representative cases for Section 5.4.3."""
    cases = [
        ("Case 1", "SDSS M67 Source 118", "Strong local neighbour contribution + large SCARLET change"),
        ("Case 2", "SDSS NGC 6791 Source 2029", "Strong local neighbour contribution + moderate SCARLET change"),
        ("Case 3", "GJ 9827", "Weak local neighbour contribution + negligible SCARLET change"),
    ]
    rows = []
    for case, target, rationale in cases:
        match = local[local["target_name"].astype(str).eq(target)]
        if len(match) != 1:
            raise ValueError(f"Representative case '{target}' not found exactly once.")
        r = match.iloc[0]
        rows.append({
            "Case": case,
            "Target": target,
            "Survey": r["survey"],
            "Selection rationale": rationale,
            "Neighbours within 10 px (n)": int(r["neighbour_count_within_10px"]),
            "Nearest-neighbour distance (px)": r["nearest_neighbour_distance_pixels"],
            "Nearest-neighbour flux / target flux": r["nearest_neighbour_flux_ratio"],
            "Local neighbour flux / target flux": r["local_neighbour_to_target_flux_ratio"],
            "Absolute flux change (%)": r["absolute_flux_change_percent"],
            "Absolute magnitude change (mag)": r["absolute_magnitude_change_mag"],
            "Neighbour source IDs": r["local_neighbour_source_ids"],
        })
    table = pd.DataFrame(rows)
    for c, d in {
        "Nearest-neighbour distance (px)": 3,
        "Nearest-neighbour flux / target flux": 4,
        "Local neighbour flux / target flux": 4,
        "Absolute flux change (%)": 4,
        "Absolute magnitude change (mag)": 5,
    }.items():
        table[c] = pd.to_numeric(table[c], errors="coerce").round(d)
    return table


def save_target_local_crowding_outputs(
    local_table: pd.DataFrame,
    survey_table: pd.DataFrame,
    associations: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    """Write the thesis-ready Section 5.4 tables.

    The target-level table is retained for the Appendix, while the concise
    survey-level summary is intended for the main thesis text.
    """
    _reset_output_directory(ADDITIONAL_RESULTS_DIR)
    local_table.to_csv(
        ADDITIONAL_RESULTS_DIR / "table_5_4_1_target_local_crowding.csv",
        index=False,
    )
    survey_table.to_csv(
        ADDITIONAL_RESULTS_DIR / "table_5_4_1_survey_local_crowding_summary.csv",
        index=False,
    )
    associations.to_csv(
        ADDITIONAL_RESULTS_DIR / "table_5_4_2_local_crowding_associations.csv",
        index=False,
    )
    cases.to_csv(
        ADDITIONAL_RESULTS_DIR / "table_5_4_3_representative_deblending_cases.csv",
        index=False,
    )


# ===========================================================================
# Coverage summary
# ===========================================================================

def build_analysis_coverage(
    summary: pd.DataFrame,
    canonical: pd.DataFrame,
    paired: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact record of the analytical population."""

    return pd.DataFrame(
        [
            {
                "dataset":
                    "experiment_summary",
                "rows":
                    len(summary),
                "columns":
                    len(summary.columns),
            },
            {
                "dataset":
                    "canonical_source_level",
                "rows":
                    len(canonical),
                "columns":
                    len(canonical.columns),
            },
            {
                "dataset":
                    "paired_source_level",
                "rows":
                    len(paired),
                "columns":
                    len(paired.columns),
            },
            {
                "dataset":
                    "paired_metrics",
                "rows":
                    len(metrics),
                "columns":
                    len(metrics.columns),
            },
        ]
    )


# ===========================================================================
# Output writing
# ===========================================================================

def save_analysis_outputs(
    coverage: pd.DataFrame,
    target_summary: pd.DataFrame,
    source_population: pd.DataFrame,
    all_source_table: pd.DataFrame,
    all_source_summary: pd.DataFrame,
    descriptive_statistics: pd.DataFrame,
    field_summary: pd.DataFrame,
    paired_metrics: pd.DataFrame,
    target_effect_summary: pd.DataFrame,
) -> None:
    """Save reproducible analytical outputs as CSV files."""

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage.to_csv(
        ANALYSIS_DIR
        / "analysis_coverage.csv",
        index=False,
    )

    target_summary.to_csv(
        ANALYSIS_DIR
        / "target_photometric_results.csv",
        index=False,
    )

    source_population.to_csv(
        ANALYSIS_DIR
        / "source_population_summary.csv",
        index=False,
    )

    all_source_table.to_csv(
        ANALYSIS_DIR
        / "all_source_photometric_measurements.csv",
        index=False,
    )

    all_source_summary.to_csv(
        ANALYSIS_DIR
        / "all_source_photometric_summary.csv",
        index=False,
    )

    descriptive_statistics.to_csv(
        ANALYSIS_DIR
        / "descriptive_statistics.csv",
        index=False,
    )

    field_summary.to_csv(
        ANALYSIS_DIR
        / "field_summary.csv",
        index=False,
    )

    paired_metrics.to_csv(
        ANALYSIS_DIR
        / "paired_metrics.csv",
        index=False,
    )

    target_effect_summary.to_csv(
        ANALYSIS_DIR
        / "target_effect_summary.csv",
        index=False,
    )


# ===========================================================================
# Main analysis
# ===========================================================================

def main() -> None:

    print(
        "Running TARGET_PHOTOMETRY scientific analysis..."
    )
    print()

    # ------------------------------------------------------------------
    # Clear generated outputs from previous runs.
    #
    # This guarantees that renamed or removed thesis artefacts cannot
    # remain as stale files.
    # ------------------------------------------------------------------

    _reset_output_directory(
        ANALYSIS_DIR
    )

    _reset_output_directory(
        TARGET_RESULTS_DIR
    )

    _reset_output_directory(
        CROSS_DATASET_RESULTS_DIR
    )

    # ------------------------------------------------------------------
    # 1. Load frozen production outputs
    # ------------------------------------------------------------------

    summary = load_experiment_summary()

    measurements = load_source_measurements()

    print(
        f"[PASS] Loaded experiment summary: "
        f"{len(summary)} rows"
    )

    print(
        f"[PASS] Loaded source measurements: "
        f"{len(measurements)} rows"
    )

    # ------------------------------------------------------------------
    # 2. Validate production inputs
    # ------------------------------------------------------------------

    validate_input_data(
        summary,
        measurements,
    )

    print(
        "[PASS] Frozen production inputs validated"
    )

    # ------------------------------------------------------------------
    # 3. Build complete local-field dataset
    # ------------------------------------------------------------------

    local_field = (
        extract_local_field_measurements(
            measurements
        )
    )

    validate_local_field_dataset(
        local_field
    )

    local_field = (
        attach_experiment_metadata(
            local_field,
            summary,
        )
    )

    validate_enriched_local_field_dataset(
        local_field
    )

    print(
        f"[PASS] Enriched local-field dataset: "
        f"{len(local_field)} rows"
    )

    # ------------------------------------------------------------------
    # 4. Canonical source-level dataset
    # ------------------------------------------------------------------

    canonical = (
        build_canonical_dataset(
            local_field
        )
    )

    validate_canonical_dataset(
        canonical
    )

    print(
        f"[PASS] Canonical dataset: "
        f"{len(canonical)} rows × "
        f"{len(canonical.columns)} columns"
    )

    # ------------------------------------------------------------------
    # 5. Paired Baseline / SCARLET dataset
    # ------------------------------------------------------------------

    paired = (
        build_paired_dataset(
            canonical
        )
    )

    validate_paired_dataset(
        paired
    )

    print(
        f"[PASS] Paired dataset: "
        f"{len(paired)} rows × "
        f"{len(paired.columns)} columns"
    )

    # ------------------------------------------------------------------
    # 6. Paired photometric metrics
    # ------------------------------------------------------------------

    metrics = (
        calculate_paired_metrics(
            paired
        )
    )

    validate_paired_metrics(
        metrics
    )

    print(
        f"[PASS] Paired metrics: "
        f"{len(metrics)} rows × "
        f"{len(metrics.columns)} columns"
    )

    # ------------------------------------------------------------------
    # 7. Existing descriptive statistics
    # ------------------------------------------------------------------

    descriptive_statistics = (
        build_descriptive_statistics(
            metrics
        )
    )

    validate_descriptive_statistics(
        descriptive_statistics
    )

    print(
        f"[PASS] Descriptive statistics: "
        f"{len(descriptive_statistics)} rows × "
        f"{len(descriptive_statistics.columns)} columns"
    )

    # ------------------------------------------------------------------
    # 8. Target-level analysis
    # ------------------------------------------------------------------

    target = (
        build_target_analysis_dataset(
            metrics
        )
    )

    target_summary = (
        build_target_summary(
            target
        )
    )

    if target_summary.empty:
        raise ValueError(
            "Target-level analytical dataset is empty."
        )

    print(
        f"[PASS] Target-level analysis: "
        f"{len(target_summary)} targets"
    )

    # ------------------------------------------------------------------
    # 9. Source-population analysis
    # ------------------------------------------------------------------

    source_population = (
        build_source_population_summary(
            metrics
        )
    )

    print(
        "[PASS] Source-population analysis completed"
    )

    # ------------------------------------------------------------------
    # 9b. Section 5.3.1 all-source photometric analysis
    # ------------------------------------------------------------------

    all_source_table = build_all_source_photometric_table(
        metrics
    )

    all_source_summary = build_all_source_photometric_summary(
        metrics
    )

    save_all_source_photometric_outputs(
        all_source_table,
        all_source_summary,
    )

    print(
        f"[PASS] All-source photometric output: "
        f"{len(all_source_table)} source measurements"
    )

    # ------------------------------------------------------------------
    # 10. Field-level analysis
    # ------------------------------------------------------------------

    field_summary = (
        build_field_summary(
            target
        )
    )

    print(
        f"[PASS] Field-level analysis: "
        f"{len(field_summary)} fields"
    )

    # ------------------------------------------------------------------
    # 12. Section 5.3.1 thesis table
    # ------------------------------------------------------------------

    thesis_table = (
        build_target_photometric_thesis_table(
            target_summary
        )
    )

    save_target_photometric_thesis_table(
        thesis_table
    )

    print(
        "[PASS] Section 5.3.1 thesis table generated"
    )

    # ------------------------------------------------------------------
    # 13. Section 5.3.1 SDSS/TESS summary table
    # ------------------------------------------------------------------

    survey_summary = build_target_photometric_survey_summary(
        target_summary
    )

    save_target_photometric_survey_summary(
        survey_summary
    )

    print(
        "[PASS] Section 5.3.1 SDSS/TESS summary table generated"
    )

    print()
    print("Section 5.3.1 SDSS/TESS summary:")
    print()
    print(survey_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 14. Section 5.3.1 thesis figure
    # ------------------------------------------------------------------

    save_target_photometric_figure(
        target_summary
    )

    print(
        "[PASS] Section 5.3.1 thesis figure generated"
    )

    save_all_source_photometric_figure(
        metrics
    )

    print(
        "[PASS] Section 5.3.1 all-source figure generated"
    )

    # ------------------------------------------------------------------
    # 14b. Section 5.5.1 cross-dataset comparison
    # ------------------------------------------------------------------

    cross_dataset_summary = build_cross_dataset_summary(
        target_summary
    )

    save_cross_dataset_summary(
        cross_dataset_summary
    )

    save_cross_dataset_figure(
        target_summary
    )

    print(
        "[PASS] Section 5.5.1 cross-dataset table and figure generated"
    )

    print()
    print("Section 5.5.1 cross-dataset summary:")
    print()
    print(cross_dataset_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 15. Section 5.3.1 / 5.3.3 supporting summary
    # ------------------------------------------------------------------

    target_effect_summary = (
        build_target_effect_summary(
            target_summary
        )
    )

    print(
        "[PASS] Target-effect summary generated"
    )

    # ------------------------------------------------------------------
    # 16. Target-local crowding analysis for Sections 5.4.1–5.4.3
    # ------------------------------------------------------------------

    target_local_crowding = build_target_local_crowding_dataset(
        measurements,
        target_summary,
    )

    local_crowding_table = build_target_local_crowding_table(
        target_local_crowding
    )
    survey_local_crowding_table = build_survey_local_crowding_table(
        target_local_crowding
    )
    local_crowding_associations = build_target_local_crowding_associations(
        target_local_crowding
    )
    representative_cases = build_representative_deblending_cases(
        target_local_crowding
    )

    save_target_local_crowding_outputs(
        local_table=local_crowding_table,
        survey_table=survey_local_crowding_table,
        associations=local_crowding_associations,
        cases=representative_cases,
    )
    save_target_local_crowding_profile_figure(
        target_local_crowding
    )
    save_target_local_crowding_effect_figure(
        target_local_crowding
    )

    print(
        "[PASS] Target-local crowding analysis generated for Sections 5.4.1–5.4.3"
    )

    # ------------------------------------------------------------------
    # 17. Coverage
    # ------------------------------------------------------------------

    coverage = (
        build_analysis_coverage(
            summary,
            canonical,
            paired,
            metrics,
        )
    )

    # ------------------------------------------------------------------
    # 18. Save reproducible analytical outputs
    # ------------------------------------------------------------------

    save_analysis_outputs(
        coverage=coverage,
        target_summary=target_summary,
        source_population=source_population,
        all_source_table=all_source_table,
        all_source_summary=all_source_summary,
        descriptive_statistics=descriptive_statistics,
        field_summary=field_summary,
        paired_metrics=metrics,
        target_effect_summary=target_effect_summary,
    )

    print()
    print(
        f"[PASS] Analysis outputs saved to: "
        f"{ANALYSIS_DIR}"
    )

    print(
        f"[PASS] Thesis outputs saved to: "
        f"{TARGET_RESULTS_DIR}"
    )

    # ------------------------------------------------------------------
    # 19. Primary target-level result preview
    # ------------------------------------------------------------------

    print()
    print(
        "Target-level photometric results:"
    )
    print()

    preview_columns = [
        column
        for column in [
            "target_name",
            "survey",
            "net_flux_baseline",
            "net_flux_scarlet",
            "flux_relative_difference_percent",
            "absolute_flux_relative_difference_percent",
            "magnitude_difference",
            "absolute_magnitude_difference",
            "background_level_difference",
            "crowding_fraction_baseline",
        ]
        if column in target_summary.columns
    ]

    print(
        target_summary[
            preview_columns
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # 20. Source-population summary
    # ------------------------------------------------------------------

    print()
    print(
        "Source-population coverage:"
    )
    print()

    print(
        source_population.to_string(
            index=False
        )
    )

    print()
    print("All-source photometric summary:")
    print()
    print(
        all_source_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "TARGET_PHOTOMETRY SCIENTIFIC ANALYSIS COMPLETED"
    )


if __name__ == "__main__":
    main()