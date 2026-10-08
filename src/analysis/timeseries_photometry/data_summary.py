"""Production data summary and audit for TESS time-series photometry.

The production input and output locations are intentionally fixed. This module
does not accept CLI input/output parameters so the thesis analysis always runs
against the canonical production dataset.

The summary is designed as a pre-analysis audit. It reports:
- experiment-level completeness and source counts;
- target-only completeness for the seven paired TESS experiments;
- target/source identity across the two production source namespaces;
- Baseline/SCARLET temporal pairing;
- target photometric-field completeness and finite-value checks;
- negative background-flux records for investigation;
- global numeric ranges and missingness;
- explicit sanity checks required before scientific metrics are calculated.

Run:
    python -m analysis.timeseries_photometry.data_summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
import math

import pandas as pd

from .canonical import canonicalize
from .io import ExperimentSet, load_experiment_set


# ---------------------------------------------------------------------------
# Canonical production locations
# ---------------------------------------------------------------------------

PRODUCTION_INPUT_DIR = Path("results/exports/timeseries_photometry")
SUMMARY_OUTPUT_DIR = Path("results/analysis/timeseries_photometry")

EXPECTED_PIPELINES = {"Baseline", "SCARLET"}
EXPECTED_TARGETS = {
    "GJ 9827",
    "HD 191939",
    "HD 209458",
    "TIC 410153583",
    "TOI-2155",
    "WASP-12",
    "WASP-33",
}
EXPECTED_CADENCES = 300

TARGET_NUMERIC_COLUMNS = [
    "net_flux",
    "instrumental_magnitude",
    "flux_uncertainty",
    "magnitude_uncertainty",
    "background_level",
    "background_flux",
    "contamination_fraction",
    "observation_time",
    "cadence_index",
]

GLOBAL_NUMERIC_COLUMNS = [
    "net_flux",
    "instrumental_magnitude",
    "flux_uncertainty",
    "magnitude_uncertainty",
    "background_level",
    "background_flux",
    "contamination_fraction",
    "observation_time",
    "cadence_index",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_bool(value: object) -> bool:
    """Normalise production boolean fields without ambiguous pandas results."""
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    if isinstance(value, (int, float)):
        return value != 0

    return False


def _safe_float(value: object) -> float | None:
    """Convert a scalar value to float, returning None when unavailable."""
    try:
        if value is None:
            return None
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None

    return result if pd.notna(result) else None


def _numeric_summary(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Return descriptive ranges and missing counts for numeric fields."""
    rows: list[dict[str, Any]] = []

    for column in columns:
        if column not in frame.columns:
            continue

        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values.map(
            lambda value: (
                math.isfinite(float(value))
                if pd.notna(value)
                else False
            )
        )

        rows.append(
            {
                "column": column,
                "rows": int(len(frame)),
                "non_null": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "finite": int(finite.sum()),
                "non_finite": int(len(values) - finite.sum()),
                "min": (
                    float(values[finite].min())
                    if finite.any()
                    else None
                ),
                "median": (
                    float(values[finite].median())
                    if finite.any()
                    else None
                ),
                "max": (
                    float(values[finite].max())
                    if finite.any()
                    else None
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "rows",
            "non_null",
            "missing",
            "finite",
            "non_finite",
            "min",
            "median",
            "max",
        ],
    )


def _unique_or_none(values: pd.Series) -> object:
    """Return a scalar when a group contains one unique value, else a marker."""
    unique = values.dropna().astype(str).unique()
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 0:
        return None
    return "|".join(sorted(unique))


# ---------------------------------------------------------------------------
# Experiment-level inventory
# ---------------------------------------------------------------------------


def _experiment_inventory(data: ExperimentSet) -> pd.DataFrame:
    """Summarise each production experiment."""
    summary = data.experiment_summary.copy()
    sources = data.source_measurements.copy()
    ts = data.timeseries_measurements.copy()

    experiment_columns = [
        "experiment_id",
        "target_name",
        "survey",
        "experiment_type",
        "pipeline",
    ]

    experiments = (
        summary[
            [column for column in experiment_columns if column in summary.columns]
        ]
        .drop_duplicates()
    )

    source_counts = (
        sources.groupby("experiment_id", dropna=False)
        .agg(
            source_rows=("source_identifier", "size"),
            unique_sources=("source_identifier", "nunique"),
            target_source_count=(
                "is_target_source",
                lambda x: int(x.map(_as_bool).sum()),
            ),
            valid_source_rows=(
                "valid",
                lambda x: int(x.map(_as_bool).sum()),
            ),
        )
        .reset_index()
    )

    ts_counts = (
        ts.groupby("experiment_id", dropna=False)
        .agg(
            timeseries_rows=("source_identifier", "size"),
            unique_timeseries_sources=("source_identifier", "nunique"),
            target_timeseries_rows=(
                "is_target_source",
                lambda x: int(x.map(_as_bool).sum()),
            ),
            valid_timeseries_rows=(
                "valid",
                lambda x: int(x.map(_as_bool).sum()),
            ),
            invalid_timeseries_rows=(
                "valid",
                lambda x: int((~x.map(_as_bool)).sum()),
            ),
            cadence_count=("cadence_index", "nunique"),
        )
        .reset_index()
    )

    time_range = (
        ts.assign(
            _observation_time=pd.to_numeric(
                ts["observation_time"],
                errors="coerce",
            )
        )
        .groupby("experiment_id", dropna=False)["_observation_time"]
        .agg(
            observation_start="min",
            observation_end="max",
        )
        .reset_index()
    )

    result = (
        experiments
        .merge(source_counts, on="experiment_id", how="left")
        .merge(ts_counts, on="experiment_id", how="left")
        .merge(time_range, on="experiment_id", how="left")
        .sort_values(["target_name", "pipeline", "experiment_id"])
        .reset_index(drop=True)
    )

    result["valid_timeseries_fraction"] = (
        result["valid_timeseries_rows"] / result["timeseries_rows"]
    )

    result["invalid_timeseries_fraction"] = (
        result["invalid_timeseries_rows"] / result["timeseries_rows"]
    )

    return result


# ---------------------------------------------------------------------------
# Target-level inventory
# ---------------------------------------------------------------------------


def _target_inventory(data: ExperimentSet) -> pd.DataFrame:
    """Summarise target time-series data by target and pipeline."""
    ts = data.timeseries_measurements.copy()
    rows: list[dict[str, Any]] = []

    for (target, pipeline), group in ts.groupby(
        ["target_name", "pipeline"],
        dropna=False,
        sort=True,
    ):
        target_mask = group["is_target_source"].map(_as_bool)
        target_group = group.loc[target_mask].copy()
        target_valid = target_group["valid"].map(_as_bool)

        times = pd.to_numeric(
            target_group["observation_time"],
            errors="coerce",
        )

        rows.append(
            {
                "target_name": target,
                "pipeline": pipeline,
                "experiment_count": int(group["experiment_id"].nunique()),
                "experiment_ids": "|".join(
                    sorted(group["experiment_id"].dropna().astype(str).unique())
                ),
                "total_rows": int(len(group)),
                "unique_sources": int(group["source_identifier"].nunique()),
                "target_source_count": int(target_group["source_identifier"].nunique()),
                "target_rows": int(len(target_group)),
                "target_valid_rows": int(target_valid.sum()),
                "target_invalid_rows": int((~target_valid).sum()),
                "target_valid_fraction": (
                    float(target_valid.mean()) if len(target_valid) else None
                ),
                "target_cadence_count": int(
                    target_group["cadence_index"].nunique()
                ),
                "cadence_min": _safe_float(
                    pd.to_numeric(
                        target_group["cadence_index"],
                        errors="coerce",
                    ).min()
                ),
                "cadence_max": _safe_float(
                    pd.to_numeric(
                        target_group["cadence_index"],
                        errors="coerce",
                    ).max()
                ),
                "observation_start": (
                    float(times.min()) if times.notna().any() else None
                ),
                "observation_end": (
                    float(times.max()) if times.notna().any() else None
                ),
                "missing_observation_time": int(times.isna().sum()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["target_name", "pipeline"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Target/source identity
# ---------------------------------------------------------------------------


def _target_source_inventory(data: ExperimentSet) -> pd.DataFrame:
    """Show target-source identity in both production source namespaces."""
    sources = data.source_measurements.copy()
    ts = data.timeseries_measurements.copy()

    source_targets = sources.loc[
        sources["is_target_source"].map(_as_bool)
    ].copy()

    ts_targets = ts.loc[
        ts["is_target_source"].map(_as_bool)
    ].copy()

    source_columns = [
        "experiment_id",
        "target_name",
        "pipeline",
        "source_identifier",
        "valid",
        "net_flux",
        "instrumental_magnitude",
        "flux_uncertainty",
        "magnitude_uncertainty",
    ]

    source_targets = source_targets[
        [column for column in source_columns if column in source_targets.columns]
    ].rename(
        columns={
            "source_identifier": "source_identifier_source_file",
            "valid": "source_valid",
        }
    )

    ts_ids = (
        ts_targets[
            [
                "experiment_id",
                "target_name",
                "pipeline",
                "source_identifier",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "source_identifier": "source_identifier_timeseries_file",
            }
        )
    )

    return (
        source_targets
        .merge(
            ts_ids,
            on=["experiment_id", "target_name", "pipeline"],
            how="outer",
        )
        .sort_values(["target_name", "pipeline"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Target data quality
# ---------------------------------------------------------------------------


def _target_data_quality(data: ExperimentSet) -> pd.DataFrame:
    """Audit the exact valid target rows that will feed scientific metrics."""
    ts = data.timeseries_measurements.copy()
    rows: list[dict[str, Any]] = []

    grouped = ts.groupby(
        ["target_name", "pipeline"],
        dropna=False,
        sort=True,
    )

    for (target_name, pipeline), group in grouped:
        target_mask = group["is_target_source"].map(_as_bool)
        target_group = group.loc[target_mask].copy()

        valid_mask = target_group["valid"].map(_as_bool)
        valid_group = target_group.loc[valid_mask].copy()

        row: dict[str, Any] = {
            "target_name": target_name,
            "pipeline": pipeline,
            "target_rows": int(len(target_group)),
            "valid_target_rows": int(len(valid_group)),
            "invalid_target_rows": int(len(target_group) - len(valid_group)),
            "target_valid_fraction": (
                float(len(valid_group) / len(target_group))
                if len(target_group)
                else None
            ),
            "cadence_count": int(target_group["cadence_index"].nunique()),
            "cadence_min": _safe_float(
                pd.to_numeric(
                    target_group["cadence_index"],
                    errors="coerce",
                ).min()
            ),
            "cadence_max": _safe_float(
                pd.to_numeric(
                    target_group["cadence_index"],
                    errors="coerce",
                ).max()
            ),
        }

        for column in TARGET_NUMERIC_COLUMNS:
            if column not in valid_group.columns:
                row[f"{column}_missing"] = None
                row[f"{column}_non_finite"] = None
                row[f"{column}_negative"] = None
                continue

            values = pd.to_numeric(
                valid_group[column],
                errors="coerce",
            )

            missing = values.isna()
            finite = values.map(
                lambda value: (
                    math.isfinite(float(value))
                    if pd.notna(value)
                    else False
                )
            )

            row[f"{column}_missing"] = int(missing.sum())
            row[f"{column}_non_finite"] = int(
                ((~missing) & (~finite)).sum()
            )

            if column in {
                "net_flux",
                "flux_uncertainty",
                "magnitude_uncertainty",
                "background_flux",
                "background_level",
            }:
                row[f"{column}_negative"] = int(
                    ((finite) & (values < 0)).sum()
                )
            else:
                row[f"{column}_negative"] = None

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["target_name", "pipeline"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Baseline / SCARLET temporal pairing
# ---------------------------------------------------------------------------


def _pairing_audit(data: ExperimentSet) -> pd.DataFrame:
    """Verify that each target has paired Baseline/SCARLET cadence coverage."""
    ts = data.timeseries_measurements.copy()
    rows: list[dict[str, Any]] = []

    target_ts = ts.loc[ts["is_target_source"].map(_as_bool)].copy()

    for target, group in target_ts.groupby("target_name", sort=True):
        by_pipeline: dict[str, pd.DataFrame] = {
            str(pipeline): frame.copy()
            for pipeline, frame in group.groupby("pipeline", sort=False)
        }

        baseline = by_pipeline.get("Baseline")
        scarlet = by_pipeline.get("SCARLET")

        row: dict[str, Any] = {
            "target_name": target,
            "baseline_rows": int(len(baseline)) if baseline is not None else 0,
            "scarlet_rows": int(len(scarlet)) if scarlet is not None else 0,
            "baseline_cadences": (
                int(baseline["cadence_index"].nunique())
                if baseline is not None else 0
            ),
            "scarlet_cadences": (
                int(scarlet["cadence_index"].nunique())
                if scarlet is not None else 0
            ),
            "baseline_start": None,
            "baseline_end": None,
            "scarlet_start": None,
            "scarlet_end": None,
            "cadence_sets_match": False,
            "observation_times_match": False,
            "paired_rows": 0,
        }

        if baseline is None or scarlet is None:
            rows.append(row)
            continue

        baseline_times = (
            baseline[["cadence_index", "observation_time"]]
            .drop_duplicates("cadence_index")
            .sort_values("cadence_index")
            .copy()
        )
        scarlet_times = (
            scarlet[["cadence_index", "observation_time"]]
            .drop_duplicates("cadence_index")
            .sort_values("cadence_index")
            .copy()
        )

        baseline_times["observation_time"] = pd.to_numeric(
            baseline_times["observation_time"],
            errors="coerce",
        )
        scarlet_times["observation_time"] = pd.to_numeric(
            scarlet_times["observation_time"],
            errors="coerce",
        )

        baseline_cadences = set(
            baseline_times["cadence_index"].dropna().tolist()
        )
        scarlet_cadences = set(
            scarlet_times["cadence_index"].dropna().tolist()
        )

        merged = baseline_times.merge(
            scarlet_times,
            on="cadence_index",
            how="outer",
            suffixes=("_baseline", "_scarlet"),
            indicator=True,
        )

        common = merged.loc[merged["_merge"] == "both"].copy()

        row["baseline_start"] = _safe_float(
            baseline_times["observation_time"].min()
        )
        row["baseline_end"] = _safe_float(
            baseline_times["observation_time"].max()
        )
        row["scarlet_start"] = _safe_float(
            scarlet_times["observation_time"].min()
        )
        row["scarlet_end"] = _safe_float(
            scarlet_times["observation_time"].max()
        )
        row["cadence_sets_match"] = baseline_cadences == scarlet_cadences
        row["paired_rows"] = int(len(common))

        if len(common):
            row["observation_times_match"] = bool(
                (
                    common["observation_time_baseline"]
                    .astype(float)
                    .eq(common["observation_time_scarlet"].astype(float))
                ).all()
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("target_name").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Background-flux anomaly audit
# ---------------------------------------------------------------------------


def _background_flux_anomalies(data: ExperimentSet) -> pd.DataFrame:
    """Return all rows with negative background flux for manual investigation."""
    ts = data.timeseries_measurements.copy()

    if "background_flux" not in ts.columns:
        return pd.DataFrame()

    values = pd.to_numeric(ts["background_flux"], errors="coerce")
    mask = values.notna() & (values < 0)

    columns = [
        "experiment_id",
        "target_name",
        "pipeline",
        "source_identifier",
        "is_target_source",
        "valid",
        "cadence_index",
        "observation_time",
        "background_level",
        "background_flux",
        "net_flux",
        "instrumental_magnitude",
    ]

    available = [column for column in columns if column in ts.columns]

    return (
        ts.loc[mask, available]
        .copy()
        .sort_values(
            ["target_name", "pipeline", "cadence_index"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Numeric and missingness summaries
# ---------------------------------------------------------------------------


def _timeseries_numeric_summary(data: ExperimentSet) -> pd.DataFrame:
    """Summarise numeric fields relevant to later photometric analysis."""
    return _numeric_summary(
        data.timeseries_measurements,
        GLOBAL_NUMERIC_COLUMNS,
    )


def _missingness_summary(data: ExperimentSet) -> pd.DataFrame:
    """Report all missing values in the three production inputs."""
    frames: list[dict[str, Any]] = []

    for name, frame in [
        ("experiment_summary", data.experiment_summary),
        ("source_measurements", data.source_measurements),
        ("timeseries_measurements", data.timeseries_measurements),
    ]:
        for column in frame.columns:
            missing = int(frame[column].isna().sum())

            if missing:
                frames.append(
                    {
                        "dataset": name,
                        "column": column,
                        "rows": int(len(frame)),
                        "missing": missing,
                        "missing_fraction": (
                            missing / len(frame) if len(frame) else None
                        ),
                    }
                )

    if not frames:
        return pd.DataFrame(
            columns=[
                "dataset",
                "column",
                "rows",
                "missing",
                "missing_fraction",
            ]
        )

    return (
        pd.DataFrame(frames)
        .sort_values(
            ["dataset", "missing"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Explicit sanity checks
# ---------------------------------------------------------------------------


def _sanity_checks(
    data: ExperimentSet,
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run explicit pre-analysis sanity checks."""
    checks: list[dict[str, Any]] = []

    def add(
        check_name: str,
        passed: bool,
        observed: object,
        expected: object,
        detail: str,
    ) -> None:
        checks.append(
            {
                "check": check_name,
                "status": "PASS" if bool(passed) else "FAIL",
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    experiments = tables["experiment_inventory"]
    target_inventory = tables["target_inventory"]
    quality = tables["target_data_quality"]
    pairing = tables["pairing_audit"]

    experiment_targets = set(
        experiments["target_name"].dropna().astype(str).unique()
    )
    experiment_pipelines = set(
        experiments["pipeline"].dropna().astype(str).unique()
    )

    add(
        "expected_target_set",
        experiment_targets == EXPECTED_TARGETS,
        "|".join(sorted(experiment_targets)),
        "|".join(sorted(EXPECTED_TARGETS)),
        "Production experiment target set matches the seven thesis TESS targets.",
    )

    add(
        "expected_pipeline_set",
        experiment_pipelines == EXPECTED_PIPELINES,
        "|".join(sorted(experiment_pipelines)),
        "|".join(sorted(EXPECTED_PIPELINES)),
        "Both paired production pipelines are present.",
    )

    add(
        "experiment_count",
        len(experiments) == 14,
        len(experiments),
        14,
        "Expected 7 targets × 2 pipelines.",
    )

    expected_pairs = len(EXPECTED_TARGETS) * len(EXPECTED_PIPELINES)
    actual_pairs = len(target_inventory)

    add(
        "target_pipeline_pair_count",
        actual_pairs == expected_pairs,
        actual_pairs,
        expected_pairs,
        "Exactly one target/pipeline inventory row is expected per experiment.",
    )

    all_target_complete = bool(
        (
            quality["target_rows"].eq(EXPECTED_CADENCES)
            & quality["valid_target_rows"].eq(EXPECTED_CADENCES)
            & quality["invalid_target_rows"].eq(0)
            & quality["cadence_count"].eq(EXPECTED_CADENCES)
            & quality["cadence_min"].eq(0)
            & quality["cadence_max"].eq(EXPECTED_CADENCES - 1)
        ).all()
    )

    add(
        "target_cadence_completeness",
        all_target_complete,
        int(
            (
                quality["target_rows"].eq(EXPECTED_CADENCES)
                & quality["valid_target_rows"].eq(EXPECTED_CADENCES)
                & quality["invalid_target_rows"].eq(0)
            ).sum()
        ),
        expected_pairs,
        "Every target/pipeline must contain 300 valid target cadences indexed 0–299.",
    )

    required_target_fields = [
        "net_flux",
        "instrumental_magnitude",
        "flux_uncertainty",
        "magnitude_uncertainty",
        "background_level",
        "background_flux",
        "contamination_fraction",
        "observation_time",
    ]
    target_missing_columns = [
        f"{column}_missing"
        for column in required_target_fields
        if f"{column}_missing" in quality.columns
    ]
    target_non_finite_columns = [
        f"{column}_non_finite"
        for column in required_target_fields
        if f"{column}_non_finite" in quality.columns
    ]

    target_missing_ok = bool(
        not quality.empty
        and quality[target_missing_columns].fillna(0).eq(0).all().all()
        and quality[target_non_finite_columns].fillna(0).eq(0).all().all()
    )

    add(
        "target_scientific_fields_complete",
        target_missing_ok,
        "complete" if target_missing_ok else "missing/non-finite values detected",
        "zero missing and zero non-finite values",
        "Every valid target row must have complete finite values for fields used by scientific metrics.",
    )

    pairing_ok = bool(
        not pairing.empty
        and pairing["cadence_sets_match"].eq(True).all()
        and pairing["observation_times_match"].eq(True).all()
        and pairing["paired_rows"].eq(EXPECTED_CADENCES).all()
    )

    add(
        "baseline_scarlet_pairing",
        pairing_ok,
        (
            int(pairing["paired_rows"].sum())
            if not pairing.empty
            else 0
        ),
        len(EXPECTED_TARGETS) * EXPECTED_CADENCES,
        "Baseline and SCARLET target light curves must share the same cadence/time grid.",
    )

    negative_background = tables["background_flux_anomalies"]
    negative_target_background = negative_background.loc[
        negative_background["is_target_source"].map(_as_bool)
        & negative_background["valid"].map(_as_bool)
    ] if not negative_background.empty else negative_background

    add(
        "negative_target_background_flux",
        negative_target_background.empty,
        int(len(negative_target_background)),
        0,
        "Valid target rows used for final metrics must not contain negative background flux.",
    )

    target_source_count_ok = bool(
        experiments["target_source_count"].eq(1).all()
    )

    add(
        "one_target_source_per_experiment",
        target_source_count_ok,
        int(experiments["target_source_count"].eq(1).sum()),
        len(experiments),
        "Each experiment must identify exactly one target source.",
    )

    return pd.DataFrame(checks)


# ---------------------------------------------------------------------------
# Public summary API
# ---------------------------------------------------------------------------


def build_summary(data: ExperimentSet) -> dict[str, pd.DataFrame]:
    """Build all descriptive and pre-analysis audit tables."""
    tables: dict[str, pd.DataFrame] = {}

    tables["experiment_inventory"] = _experiment_inventory(data)
    tables["target_inventory"] = _target_inventory(data)
    tables["target_source_inventory"] = _target_source_inventory(data)
    tables["target_data_quality"] = _target_data_quality(data)
    tables["pairing_audit"] = _pairing_audit(data)
    tables["background_flux_anomalies"] = _background_flux_anomalies(data)
    tables["timeseries_numeric_summary"] = _timeseries_numeric_summary(data)
    tables["missingness_summary"] = _missingness_summary(data)
    tables["sanity_checks"] = _sanity_checks(data, tables)

    return tables


def print_summary(tables: dict[str, pd.DataFrame]) -> None:
    """Print the production audit tables to the console."""
    print("\n=== TESS TIME-SERIES PRODUCTION DATA SUMMARY ===")

    print("\nSanity checks:")
    print(tables["sanity_checks"].to_string(index=False))

    print("\nExperiment inventory:")
    print(tables["experiment_inventory"].to_string(index=False))

    print("\nTarget/pipeline inventory:")
    print(tables["target_inventory"].to_string(index=False))

    print("\nTarget data quality:")
    print(tables["target_data_quality"].to_string(index=False))

    print("\nBaseline/SCARLET pairing audit:")
    print(tables["pairing_audit"].to_string(index=False))

    print("\nTarget source identity:")
    print(tables["target_source_inventory"].to_string(index=False))

    print("\nNegative background-flux records:")
    anomalies = tables["background_flux_anomalies"]
    if anomalies.empty:
        print("None detected.")
    else:
        print(anomalies.to_string(index=False))

    print("\nTime-series numeric ranges:")
    print(tables["timeseries_numeric_summary"].to_string(index=False))

    print("\nMissing values:")
    missing = tables["missingness_summary"]
    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.to_string(index=False))


def write_summary(
    tables: dict[str, pd.DataFrame],
    output_directory: Path = SUMMARY_OUTPUT_DIR,
) -> None:
    """Write reproducible audit tables to the canonical analysis directory."""
    output_directory.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        table.to_csv(
            output_directory / f"{name}.csv",
            index=False,
        )


def run() -> dict[str, pd.DataFrame]:
    """Run the complete canonical production-data summary."""
    if not PRODUCTION_INPUT_DIR.exists():
        raise FileNotFoundError(
            "Canonical production input directory does not exist: "
            f"{PRODUCTION_INPUT_DIR}"
        )

    data = canonicalize(
        load_experiment_set(PRODUCTION_INPUT_DIR)
    )

    tables = build_summary(data)

    print_summary(tables)

    write_summary(
        tables,
        SUMMARY_OUTPUT_DIR,
    )

    print(
        f"\nSummary CSVs written to: {SUMMARY_OUTPUT_DIR}"
    )

    return tables


def main() -> int:
    """CLI entry point with no user-supplied parameters."""
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
