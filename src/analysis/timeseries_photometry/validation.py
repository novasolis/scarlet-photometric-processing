"""Structural and relational validation for the complete time-series experiment.

The three production exports use different source-identifier namespaces.
``source_measurements.csv`` uses experiment-local source identifiers such as
``12`` while ``timeseries_measurements.csv`` uses identifiers such as
``TS-001``. These identifiers are therefore never compared directly.

Cross-file experiment identity is established with:
    experiment_id + target_name + pipeline

Within the time-series table, Baseline and SCARLET cadence rows are paired by:
    target_name + cadence_index + source_identifier
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .dataset import target_sources, target_timeseries
from .io import ExperimentSet


PAIR_KEY = ["target_name", "cadence_index", "source_identifier"]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Complete validation report and pass/fail state."""

    issues: pd.DataFrame

    @property
    def errors(self) -> int:
        return int((self.issues["severity"] == "ERROR").sum())

    @property
    def warnings(self) -> int:
        return int((self.issues["severity"] == "WARNING").sum())

    @property
    def passed(self) -> bool:
        return self.errors == 0


def _issue(
    target: str,
    code: str,
    message: str,
    severity: str = "ERROR",
) -> dict[str, str]:
    return {
        "target_name": target,
        "severity": severity,
        "code": code,
        "message": message,
    }


def _required_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> list[dict[str, str]]:
    missing = sorted(set(columns).difference(dataframe.columns))
    if not missing:
        return []

    return [
        _issue(
            "<dataset>",
            "MISSING_COLUMN",
            f"{name} is missing required columns: {', '.join(missing)}",
        )
    ]


def validate_structure(
    data: ExperimentSet,
    *,
    expected_pipelines: tuple[str, str] = ("Baseline", "SCARLET"),
    expected_cadences: int | None = None,
    time_tolerance: float = 1e-10,
) -> ValidationResult:
    """Validate all three datasets and their relational consistency."""
    issues: list[dict[str, str]] = []

    issues.extend(
        _required_columns(
            data.experiment_summary,
            {
                "experiment_id",
                "target_name",
                "survey",
                "experiment_type",
                "pipeline",
            },
            "experiment_summary.csv",
        )
    )
    issues.extend(
        _required_columns(
            data.source_measurements,
            {
                "experiment_id",
                "target_name",
                "pipeline",
                "source_identifier",
                "is_target_source",
                "valid",
            },
            "source_measurements.csv",
        )
    )
    issues.extend(
        _required_columns(
            data.timeseries_measurements,
            {
                "experiment_id",
                "target_name",
                "pipeline",
                "cadence_index",
                "observation_time",
                "source_identifier",
                "is_target_source",
                "valid",
            },
            "timeseries_measurements.csv",
        )
    )

    if issues:
        return ValidationResult(pd.DataFrame(issues))

    summary = data.experiment_summary
    sources = data.source_measurements
    ts = data.timeseries_measurements

    if summary["experiment_id"].duplicated().any():
        issues.append(
            _issue(
                "<dataset>",
                "DUPLICATE_EXPERIMENT_ID",
                "experiment_summary.csv contains duplicate experiment_id values.",
            )
        )

    target_ts = target_timeseries(data)
    target_src = target_sources(data)

    # Validate the experiment-level target/pipeline structure.
    for raw_target, group in summary.groupby("target_name", sort=True):
        target = str(raw_target)
        pipelines = {
            str(value)
            for value in group["pipeline"].dropna().tolist()
        }

        for pipeline in expected_pipelines:
            if pipeline not in pipelines:
                issues.append(
                    _issue(
                        target,
                        "MISSING_PIPELINE",
                        f"Missing {pipeline} experiment in experiment_summary.csv.",
                    )
                )

            current = group.loc[
                group["pipeline"].astype("string") == pipeline
            ]

            if len(current) > 1:
                issues.append(
                    _issue(
                        target,
                        "MULTIPLE_EXPERIMENTS",
                        f"More than one {pipeline} experiment exists for this target.",
                    )
                )

            if len(current) != 1:
                continue

            experiment_id = str(current.iloc[0]["experiment_id"])

            current_ts = target_ts.loc[
                target_ts["experiment_id"].astype("string") == experiment_id
            ]

            if current_ts.empty:
                issues.append(
                    _issue(
                        target,
                        "NO_TARGET_TIMESERIES",
                        f"No target time-series rows found for {pipeline} "
                        f"experiment {experiment_id}.",
                    )
                )
                continue

            duplicate = current_ts.duplicated(
                ["experiment_id", "cadence_index", "source_identifier"],
                keep=False,
            )
            if duplicate.any():
                issues.append(
                    _issue(
                        target,
                        "DUPLICATE_TIMESERIES_KEYS",
                        f"{pipeline} contains {int(duplicate.sum())} rows "
                        "involved in duplicate cadence/source keys.",
                    )
                )

            if expected_cadences is not None:
                cadence_count = int(current_ts["cadence_index"].nunique())
                if cadence_count != expected_cadences:
                    issues.append(
                        _issue(
                            target,
                            "UNEXPECTED_CADENCE_COUNT",
                            f"{pipeline} has {cadence_count} target cadences; "
                            f"expected {expected_cadences}.",
                        )
                    )

            if current_ts["cadence_index"].isna().any():
                issues.append(
                    _issue(
                        target,
                        "MISSING_CADENCE_INDEX",
                        f"{pipeline} contains target rows with missing cadence_index.",
                    )
                )

            if current_ts["observation_time"].isna().any():
                issues.append(
                    _issue(
                        target,
                        "MISSING_OBSERVATION_TIME",
                        f"{pipeline} contains target rows with missing "
                        "observation_time.",
                    )
                )

    # Validate source-level target cardinality independently. The source ID
    # namespace is local to source_measurements.csv and is not compared with
    # the TS-* namespace used by timeseries_measurements.csv.
    source_target_counts = (
        target_src.groupby("experiment_id", dropna=False)
        .size()
    )
    invalid_source_targets = source_target_counts[
        source_target_counts != 1
    ]
    if not invalid_source_targets.empty:
        issues.append(
            _issue(
                "<dataset>",
                "SOURCE_TARGET_COUNT",
                f"{len(invalid_source_targets)} experiments do not have "
                "exactly one target source in source_measurements.csv.",
            )
        )

    # Cross-file identity is semantic, not source-ID based.
    identity = ["experiment_id", "target_name", "pipeline"]

    summary_identity = set(
        map(
            tuple,
            summary[identity].drop_duplicates().itertuples(
                index=False, name=None
            ),
        )
    )
    source_identity = set(
        map(
            tuple,
            sources[identity].drop_duplicates().itertuples(
                index=False, name=None
            ),
        )
    )
    ts_identity = set(
        map(
            tuple,
            ts[identity].drop_duplicates().itertuples(
                index=False, name=None
            ),
        )
    )

    missing_source_identity = summary_identity - source_identity
    if missing_source_identity:
        issues.append(
            _issue(
                "<dataset>",
                "SUMMARY_SOURCE_IDENTITY_MISMATCH",
                f"{len(missing_source_identity)} experiment/target/pipeline "
                "identities from experiment_summary.csv have no "
                "source_measurements row.",
            )
        )

    missing_ts_identity = summary_identity - ts_identity
    if missing_ts_identity:
        issues.append(
            _issue(
                "<dataset>",
                "SUMMARY_TIMESERIES_IDENTITY_MISMATCH",
                f"{len(missing_ts_identity)} experiment/target/pipeline "
                "identities from experiment_summary.csv have no "
                "timeseries_measurements row.",
            )
        )

    source_without_ts = source_identity - ts_identity
    ts_without_source = ts_identity - source_identity
    if source_without_ts or ts_without_source:
        issues.append(
            _issue(
                "<dataset>",
                "TARGET_IDENTITY_MISMATCH",
                "Experiment/target/pipeline identities differ between "
                "source_measurements.csv and timeseries_measurements.csv.",
            )
        )

    # Baseline/SCARLET timestamp pairing.
    #
    # This is deliberately performed inside timeseries_measurements.csv.
    # source_identifier is meaningful here because both pipelines use the
    # same TS-* time-series namespace for their corresponding target source.
    for raw_target, group in target_ts.groupby("target_name", sort=True):
        target = str(raw_target)

        baseline = group.loc[
            group["pipeline"].astype("string") == "Baseline",
            PAIR_KEY + ["observation_time"],
        ].rename(columns={"observation_time": "baseline_time"})

        scarlet = group.loc[
            group["pipeline"].astype("string") == "SCARLET",
            PAIR_KEY + ["observation_time"],
        ].rename(columns={"observation_time": "scarlet_time"})

        if baseline.empty or scarlet.empty:
            continue

        paired = baseline.merge(
            scarlet,
            on=PAIR_KEY,
            how="outer",
            indicator=True,
        )

        if (paired["_merge"] != "both").any():
            left = int((paired["_merge"] == "left_only").sum())
            right = int((paired["_merge"] == "right_only").sum())
            issues.append(
                _issue(
                    target,
                    "UNPAIRED_CADENCES",
                    f"Baseline-only rows: {left}; SCARLET-only rows: {right}.",
                )
            )

        both = paired.loc[paired["_merge"] == "both"].copy()
        if not both.empty:
            delta = (
                pd.to_numeric(both["baseline_time"], errors="coerce")
                - pd.to_numeric(both["scarlet_time"], errors="coerce")
            ).abs()

            if delta.notna().any():
                maximum = float(delta.max())
                if maximum > time_tolerance:
                    issues.append(
                        _issue(
                            target,
                            "TIMESTAMP_MISMATCH",
                            f"Maximum Baseline/SCARLET timestamp difference "
                            f"is {maximum:.12g}; tolerance is "
                            f"{time_tolerance:.12g}.",
                        )
                    )

    if not issues:
        issues.append(
            _issue(
                "<dataset>",
                "VALID",
                "All three production inputs and their relationships passed validation.",
                "INFO",
            )
        )

    return ValidationResult(pd.DataFrame(issues))


def validation_summary(result: ValidationResult) -> dict[str, object]:
    """Return the compact summary consumed by integration tests."""
    return {
        "status": "PASS" if result.passed else "FAIL",
        "errors": result.errors,
        "warnings": result.warnings,
    }
