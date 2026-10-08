"""Production orchestrator for TESS time-series photometry analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import pandas as pd

from .canonical import canonicalize
from .dataset import attach_source_context, target_timeseries
from .io import ExperimentSet, load_experiment_set, write_csv
from .metrics import add_normalised_flux_columns, calculate_target_metrics
from .pairing import pair_target_timeseries
from .statistics import (
    build_experiment_context,
    build_overall_summary,
    build_source_context,
    build_target_summary,
)
from .validation import validate_structure, validation_summary


DEFAULT_INPUT_DIRECTORY: Final[Path] = (
    Path("results")
    / "exports"
    / "timeseries_photometry"
)

DEFAULT_OUTPUT_DIRECTORY: Final[Path] = (
    Path("results")
    / "analysis"
    / "timeseries_photometry"
)


def run_analysis(
    input_directory: Path | str = DEFAULT_INPUT_DIRECTORY,
    output_directory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
    *,
    expected_cadences: int | None = None,
    fail_on_validation_error: bool = True,
) -> dict[str, Path]:
    """Run the complete three-input production analysis.

    Returns a mapping of stable output names to their generated paths.
    """
    data = canonicalize(load_experiment_set(input_directory))

    validation = validate_structure(
        data,
        expected_cadences=expected_cadences,
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    validation_report_path = write_csv(
        validation.issues,
        output / "validation_report.csv",
    )

    validation_summary_path = output / "validation_summary.json"
    validation_summary_path.write_text(
        json.dumps(
            validation_summary(validation),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if not validation.passed and fail_on_validation_error:
        raise ValueError(
            "Time-series input validation failed. "
            f"See {validation_report_path}"
        )

    target_ts = target_timeseries(data)
    target_ts_context = attach_source_context(data, target_ts)

    paired = pair_target_timeseries(data)
    paired = add_normalised_flux_columns(paired)

    metrics = calculate_target_metrics(paired)
    target_summary = build_target_summary(metrics)
    overall_summary = build_overall_summary(metrics)

    experiment_context = build_experiment_context(
        data.experiment_summary
    )
    source_context = build_source_context(
        data.source_measurements
    )

    output_map = {
        "validation_report": validation_report_path,
        "validation_summary": validation_summary_path,
        "experiment_context": write_csv(
            experiment_context,
            output / "experiment_context.csv",
        ),
        "source_context": write_csv(
            source_context,
            output / "source_context.csv",
        ),
        "target_timeseries_canonical": write_csv(
            target_ts_context,
            output / "target_timeseries_canonical.csv",
        ),
        "paired_timeseries": write_csv(
            paired,
            output / "paired_timeseries.csv",
        ),
        "paired_timeseries_normalised": write_csv(
            paired[
                [
                    column
                    for column in paired.columns
                    if column in {
                        "target_name",
                        "cadence_index",
                        "source_identifier",
                        "baseline_time",
                        "scarlet_time",
                        "baseline_flux_normalised",
                        "scarlet_flux_normalised",
                        "pair_status",
                    }
                ]
            ],
            output / "paired_timeseries_normalised.csv",
        ),
        "target_metrics": write_csv(
            metrics,
            output / "target_timeseries_metrics.csv",
        ),
        "target_summary": write_csv(
            target_summary,
            output / "target_timeseries_summary.csv",
        ),
        "overall_summary": write_csv(
            overall_summary,
            output / "overall_timeseries_summary.csv",
        ),
    }

    metadata = {
        "input_directory": str(Path(input_directory).resolve()),
        "output_directory": str(output.resolve()),
        "input_rows": {
            "experiment_summary": int(len(data.experiment_summary)),
            "source_measurements": int(len(data.source_measurements)),
            "timeseries_measurements": int(len(data.timeseries_measurements)),
            "target_timeseries": int(len(target_ts)),
            "paired_timeseries": int(
                (paired["pair_status"] == "paired").sum()
            ),
        },
        "validation": validation_summary(validation),
        "analysis": {
            "normalisation": "independent median normalisation per pipeline and target",
            "primary_scatter_metric": "standard deviation of median-normalised target flux",
            "temporal_stability_metric": "standard deviation of cadence-to-cadence differences",
            "pairing_key": [
                "target_name",
                "cadence_index",
                "source_identifier",
            ],
        },
    }

    metadata_path = output / "analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_map["analysis_metadata"] = metadata_path

    return output_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run production TESS time-series photometry analysis.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help=(
            "Directory containing experiment_summary.csv, "
            "source_measurements.csv and timeseries_measurements.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--expected-cadences",
        type=int,
        default=None,
        help="Optional regression expectation, e.g. 300.",
    )
    parser.add_argument(
        "--allow-validation-errors",
        action="store_true",
        help="Continue analysis even when structural validation reports errors.",
    )

    args = parser.parse_args()

    outputs = run_analysis(
        args.input_dir,
        args.output_dir,
        expected_cadences=args.expected_cadences,
        fail_on_validation_error=not args.allow_validation_errors,
    )

    for name, path in outputs.items():
        print(f"{name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
