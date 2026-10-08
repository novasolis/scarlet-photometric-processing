"""Production orchestrator for TESS differential-photometry RQ4 analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd

from .dataset import load_differential_measurements, load_target_timeseries
from .metrics import (
    differential_measurement_summary,
    differential_timeseries_metrics,
    paired_light_curve_correlation,
    paired_metric_changes,
)
from .validation import validate_differential_measurements


DEFAULT_INPUT_DIRECTORY: Final[Path] = (
    Path("results") / "exports" / "differential_photometry"
)

DEFAULT_TARGET_TIMESERIES_DIRECTORY: Final[Path] = (
    Path("results") / "exports" / "timeseries_photometry"
)

DEFAULT_OUTPUT_DIRECTORY: Final[Path] = (
    Path("results") / "analysis" / "differential_photometry"
)

DIFFERENTIAL_FILENAME: Final[str] = "differential_timeseries_measurements.csv"
TIMESERIES_FILENAME: Final[str] = "timeseries_measurements.csv"


def _resolve_csv(directory: Path, filename: str) -> Path:
    """Resolve a canonical export CSV from a project results directory."""
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required analysis input was not found: {path.resolve()}"
        )
    return path


def run_analysis(
    input_directory: Path | str = DEFAULT_INPUT_DIRECTORY,
    output_directory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
    *,
    target_timeseries_directory: Path | str = DEFAULT_TARGET_TIMESERIES_DIRECTORY,
    expected_cadences: int = 300,
    fail_on_validation_error: bool = True,
) -> dict[str, Path]:
    """Run the complete compact differential-photometry RQ4 analysis."""

    input_directory = Path(input_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    differential_csv = _resolve_csv(
        input_directory,
        DIFFERENTIAL_FILENAME,
    )
    target_timeseries_csv = _resolve_csv(
        Path(target_timeseries_directory),
        TIMESERIES_FILENAME,
    )

    differential = load_differential_measurements(differential_csv)

    validation = validate_differential_measurements(
        differential,
        expected_targets=tuple(sorted(differential["target_name"].unique())),
        expected_cadences=expected_cadences,
    )

    validation_path = output / "validation_summary.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not validation["valid"] and fail_on_validation_error:
        raise ValueError(
            "Differential-photometry input validation failed. "
            f"See {validation_path.resolve()}"
        )

    measurement_summary = differential_measurement_summary(differential)
    timeseries_summary = differential_timeseries_metrics(differential)
    metric_changes = paired_metric_changes(timeseries_summary)
    correlations = paired_light_curve_correlation(differential)

    target = load_target_timeseries(target_timeseries_csv)

    target_rows: list[dict[str, object]] = []
    for (target_name, pipeline), group in target.groupby(
        ["target_name", "pipeline"],
        sort=True,
    ):
        group = group.sort_values("cadence_index")
        flux = group["net_flux"].to_numpy(dtype=float)
        median_flux = float(pd.Series(flux).median())
        normalised = pd.Series(flux / median_flux - 1.0)

        target_rows.append(
            {
                "target_name": target_name,
                "pipeline": pipeline,
                "target_flux_scatter_percent": float(
                    normalised.std(ddof=1) * 100
                ),
                "target_temporal_stability_percent": float(
                    normalised.diff().std(ddof=1) * 100
                ),
            }
        )

    target_comparison = pd.DataFrame(target_rows)

    outputs = {
        "validation_summary": validation_path,
        "differential_measurement_summary": output
        / "differential_measurement_summary.csv",
        "differential_timeseries_metrics": output
        / "differential_timeseries_metrics.csv",
        "differential_metric_changes": output
        / "differential_metric_changes.csv",
        "differential_light_curve_correlations": output
        / "differential_light_curve_correlations.csv",
        "target_timeseries_comparison_metrics": output
        / "target_timeseries_comparison_metrics.csv",
    }

    measurement_summary.to_csv(
        outputs["differential_measurement_summary"],
        index=False,
    )
    timeseries_summary.to_csv(
        outputs["differential_timeseries_metrics"],
        index=False,
    )
    metric_changes.to_csv(
        outputs["differential_metric_changes"],
        index=False,
    )
    correlations.to_csv(
        outputs["differential_light_curve_correlations"],
        index=False,
    )
    target_comparison.to_csv(
        outputs["target_timeseries_comparison_metrics"],
        index=False,
    )

    metadata = {
        "input_directory": str(input_directory.resolve()),
        "target_timeseries_directory": str(
            Path(target_timeseries_directory).resolve()
        ),
        "output_directory": str(output.resolve()),
        "input_files": {
            "differential_measurements": str(differential_csv.resolve()),
            "target_timeseries": str(target_timeseries_csv.resolve()),
        },
        "input_rows": {
            "differential_measurements": int(len(differential)),
            "target_timeseries": int(len(target)),
        },
        "targets": sorted(differential["target_name"].unique().tolist()),
        "pipelines": sorted(differential["pipeline"].unique().tolist()),
        "analysis": {
            "primary_measurement_comparisons": [
                "differential_flux",
                "differential_magnitude",
            ],
            "primary_precision_metrics": [
                "relative differential-flux scatter",
                "differential-magnitude scatter",
            ],
            "temporal_stability_metric": (
                "standard deviation of cadence-to-cadence differences "
                "in median-normalised differential flux"
            ),
            "pairing_key": [
                "target_name",
                "pipeline",
                "cadence_index",
            ],
            "baseline_pipeline": "Baseline",
            "comparison_pipeline": "SCARLET",
        },
    }

    metadata_path = output / "analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    outputs["analysis_metadata"] = metadata_path

    return outputs


def main() -> int:
    """Run RQ4 using the project's standard results directories."""
    outputs = run_analysis()

    for name, path in outputs.items():
        print(f"{name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
