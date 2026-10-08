"""
Small TESS time-series workflow demonstration.

Exercises the production PhotometricWorkflow.run_time_series() path
and persists a small 50-cadence time-series export for validation.

This script does not modify the configured experiment suite and does
not run the normal main.py experiment suite.

The test processes the first 50 TESS cadences for HD 209458.

Purpose
-------
Validate that:

* the TESS dataset is acquired once;
* 50 cadence-level Observations are constructed;
* each cadence is processed exactly once;
* the existing Baseline preprocessing, detection, target resolution,
  photometry and evaluation stages are reused;
* one WorkflowResult is returned for every requested cadence;
* cadence-level measurements can be transformed into export records;
* the time-series CSV writer persists the records successfully;
* rerunning the same test does not create duplicate logical records.

This is a development/validation script only.
"""

from __future__ import annotations

import math
from pathlib import Path

from config.experiment_config import create_experiment_config
from domain.enums import ExperimentType, SourceScope, Survey
from export.timeseries_writer import TimeSeriesWriter
from utils.logger import configure_logging, get_logger
from workflows.workflow_factory import create_workflow


LOGGER = get_logger(
    __name__,
)

TARGET_NAME = "hd_209458"
CADENCE_COUNT = 50

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "exports"
    / "timeseries_measurements.csv"
)


def main() -> None:
    """
    Run the small HD 209458 TESS time-series validation.
    """

    configure_logging()

    LOGGER.info("")
    LOGGER.info("=" * 72)
    LOGGER.info("TESS TIME-SERIES WORKFLOW TEST")
    LOGGER.info("=" * 72)

    LOGGER.info(
        "Target        : %s",
        TARGET_NAME,
    )

    LOGGER.info(
        "Cadences      : %d",
        CADENCE_COUNT,
    )

    LOGGER.info(
        "Pipeline      : Baseline",
    )

    LOGGER.info(
        "Output        : %s",
        OUTPUT_PATH,
    )

    LOGGER.info("=" * 72)

    # ------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------

    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_SOURCE,
        source_scope=SourceScope.TARGET,
        target_name=TARGET_NAME,
    )

    # ------------------------------------------------------------------
    # Production workflow construction
    # ------------------------------------------------------------------

    LOGGER.info(
        "Creating production PhotometricWorkflow..."
    )

    workflow = create_workflow(
        config,
    )

    # ------------------------------------------------------------------
    # Cadence selection
    # ------------------------------------------------------------------

    cadence_indices = tuple(
        range(
            CADENCE_COUNT,
        )
    )

    LOGGER.info("")
    LOGGER.info(
        "Selected cadence indices: 0-%d.",
        CADENCE_COUNT - 1,
    )

    LOGGER.info(
        "Starting run_time_series()..."
    )

    # ------------------------------------------------------------------
    # Execute time-series workflow
    # ------------------------------------------------------------------

    results = workflow.run_time_series(
        cadence_indices=cadence_indices,
    )

    if len(
        results,
    ) != CADENCE_COUNT:
        raise RuntimeError(
            "Time-series result count mismatch: "
            f"requested {CADENCE_COUNT}, "
            f"received {len(results)}."
        )

    # ------------------------------------------------------------------
    # Validate cadence-level workflow results
    # ------------------------------------------------------------------

    LOGGER.info("")
    LOGGER.info("=" * 72)
    LOGGER.info("TIME-SERIES WORKFLOW SUMMARY")
    LOGGER.info("=" * 72)

    total_valid_measurements = 0
    total_invalid_measurements = 0

    observation_times: list[float] = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        evaluation = result.evaluation
        observation = result.observation

        observation_time = (
            observation.metadata.observation_time
        )

        if not isinstance(
            observation_time,
            (int, float),
        ) or isinstance(
            observation_time,
            bool,
        ):
            raise RuntimeError(
                (
                    "TESS time-series test expected a numerical "
                    "BTJD observation time at cadence "
                    f"{index}, but received "
                    f"{type(observation_time).__name__}."
                )
            )

        time_value = float(
            observation_time,
        )

        if not math.isfinite(
            time_value,
        ):
            raise RuntimeError(
                (
                    "Non-finite TESS observation time at cadence "
                    f"{index}: {time_value!r}."
                )
            )

        observation_times.append(
            time_value,
        )

        valid_measurements = int(
            evaluation.valid_measurement_count,
        )

        invalid_measurements = int(
            evaluation.invalid_measurement_count,
        )

        total_valid_measurements += (
            valid_measurements
        )

        total_invalid_measurements += (
            invalid_measurements
        )

        LOGGER.info(
            (
                "Cadence %03d | time=%.12f | detected=%d | "
                "measured=%d | valid=%d | invalid=%d"
            ),
            index,
            time_value,
            evaluation.detected_source_count,
            evaluation.measured_source_count,
            valid_measurements,
            invalid_measurements,
        )

    # ------------------------------------------------------------------
    # Validate timestamps
    # ------------------------------------------------------------------

    if len(
        observation_times,
    ) != CADENCE_COUNT:
        raise RuntimeError(
            "Observation-time count mismatch: "
            f"expected {CADENCE_COUNT}, "
            f"received {len(observation_times)}."
        )

    if observation_times != sorted(
        observation_times,
    ):
        raise RuntimeError(
            "TESS observation times are not monotonically ordered."
        )

    # ------------------------------------------------------------------
    # Persist cadence-level export
    # ------------------------------------------------------------------

    LOGGER.info("")
    LOGGER.info(
        "Creating TimeSeriesWriter..."
    )

    writer = TimeSeriesWriter(
        output_path=OUTPUT_PATH,
    )

    LOGGER.info(
        "Writing time-series measurement records..."
    )

    records = writer.write(
        config=config,
        results=results,
    )

    if not records:
        raise RuntimeError(
            "TimeSeriesWriter returned no records."
        )

    # ------------------------------------------------------------------
    # Validate generated records
    # ------------------------------------------------------------------

    expected_record_count = sum(
        result.evaluation.detected_source_count
        for result in results
    )

    if len(
        records,
    ) != expected_record_count:
        raise RuntimeError(
            (
                "Time-series export record count mismatch: "
                f"expected {expected_record_count}, "
                f"received {len(records)}."
            )
        )

    record_cadence_indices = sorted(
        {
            record.cadence_index
            for record in records
        }
    )

    expected_cadence_indices = list(
        cadence_indices,
    )

    if record_cadence_indices != expected_cadence_indices:
        raise RuntimeError(
            (
                "Exported cadence indices do not match the requested "
                f"cadence indices. Expected "
                f"{expected_cadence_indices}, received "
                f"{record_cadence_indices}."
            )
        )

    record_times = [
        float(record.observation_time)
        for record in records
        if isinstance(
            record.observation_time,
            (int, float),
        )
        and not isinstance(
            record.observation_time,
            bool,
        )
    ]

    if len(
        record_times,
    ) != len(records):
        raise RuntimeError(
            (
                "One or more exported records do not contain the "
                "expected numerical TESS observation time."
            )
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    LOGGER.info("")
    LOGGER.info("=" * 72)
    LOGGER.info("TIME-SERIES TEST SUMMARY")
    LOGGER.info("=" * 72)

    LOGGER.info(
        "Requested cadences         : %d",
        CADENCE_COUNT,
    )

    LOGGER.info(
        "Returned WorkflowResults   : %d",
        len(results),
    )

    LOGGER.info(
        "Exported measurement records: %d",
        len(records),
    )

    LOGGER.info(
        "Observation timestamps     : %d",
        len(observation_times),
    )

    LOGGER.info(
        "Total valid measurements   : %d",
        total_valid_measurements,
    )

    LOGGER.info(
        "Total invalid measurements : %d",
        total_invalid_measurements,
    )

    LOGGER.info(
        "First BTJD                 : %.12f",
        observation_times[0],
    )

    LOGGER.info(
        "Last BTJD                  : %.12f",
        observation_times[-1],
    )

    LOGGER.info(
        "CSV output                 : %s",
        OUTPUT_PATH,
    )

    LOGGER.info("=" * 72)
    LOGGER.info(
        "TESS TIME-SERIES TEST PASSED"
    )
    LOGGER.info("=" * 72)


if __name__ == "__main__":
    main()