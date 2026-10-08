"""
Application entry point for the photometric analysis experiment suite.

This module is responsible only for application-level orchestration:

    ExperimentConfig
        ↓
    create_workflow()
        ↓
    PhotometricWorkflow
        ↓
    WorkflowResult(s)
        ↓
    Exporter / Time-Series Writers
        ↓
    persisted experiment products

Scientific processing remains inside the production workflow and its
injected components. Post-experiment scientific analysis and figure
generation remain separate consumers of the persisted experiment products.

Execution is controlled by the active experiment set defined in
``config.experiments``.

Single-observation target-photometry experiments use:

    WorkflowResult
        ->
    ExperimentExporter
        ->
    ExperimentSummary + SourceMeasurementRecord
        ->
    ExportWriter

Field-analysis experiments use:

    FieldAnalysisResult
        ->
    field-analysis summary logging
        ->
    return

FIELD_ANALYSIS deliberately produces no photometric export because no
photometry or evaluation is performed.

Time-series experiments use:

    tuple[WorkflowResult, ...]
        ->
    TimeSeriesWriter
        ->
    TimeSeriesMeasurementRecord

Differential-photometry time-series experiments additionally use:

    tuple[WorkflowResult, ...]
        ->
    DifferentialTimeSeriesWriter
        ->
    DifferentialTimeSeriesMeasurementRecord

The complete cadence-level outputs are persisted independently of the
representative first-cadence standard experiment/source summary export.

Important architectural boundary
---------------------------------
``main.py`` does not perform scientific calculations. It does not alter
baseline or SCARLET processing semantics, photometric calculations,
evaluation metrics, or post-experiment analysis.

The same ``ExperimentConfig`` is passed unchanged to the workflow factory
and exporter, preserving experimental provenance.

For production execution, a suite with one or more failed experiments is
treated as an unsuccessful application run after all configured
experiments have been attempted. Partial exports must therefore be
reviewed before they are used by downstream analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from config.experiments import (
    ACTIVE_EXPERIMENT_SET,
    get_active_experiments,
)
from core.models import ExperimentConfig
from domain.enums import ExperimentType
from export.differential_timeseries_writer import (
    DifferentialTimeSeriesWriter,
)
from export.exporter import ExperimentExporter
from export.models import (
    DifferentialTimeSeriesMeasurementRecord,
    ExperimentSummary,
    SourceMeasurementRecord,
    TimeSeriesMeasurementRecord,
)
from export.timeseries_writer import TimeSeriesWriter
from export.writer import ExportWriter
from utils.logger import (
    configure_logging,
    get_logger,
)
from workflows.models import FieldAnalysisResult, WorkflowResult
from workflows.photometric_workflow import PhotometricWorkflow
from workflows.workflow_factory import create_workflow


logger = get_logger(__name__)


# ============================================================================
# Export lifecycle
# ============================================================================


EXPORT_ROOT = Path("results") / "exports"


def _experiment_set_export_directory() -> Path:
    """
    Return the export directory for the active experiment set.

    Each experiment set owns an independent export namespace beneath
    ``results/exports``. This keeps reset and persistence operations scoped
    to the currently active experiment set without changing the exporter
    or writer semantics.
    """
    return EXPORT_ROOT / ACTIVE_EXPERIMENT_SET.value


def _reset_export_outputs(
    *,
    writer: ExportWriter,
    time_series_writer: TimeSeriesWriter,
    differential_time_series_writer: DifferentialTimeSeriesWriter,
) -> None:
    """
    Reset primary CSV exports before executing a new experiment suite.

    One invocation of ``main()`` represents one complete experiment-suite
    execution. Existing primary CSV outputs are therefore removed before
    new results are written.

    This operation does not modify scientific source data, cached acquisition
    products, or post-experiment analysis outputs.

    Parameters
    ----------
    writer
        Standard experiment/source export writer.

    time_series_writer
        Cadence-level time-series writer.

    differential_time_series_writer
        Differential-photometry time-series writer.

    Raises
    ------
    OSError
        If an existing export path is not a file or cannot be removed.
    """
    export_paths = (
        writer.summary_path,
        writer.source_measurements_path,
        time_series_writer.output_path,
        differential_time_series_writer.output_path,
    )

    logger.info(
        "Resetting primary experiment-suite CSV exports.",
    )

    for path in export_paths:

        if not path.exists():
            logger.info(
                "Export does not exist; nothing to reset: %s",
                path,
            )
            continue

        if not path.is_file():
            raise OSError(
                f"Export path exists but is not a file: {path}",
            )

        try:
            path.unlink()
        except OSError as exc:
            raise OSError(
                f"Failed to reset export file: {path}",
            ) from exc

        logger.info(
            "Removed previous export: %s",
            path,
        )

    logger.info(
        "Primary experiment-suite CSV exports reset successfully.",
    )


# ============================================================================
# Application entry point
# ============================================================================


def main() -> None:
    """
    Execute the active photometric experiment suite.

    The active study scope is defined centrally by
    ``config.experiments.ACTIVE_EXPERIMENT_SET``.

    Each experiment is executed independently. A failure in one experiment
    is logged and does not prevent subsequent configured experiments from
    being attempted.

    Execution modes
    ---------------

    Single observation
        1. Create the production workflow.
        2. Execute ``workflow.run()``.
        3. Report the completed ``WorkflowResult``.
        4. Transform the result using ``ExperimentExporter``.
        5. Persist the standard export using ``ExportWriter``.

    Time series
        1. Create the production workflow.
        2. Execute ``workflow.run_time_series()``.
        3. Validate the returned cadence-level results.
        4. Persist all cadence-level source measurements.
        5. Persist differential cadence-level measurements when required.
        6. Export the first successfully processed cadence as the
           representative standard experiment/source result.

    Notes
    -----
    Baseline and SCARLET are processing alternatives represented by
    ``ExperimentConfig.pipeline``. ``main.py`` does not special-case either
    pipeline. The workflow factory owns that distinction.

    Post-experiment analysis is intentionally not executed here. Persisted
    experiment products are the stable boundary consumed by analysis and
    figure-generation modules.
    """
    configure_logging()

    active_experiments = get_active_experiments()

    exporter = ExperimentExporter(
        logger=logger,
    )

    export_directory = _experiment_set_export_directory()

    writer = ExportWriter(
        output_directory=export_directory,
    )

    time_series_writer = TimeSeriesWriter(
        output_path=(
            export_directory
            / "timeseries_measurements.csv"
        ),
        logger=logger,
    )

    differential_time_series_writer = (
        DifferentialTimeSeriesWriter(
            output_path=(
                export_directory
                / "differential_timeseries_measurements.csv"
            ),
            logger=logger,
        )
    )

    total_experiments = len(
        active_experiments,
    )

    logger.info("")
    logger.info("=" * 72)
    logger.info(
        "PHOTOMETRIC ANALYSIS EXPERIMENT SUITE",
    )
    logger.info("=" * 72)

    logger.info(
        "Active experiment set: %s",
        ACTIVE_EXPERIMENT_SET.value,
    )

    logger.info(
        "Experiment-set export directory: %s",
        export_directory,
    )

    logger.info(
        "Configured experiments: %d",
        total_experiments,
    )

    logger.info(
        "Standard export directory: %s",
        writer.output_directory,
    )

    logger.info(
        "Time-series export path: %s",
        time_series_writer.output_path,
    )

    logger.info(
        "Differential time-series export path: %s",
        differential_time_series_writer.output_path,
    )

    if total_experiments == 0:
        logger.warning(
            "Active experiment set contains no experiments.",
        )
        return

    _reset_export_outputs(
        writer=writer,
        time_series_writer=time_series_writer,
        differential_time_series_writer=(
            differential_time_series_writer
        ),
    )

    successful_experiments = 0
    failed_experiments = 0

    for index, config in enumerate(
        active_experiments,
        start=1,
    ):
        _log_experiment_configuration(
            index=index,
            total_experiments=total_experiments,
            config=config,
        )

        try:
            workflow = create_workflow(
                config,
            )

            if config.time_series.enabled:
                _execute_time_series_experiment(
                    index=index,
                    total_experiments=total_experiments,
                    config=config,
                    workflow=workflow,
                    exporter=exporter,
                    writer=writer,
                    time_series_writer=time_series_writer,
                    differential_time_series_writer=(
                        differential_time_series_writer
                    ),
                )

            else:
                _execute_single_observation_experiment(
                    index=index,
                    total_experiments=total_experiments,
                    config=config,
                    workflow=workflow,
                    exporter=exporter,
                    writer=writer,
                )

            successful_experiments += 1

        except Exception:
            failed_experiments += 1

            logger.exception(
                "Experiment failed for target '%s'.",
                config.target.target_name,
            )

    logger.info("")
    logger.info("=" * 72)
    logger.info(
        "PHOTOMETRIC ANALYSIS EXPERIMENT SUITE COMPLETED",
    )
    logger.info("=" * 72)

    logger.info(
        "Active experiment set : %s",
        ACTIVE_EXPERIMENT_SET.value,
    )

    logger.info(
        "Total experiments      : %d",
        total_experiments,
    )

    logger.info(
        "Successful experiments : %d",
        successful_experiments,
    )

    logger.info(
        "Failed experiments     : %d",
        failed_experiments,
    )

    logger.info(
        "Standard results       : %s",
        writer.output_directory,
    )

    logger.info(
        "Time-series results    : %s",
        time_series_writer.output_path,
    )

    logger.info(
        "Differential results   : %s",
        differential_time_series_writer.output_path,
    )

    logger.info(
        "Post-experiment analysis remains a separate processing stage.",
    )

    logger.info("=" * 72)

    # A production-suite invocation must not report a successful process exit
    # when one or more configured experiments failed. We still attempt every
    # configured experiment above so a single failure does not hide failures
    # or results from subsequent experiments.
    if failed_experiments:
        raise RuntimeError(
            (
                "Photometric experiment suite completed with "
                f"{failed_experiments} failed experiment(s) out of "
                f"{total_experiments}. Review the logged exceptions before "
                "using the exported results for analysis."
            ),
        )


# ============================================================================
# Single-observation execution
# ============================================================================


def _execute_single_observation_experiment(
    *,
    index: int,
    total_experiments: int,
    config: ExperimentConfig,
    workflow: PhotometricWorkflow,
    exporter: ExperimentExporter,
    writer: ExportWriter,
) -> None:
    """
    Execute and persist a single-observation experiment.

    No scientific processing is implemented here. The function only
    orchestrates production workflow execution and export.
    """
    logger.info(
        "Execution mode: single observation.",
    )

    result = workflow.run()

    _log_experiment_summary(
        result,
    )

    # FIELD_ANALYSIS intentionally stops after source detection and
    # crowding analysis. It does not produce photometric measurements,
    # evaluation metrics, or standard experiment/source exports.
    if isinstance(result, FieldAnalysisResult):
        logger.info(
            "FIELD_ANALYSIS does not require standard photometric exports.",
        )
        logger.info(
            "Experiment %d of %d completed successfully.",
            index,
            total_experiments,
        )
        return

    summary, source_records = exporter.export(
        config=config,
        result=result,
    )

    _log_export_summary(
        summary=summary,
        source_records=source_records,
    )

    writer.write(
        summary=summary,
        source_records=source_records,
    )

    logger.info(
        "Experiment %d of %d completed successfully.",
        index,
        total_experiments,
    )


# ============================================================================
# Time-series execution
# ============================================================================


def _execute_time_series_experiment(
    *,
    index: int,
    total_experiments: int,
    config: ExperimentConfig,
    workflow: PhotometricWorkflow,
    exporter: ExperimentExporter,
    writer: ExportWriter,
    time_series_writer: TimeSeriesWriter,
    differential_time_series_writer: DifferentialTimeSeriesWriter,
) -> None:
    """
    Execute and persist a cadence-level time-series experiment.

    The complete cadence-level series is persisted through
    ``TimeSeriesWriter``.

    Differential-photometry experiments additionally persist their complete
    cadence-level differential measurements through
    ``DifferentialTimeSeriesWriter``.

    The first processed cadence is also exported through the existing
    standard ``ExperimentExporter`` so that standard experiment/source
    outputs remain available.

    No scientific processing is performed by this function.
    """
    if config.survey.value != "TESS":
        raise ValueError(
            (
                "Time-series execution is currently supported only "
                "for TESS experiments."
            ),
        )

    cadence_indices = config.time_series.cadence_indices

    if cadence_indices is None:
        logger.info(
            "Execution mode: full TESS time series "
            "(all available cadences).",
        )

    else:
        logger.info(
            "Execution mode: TESS time series "
            "(%d configured cadences).",
            len(cadence_indices),
        )

        if cadence_indices:
            logger.info(
                "First configured cadence: %d.",
                cadence_indices[0],
            )

            logger.info(
                "Last configured cadence: %d.",
                cadence_indices[-1],
            )

    # ------------------------------------------------------------------
    # 1. Execute production time-series workflow.
    # ------------------------------------------------------------------

    results = workflow.run_time_series(
        cadence_indices=cadence_indices,
    )

    # ------------------------------------------------------------------
    # 2. Validate workflow output before persistence.
    # ------------------------------------------------------------------

    _validate_time_series_results(
        config=config,
        results=results,
    )

    # ------------------------------------------------------------------
    # 3. Report cadence-level results.
    # ------------------------------------------------------------------

    _log_time_series_summary(
        config=config,
        results=results,
    )

    # ------------------------------------------------------------------
    # 4. Persist complete cadence-level source measurements.
    # ------------------------------------------------------------------

    time_series_records = time_series_writer.write(
        config=config,
        results=results,
    )

    _validate_time_series_export(
        results=results,
        records=time_series_records,
    )

    logger.info(
        "Time-series source export completed successfully: %d record(s).",
        len(time_series_records),
    )

    # ------------------------------------------------------------------
    # 5. Persist differential cadence-level measurements when applicable.
    # ------------------------------------------------------------------

    if config.experiment_type is ExperimentType.DIFFERENTIAL_PHOTOMETRY:

        differential_records = (
            differential_time_series_writer.write(
                config=config,
                results=results,
            )
        )

        _validate_differential_time_series_export(
            results=results,
            records=differential_records,
        )

        logger.info(
            (
                "Differential time-series export completed successfully: "
                "%d record(s)."
            ),
            len(differential_records),
        )

    # ------------------------------------------------------------------
    # 6. Export representative standard experiment/source result.
    # ------------------------------------------------------------------

    representative_result = results[0]

    representative_cadence = (
        _extract_cadence_index_from_workflow_result(
            representative_result,
        )
    )

    logger.info(
        (
            "Creating standard experiment/source exports from "
            "representative cadence %d."
        ),
        representative_cadence,
    )

    summary, source_records = exporter.export(
        config=config,
        result=representative_result,
    )

    _log_export_summary(
        summary=summary,
        source_records=source_records,
    )

    writer.write(
        summary=summary,
        source_records=source_records,
    )

    logger.info(
        (
            "Standard experiment/source export completed successfully: "
            "1 experiment summary and %d source record(s)."
        ),
        len(source_records),
    )

    logger.info(
        "Experiment %d of %d completed successfully.",
        index,
        total_experiments,
    )


# ============================================================================
# Time-series validation
# ============================================================================


def _validate_time_series_results(
    *,
    config: ExperimentConfig,
    results: Sequence[WorkflowResult],
) -> None:
    """
    Validate the structural integrity of time-series workflow results.

    Validation is deliberately limited to orchestration-level invariants.
    Scientific values are validated by the workflow and evaluator.
    """
    result_sequence = tuple(results)

    if not result_sequence:
        raise ValueError(
            (
                "Time-series workflow returned no WorkflowResult "
                f"objects for target '{config.target.target_name}'."
            ),
        )

    requested_cadences = config.time_series.cadence_indices

    if requested_cadences is not None:
        expected_count = len(
            requested_cadences,
        )

        if len(result_sequence) != expected_count:
            raise ValueError(
                (
                    "Time-series result count mismatch: "
                    f"expected {expected_count}, "
                    f"received {len(result_sequence)}."
                ),
            )

    observation_times: list[float] = []

    for result_index, result in enumerate(
        result_sequence,
        start=1,
    ):
        if not isinstance(
            result,
            WorkflowResult,
        ):
            raise ValueError(
                (
                    "Time-series result at position "
                    f"{result_index} is not a WorkflowResult."
                ),
            )

        if result.evaluation is None:
            raise ValueError(
                (
                    "Time-series result at position "
                    f"{result_index} contains no evaluation."
                ),
            )

        if result.observation is None:
            raise ValueError(
                (
                    "Time-series result at position "
                    f"{result_index} contains no Observation."
                ),
            )

        observation_time = (
            result.observation.metadata.observation_time
        )

        if not isinstance(
            observation_time,
            (int, float),
        ) or isinstance(
            observation_time,
            bool,
        ):
            raise ValueError(
                (
                    "TESS time-series result at position "
                    f"{result_index} has a non-numeric observation "
                    f"time: {type(observation_time).__name__}."
                ),
            )

        observation_times.append(
            float(observation_time),
        )

    if observation_times != sorted(
        observation_times,
    ):
        raise ValueError(
            "TESS time-series observation times are not monotonically ordered.",
        )


def _validate_time_series_export(
    *,
    results: Sequence[WorkflowResult],
    records: Sequence[TimeSeriesMeasurementRecord],
) -> None:
    """
    Validate the retained persistent-source time-series export.

    TimeSeriesWriter intentionally exports only the persistent source
    population: sources that remain reliably identifiable across all selected
    cadences. Therefore the number of raw detections in WorkflowResult objects
    is not the expected number of exported records.

    Validation is instead based on the source scope actually retained by the
    writer. Every exported cadence for an experiment must contain the same
    persistent source population, with exactly one record per source/cadence.
    """
    result_sequence = tuple(results)
    record_sequence = tuple(records)

    if not record_sequence:
        raise ValueError(
            "TimeSeriesWriter returned no export records.",
        )

    # ------------------------------------------------------------------
    # 1. Every logical source/cadence record must be unique.
    # ------------------------------------------------------------------

    keys = [
        (
            record.experiment_id,
            record.cadence_index,
            record.source_identifier,
        )
        for record in record_sequence
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate logical time-series export keys detected.",
        )

    # ------------------------------------------------------------------
    # 2. Validate the persistent source scope independently for each
    #    experiment. This keeps the validator correct if the writer is
    #    ever used with more than one experiment in a single call.
    # ------------------------------------------------------------------

    experiment_ids = list(
        dict.fromkeys(
            record.experiment_id
            for record in record_sequence
        ),
    )

    for experiment_id in experiment_ids:
        experiment_records = tuple(
            record
            for record in record_sequence
            if record.experiment_id == experiment_id
        )

        cadence_indices = list(
            dict.fromkeys(
                record.cadence_index
                for record in experiment_records
            ),
        )

        if not cadence_indices:
            raise ValueError(
                (
                    "Time-series export contains no cadence records for "
                    f"experiment {experiment_id!r}."
                ),
            )

        source_scope_by_cadence = {
            cadence_index: {
                record.source_identifier
                for record in experiment_records
                if record.cadence_index == cadence_index
            }
            for cadence_index in cadence_indices
        }

        reference_cadence = cadence_indices[0]
        reference_scope = source_scope_by_cadence[reference_cadence]

        for cadence_index, source_scope in source_scope_by_cadence.items():
            if source_scope != reference_scope:
                missing_sources = sorted(
                    reference_scope - source_scope,
                )
                unexpected_sources = sorted(
                    source_scope - reference_scope,
                )

                raise ValueError(
                    (
                        "Time-series export source scope is inconsistent "
                        f"for experiment {experiment_id!r} at cadence "
                        f"{cadence_index}: "
                        f"expected {len(reference_scope)} retained source(s), "
                        f"received {len(source_scope)}. "
                        f"Missing={missing_sources}, "
                        f"unexpected={unexpected_sources}."
                    ),
                )

        expected_record_count = (
            len(cadence_indices) * len(reference_scope)
        )

        if len(experiment_records) != expected_record_count:
            raise ValueError(
                (
                    "Time-series export record count mismatch for "
                    f"experiment {experiment_id!r}: expected "
                    f"{expected_record_count} record(s) from "
                    f"{len(cadence_indices)} cadence(s) × "
                    f"{len(reference_scope)} retained persistent source(s), "
                    f"received {len(experiment_records)}."
                ),
            )

    # ------------------------------------------------------------------
    # 3. The writer must export one cadence record for each workflow
    #    result. This verifies that source dropout has not accidentally
    #    removed an entire cadence.
    # ------------------------------------------------------------------

    if result_sequence:
        result_cadences = [
            _extract_cadence_index_from_workflow_result(
                result,
            )
            for result in result_sequence
        ]

        exported_cadences = [
            record.cadence_index
            for record in record_sequence
        ]

        unique_result_cadences = list(
            dict.fromkeys(result_cadences)
        )
        unique_exported_cadences = list(
            dict.fromkeys(exported_cadences)
        )

        if unique_result_cadences != unique_exported_cadences:
            raise ValueError(
                (
                    "Time-series export cadence coverage does not match "
                    "workflow result cadence coverage: "
                    f"expected {unique_result_cadences}, "
                    f"received {unique_exported_cadences}."
                ),
            )


def _validate_differential_time_series_export(
    *,
    results: Sequence[WorkflowResult],
    records: Sequence[DifferentialTimeSeriesMeasurementRecord],
) -> None:
    """
    Validate the relationship between differential workflow results and
    differential cadence-level export records.

    Differential photometry produces one aggregate measurement per processed
    cadence rather than one record per detected source.
    """
    result_sequence = tuple(results)
    record_sequence = tuple(records)

    if not record_sequence:
        raise ValueError(
            "DifferentialTimeSeriesWriter returned no export records.",
        )

    expected_record_count = len(
        result_sequence,
    )

    if len(record_sequence) != expected_record_count:
        raise ValueError(
            (
                "Differential time-series export record count mismatch: "
                f"expected {expected_record_count}, "
                f"received {len(record_sequence)}."
            ),
        )

    keys = [
        (
            record.experiment_id,
            record.cadence_index,
        )
        for record in record_sequence
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate logical differential time-series export keys detected.",
        )

    result_cadences = [
        _extract_cadence_index_from_workflow_result(
            result,
        )
        for result in result_sequence
    ]

    record_cadences = [
        record.cadence_index
        for record in record_sequence
    ]

    if result_cadences != record_cadences:
        raise ValueError(
            (
                "Differential time-series export cadence ordering does not "
                "match the workflow result ordering."
            ),
        )


def _extract_cadence_index_from_workflow_result(
    result: WorkflowResult,
) -> int:
    """
    Extract a cadence index from a workflow result.

    This reads existing raw-image provenance only. It performs no scientific
    processing.
    """
    if result.observation is None:
        raise ValueError(
            "Cannot validate cadence provenance without an Observation.",
        )

    try:
        raw_image = result.observation.images["raw"].image

    except KeyError as exc:
        raise ValueError(
            (
                "Cannot validate cadence provenance because the raw "
                "image product is missing."
            ),
        ) from exc

    metadata = raw_image.meta

    if metadata is None or "cadence_index" not in metadata:
        raise ValueError(
            (
                "Cannot validate cadence provenance because "
                "'cadence_index' is missing from raw image metadata."
            ),
        )

    cadence_index = metadata["cadence_index"]

    if (
        isinstance(cadence_index, bool)
        or not isinstance(cadence_index, int)
        or cadence_index < 0
    ):
        raise ValueError(
            (
                "WorkflowResult raw-image cadence_index must be a "
                f"non-negative integer, received {cadence_index!r}."
            ),
        )

    return cadence_index


# ============================================================================
# Logging
# ============================================================================


def _log_experiment_configuration(
    *,
    index: int,
    total_experiments: int,
    config: ExperimentConfig,
) -> None:
    """
    Log the configuration of an experiment before execution.

    This function reports configuration only and performs no scientific
    processing.
    """
    logger.info("")
    logger.info("=" * 72)
    logger.info(
        "Experiment %d of %d",
        index,
        total_experiments,
    )
    logger.info("=" * 72)

    logger.info(
        "Survey       : %s",
        config.survey.value,
    )

    logger.info(
        "Target       : %s",
        config.target.target_name,
    )

    logger.info(
        "Experiment   : %s",
        config.experiment_type.value,
    )

    logger.info(
        "Pipeline     : %s",
        config.pipeline.value,
    )

    logger.info(
        "Time series  : %s",
        "enabled"
        if config.time_series.enabled
        else "disabled",
    )

    if config.time_series.enabled:
        cadence_indices = config.time_series.cadence_indices

        if cadence_indices is None:
            logger.info(
                "Cadence scope: all available cadences.",
            )

        else:
            logger.info(
                "Cadence count: %d.",
                len(cadence_indices),
            )

            if cadence_indices:
                logger.info(
                    "Cadence range: %d -> %d.",
                    cadence_indices[0],
                    cadence_indices[-1],
                )

    if config.experiment_type is ExperimentType.DIFFERENTIAL_PHOTOMETRY:
        differential_config = config.differential_photometry

        if differential_config is not None:
            logger.info(
                "Reference stars: %d.",
                len(differential_config.reference_stars),
            )


def _log_experiment_summary(
    result: WorkflowResult | FieldAnalysisResult,
) -> None:
    """
    Log the high-level result of a completed workflow.

    Evaluation metrics have already been calculated by the evaluator.
    This function only reports them.

    ``FIELD_ANALYSIS`` is intentionally handled separately because it
    contains no photometric evaluation. Its result is a field-level
    source catalogue plus the already-computed neighbour counts.
    """
    if isinstance(result, FieldAnalysisResult):
        observation = result.observation
        detections = observation.detections

        detected_count = (
            len(detections.sources)
            if detections is not None
            else 0
        )

        neighbour_counts = result.neighbour_counts
        crowded_count = sum(
            1
            for count in neighbour_counts.values()
            if count > 0
        )

        logger.info("")
        logger.info("-" * 72)
        logger.info("FIELD ANALYSIS SUMMARY")
        logger.info("-" * 72)

        logger.info(
            "Field                  : %s",
            observation.metadata.target_name,
        )

        logger.info(
            "Detected sources       : %d",
            detected_count,
        )

        logger.info(
            "Neighbour-count entries: %d",
            len(neighbour_counts),
        )

        logger.info(
            "Crowded sources        : %d",
            crowded_count,
        )

        logger.info("-" * 72)
        return

    if result.evaluation is None:
        logger.warning(
            "Workflow completed without an evaluation result.",
        )
        return

    if result.observation is None:
        logger.warning(
            "Workflow completed without an Observation.",
        )
        return

    evaluation = result.evaluation

    logger.info("")
    logger.info("-" * 72)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("-" * 72)

    logger.info(
        "Target                  : %s",
        result.observation.metadata.target_name,
    )

    logger.info(
        "Algorithm               : %s",
        evaluation.algorithm,
    )

    logger.info(
        "Detected sources        : %d",
        evaluation.detected_source_count,
    )

    logger.info(
        "Measured sources        : %d",
        evaluation.measured_source_count,
    )

    logger.info(
        "Measurement completeness: %.2f%%",
        evaluation.measurement_completeness * 100.0,
    )

    logger.info(
        "Valid measurements      : %d",
        evaluation.valid_measurement_count,
    )

    logger.info(
        "Invalid measurements    : %d",
        evaluation.invalid_measurement_count,
    )

    logger.info(
        "Missing measurements    : %d",
        evaluation.missing_measurement_count,
    )

    logger.info(
        "Non-positive flux       : %d",
        evaluation.non_positive_flux_count,
    )

    logger.info(
        "Crowded sources         : %d",
        evaluation.crowded_source_count,
    )

    logger.info(
        "Crowding fraction       : %.2f%%",
        evaluation.crowding_fraction * 100.0,
    )

    logger.info("-" * 72)


def _log_time_series_summary(
    *,
    config: ExperimentConfig,
    results: Sequence[WorkflowResult],
) -> None:
    """
    Log an aggregate summary of a completed time-series experiment.

    No new scientific quantities are calculated. The function only
    aggregates already-evaluated workflow metadata for operational reporting.
    """
    result_sequence = tuple(results)

    cadence_count = len(
        result_sequence,
    )

    total_detected = 0
    total_measured = 0
    total_valid = 0
    total_invalid = 0

    observation_times: list[float] = []

    for result in result_sequence:

        evaluation = result.evaluation

        if evaluation is None:
            continue

        total_detected += int(
            evaluation.detected_source_count,
        )

        total_measured += int(
            evaluation.measured_source_count,
        )

        total_valid += int(
            evaluation.valid_measurement_count,
        )

        total_invalid += int(
            evaluation.invalid_measurement_count,
        )

        if result.observation is None:
            continue

        observation_time = (
            result.observation.metadata.observation_time
        )

        if isinstance(
            observation_time,
            (int, float),
        ) and not isinstance(
            observation_time,
            bool,
        ):
            observation_times.append(
                float(observation_time),
            )

    logger.info("")
    logger.info("-" * 72)
    logger.info("TIME-SERIES SUMMARY")
    logger.info("-" * 72)

    logger.info(
        "Target                    : %s",
        config.target.target_name,
    )

    logger.info(
        "Cadences processed        : %d",
        cadence_count,
    )

    logger.info(
        "Total detected sources    : %d",
        total_detected,
    )

    logger.info(
        "Total measured sources    : %d",
        total_measured,
    )

    logger.info(
        "Total valid measurements  : %d",
        total_valid,
    )

    logger.info(
        "Total invalid measurements: %d",
        total_invalid,
    )

    if observation_times:
        logger.info(
            "First BTJD                : %.12f",
            observation_times[0],
        )

        logger.info(
            "Last BTJD                 : %.12f",
            observation_times[-1],
        )

    logger.info("-" * 72)


def _log_export_summary(
    *,
    summary: ExperimentSummary,
    source_records: tuple[SourceMeasurementRecord, ...],
) -> None:
    """
    Log the structured export produced for a completed experiment.

    No scientific calculations or persistence occur here.
    """
    target_source = summary.target_source_identifier

    logger.info("")
    logger.info("-" * 72)
    logger.info("EXPORT SUMMARY")
    logger.info("-" * 72)

    logger.info(
        "Experiment ID          : %s",
        summary.experiment_id,
    )

    logger.info(
        "Survey                 : %s",
        summary.survey,
    )

    logger.info(
        "Target                 : %s",
        summary.target_name,
    )

    logger.info(
        "Experiment type        : %s",
        summary.experiment_type,
    )

    logger.info(
        "Pipeline               : %s",
        summary.pipeline,
    )

    logger.info(
        "Detected sources       : %d",
        summary.detected_source_count,
    )

    logger.info(
        "Source records         : %d",
        len(source_records),
    )

    if target_source is not None:

        logger.info(
            "Target source          : %s",
            target_source,
        )

        if summary.target_source_separation_pixels is not None:
            logger.info(
                "Target separation      : %.3f pixels",
                summary.target_source_separation_pixels,
            )

    else:
        logger.info(
            "Target source          : None",
        )

    logger.info("-" * 72)


# ============================================================================
# Script entry point
# ============================================================================


if __name__ == "__main__":
    main()