"""
Photometry experiment result exporter.

Transforms completed photometric workflow results into immutable,
survey-independent export records.

This module is deliberately limited to the transformation boundary:

    ExperimentConfig + WorkflowResult
                |
                v
        ExperimentSummary
        SourceMeasurementRecord

No scientific calculations are performed here.

Scientific quantities are calculated by the photometry and evaluation
stages and are copied into the export records without modification.

No CSV, JSON or other file I/O is performed by this module. Persistence
is intentionally kept separate so that the same export records can later
support multiple output formats.

Time-series export is intentionally handled separately. This exporter
remains responsible for one completed Observation / experiment result.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

from core.models import ExperimentConfig
from domain.models import (
    DetectedSource,
    Observation,
    PhotometryMeasurement,
)
from evaluation.models import SourceEvaluation
from export.models import (
    ExperimentSummary,
    SourceMeasurementRecord,
)
from utils.logger import get_logger
from workflows.models import WorkflowResult


class ExperimentExporter:
    """
    Transform completed single-observation workflow results into
    immutable export records.

    The exporter is survey-independent. Survey-specific values are read
    from the common Observation provenance and experiment configuration
    models rather than from TESS, SDSS or other acquisition libraries.

    Parameters
    ----------
    logger
        Optional logger used for export diagnostics. If omitted, the
        module-specific project logger is created automatically.

    Notes
    -----
    The exporter does not modify the supplied Observation or
    WorkflowResult.

    The exporter does not perform scientific calculations. Evaluation
    metrics are copied directly from PhotometryEvaluation and source
    quality classifications are copied from SourceEvaluation.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the experiment exporter.

        Parameters
        ----------
        logger
            Optional logger used for diagnostic output.
        """

        self._logger = logger or get_logger(
            __name__,
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def export(
        self,
        *,
        config: ExperimentConfig,
        result: WorkflowResult,
    ) -> tuple[
        ExperimentSummary,
        tuple[SourceMeasurementRecord, ...],
    ]:
        """
        Transform one completed workflow result into export records.

        Parameters
        ----------
        config
            Immutable experiment configuration used to execute the
            workflow.

        result
            Completed workflow result containing the Observation,
            photometric evaluation and optional target-source match.

        Returns
        -------
        tuple[ExperimentSummary, tuple[SourceMeasurementRecord, ...]]
            Experiment-level summary followed by one source-level record
            for each LocalField source.

        Raises
        ------
        ValueError
            If the configuration or workflow result is missing required
            information or if the workflow is incomplete.
        """

        self._validate_inputs(
            config=config,
            result=result,
        )

        observation = result.observation

        if observation.detections is None:
            raise ValueError(
                "Cannot export workflow result without completed "
                "detection results.",
            )

        if observation.photometry is None:
            raise ValueError(
                "Cannot export workflow result without completed "
                "photometry results.",
            )

        if result.evaluation is None:
            raise ValueError(
                "Cannot export workflow result without completed "
                "evaluation results.",
            )

        detections = observation.detections
        photometry = observation.photometry
        evaluation = result.evaluation

        experiment_id = self._build_experiment_id(
            config=config,
            observation=observation,
        )

        self._logger.info(
            "Starting experiment result export for '%s'.",
            observation.metadata.target_name,
        )

        self._logger.info(
            "Experiment ID: %s.",
            experiment_id,
        )

        self._logger.info(
            "Detected sources available for export: %d.",
            len(detections.sources),
        )

        self._logger.info(
            "Photometric measurements available for export: %d.",
            len(photometry.measurements),
        )

        self._logger.info(
            "Source evaluations available for export: %d.",
            len(evaluation.source_evaluations),
        )

        summary = self._build_summary(
            experiment_id=experiment_id,
            config=config,
            result=result,
        )

        source_records = self._build_source_measurements(
            experiment_id=experiment_id,
            config=config,
            result=result,
        )

        self._validate_export_consistency(
            result=result,
            summary=summary,
            source_records=source_records,
        )

        self._logger.info(
            "Experiment summary created successfully.",
        )

        self._logger.info(
            "Source measurement records created: %d.",
            len(source_records),
        )

        self._logger.info(
            (
                "Experiment result export transformation completed "
                "successfully for '%s'."
            ),
            observation.metadata.target_name,
        )

        return (
            summary,
            tuple(source_records),
        )

    # ========================================================================
    # Experiment-level export
    # ========================================================================

    def _build_summary(
        self,
        *,
        experiment_id: str,
        config: ExperimentConfig,
        result: WorkflowResult,
    ) -> ExperimentSummary:
        """
        Build the experiment-level export summary.

        No scientific values are calculated here. All evaluation metrics
        are copied directly from the completed PhotometryEvaluation.
        """

        observation = result.observation
        evaluation = result.evaluation

        if observation.detections is None:
            raise ValueError(
                "Cannot build experiment summary without completed "
                "detection results.",
            )

        if observation.photometry is None:
            raise ValueError(
                "Cannot build experiment summary without completed "
                "photometry results.",
            )

        if evaluation is None:
            raise ValueError(
                "Cannot build experiment summary without completed "
                "evaluation results.",
            )

        detections = observation.detections
        photometry = observation.photometry
        target_match = result.target_match

        return ExperimentSummary(
            experiment_id=experiment_id,
            target_name=observation.metadata.target_name,
            survey=observation.provenance.survey.value,
            experiment_type=config.experiment_type.value,
            pipeline=config.pipeline.value,
            observation_time=observation.metadata.observation_time,
            observation_time_scale=(
                observation.metadata.observation_time_scale
            ),
            observation_time_format=(
                observation.metadata.observation_time_format
            ),
            instrument=observation.metadata.instrument,
            filter_name=observation.metadata.filter_name,
            exposure_time=observation.metadata.exposure_time,
            dataset_identifier=(
                observation.provenance.dataset_identifier
            ),
            # Preserve the workflow evaluation population count. The
            # source-level export population is separately defined by
            # result.local_field and is intentionally not inferred from
            # the restored global detection catalogue.
            detected_source_count=evaluation.detected_source_count,
            detection_algorithm=detections.algorithm,
            detection_threshold=detections.detection_threshold,
            target_source_identifier=(
                target_match.source_identifier
                if target_match is not None
                else None
            ),
            target_x=(
                target_match.target_x
                if target_match is not None
                else None
            ),
            target_y=(
                target_match.target_y
                if target_match is not None
                else None
            ),
            source_x=(
                target_match.source_x
                if target_match is not None
                else None
            ),
            source_y=(
                target_match.source_y
                if target_match is not None
                else None
            ),
            target_source_separation_pixels=(
                target_match.separation_pixels
                if target_match is not None
                else None
            ),
            photometry_algorithm=photometry.algorithm,
            aperture_radius=photometry.aperture_radius,
            annulus_inner_radius=(
                photometry.annulus_inner_radius
            ),
            annulus_outer_radius=(
                photometry.annulus_outer_radius
            ),
            measured_source_count=evaluation.measured_source_count,
            valid_measurement_count=(
                evaluation.valid_measurement_count
            ),
            measurement_completeness=(
                evaluation.measurement_completeness
            ),
            valid_measurement_fraction=(
                evaluation.valid_measurement_fraction
            ),
            invalid_measurement_count=(
                evaluation.invalid_measurement_count
            ),
            missing_measurement_count=(
                evaluation.missing_measurement_count
            ),
            non_positive_flux_count=(
                evaluation.non_positive_flux_count
            ),
            crowded_source_count=(
                evaluation.crowded_source_count
            ),
            crowding_fraction=(
                evaluation.crowding_fraction
            ),
            crowding_radius=(
                evaluation.crowding_radius
            ),
            median_net_flux=(
                evaluation.median_net_flux
            ),
            median_background_level=(
                evaluation.median_background_level
            ),
            median_background_fraction=(
                evaluation.median_background_fraction
            ),
            evaluation_algorithm=evaluation.algorithm,
            evaluation_timestamp=evaluation.evaluation_timestamp,
        )

    # ========================================================================
    # Source-level export
    # ========================================================================

    def _build_source_measurements(
        self,
        *,
        experiment_id: str,
        config: ExperimentConfig,
        result: WorkflowResult,
    ) -> list[SourceMeasurementRecord]:
        """
        Build source-level export records for the authoritative LocalField.

        The returned workflow observation intentionally restores the complete
        detection catalogue for provenance and diagnostics. Consequently,
        ``observation.detections.sources`` must not be used as the downstream
        experimental population here.

        The LocalField stored on WorkflowResult is the authoritative source
        population for baseline/SCARLET photometry and evaluation. Export
        completeness is therefore validated against ``result.local_field``,
        not against the global detection catalogue.
        """

        observation = result.observation

        if observation.detections is None:
            raise ValueError(
                "Cannot build source measurement records without "
                "completed detection results.",
            )

        if observation.photometry is None:
            raise ValueError(
                "Cannot build source measurement records without "
                "completed photometry results.",
            )

        if result.evaluation is None:
            raise ValueError(
                "Cannot build source measurement records without "
                "completed evaluation results.",
            )

        local_field = result.local_field
        if local_field is None:
            raise ValueError(
                "Cannot build source measurement records without an "
                "authoritative LocalField.",
            )

        photometry = observation.photometry
        evaluation = result.evaluation

        # The LocalField is the only source population permitted to cross
        # the workflow -> export boundary for source-level photometry.
        export_sources = tuple(local_field.sources)

        if not export_sources:
            raise ValueError(
                "Cannot build source measurement records because the "
                "authoritative LocalField is empty.",
            )

        measurements_by_source = self._index_measurements(
            photometry.measurements,
        )

        evaluations_by_source = self._index_source_evaluations(
            evaluation.source_evaluations,
        )

        self._validate_evaluation_completeness(
            detected_sources=export_sources,
            evaluation_by_source=evaluations_by_source,
        )

        target_source_identifier = (
            result.target_match.source_identifier
            if result.target_match is not None
            else None
        )

        records: list[SourceMeasurementRecord] = []

        for source in export_sources:
            measurement = measurements_by_source.get(
                source.identifier,
            )

            source_evaluation = evaluations_by_source.get(
                source.identifier,
            )

            if source_evaluation is None:
                raise ValueError(
                    (
                        "Photometry evaluation is missing an evaluation "
                        "record for LocalField source "
                        f"'{source.identifier}'."
                    ),
                )

            records.append(
                self._create_source_measurement_record(
                    experiment_id=experiment_id,
                    config=config,
                    source=source,
                    measurement=measurement,
                    quality_flag=source_evaluation.quality_flag,
                    target_source_identifier=(
                        target_source_identifier
                    ),
                ),
            )

        export_source_identifiers = {
            source.identifier
            for source in export_sources
        }

        orphan_measurement_identifiers = (
            set(measurements_by_source)
            - export_source_identifiers
        )

        if orphan_measurement_identifiers:
            self._logger.warning(
                (
                    "Ignoring %d orphan photometric measurement "
                    "record(s) because they do not correspond to the "
                    "authoritative LocalField."
                ),
                len(orphan_measurement_identifiers),
            )

        orphan_evaluation_identifiers = (
            set(evaluations_by_source)
            - export_source_identifiers
        )

        if orphan_evaluation_identifiers:
            self._logger.warning(
                (
                    "Ignoring %d orphan source evaluation record(s) "
                    "because they do not correspond to the "
                    "authoritative LocalField."
                ),
                len(orphan_evaluation_identifiers),
            )

        return records

    def _create_source_measurement_record(
        self,
        *,
        experiment_id: str,
        config: ExperimentConfig,
        source: DetectedSource,
        measurement: PhotometryMeasurement | None,
        quality_flag: str,
        target_source_identifier: str | None,
    ) -> SourceMeasurementRecord:
        """
        Convert one LocalField source and its optional measurement into
        an immutable export record.
        """

        if measurement is not None:
            if (
                measurement.source_identifier
                != source.identifier
            ):
                raise ValueError(
                    (
                        "Photometric measurement source identifier "
                        "does not match the LocalField source identifier."
                    ),
                )

        return SourceMeasurementRecord(
            experiment_id=experiment_id,
            target_name=config.target.target_name,
            survey=config.survey.value,
            experiment_type=config.experiment_type.value,
            pipeline=config.pipeline.value,
            source_identifier=source.identifier,
            is_target_source=(
                source.identifier
                == target_source_identifier
            ),
            x_centroid=source.x_centroid,
            y_centroid=source.y_centroid,
            peak_value=source.peak_value,
            aperture_flux=(
                measurement.aperture_flux
                if measurement is not None
                else None
            ),
            background_level=(
                measurement.background_level
                if measurement is not None
                else None
            ),
            background_flux=(
                measurement.background_flux
                if measurement is not None
                else None
            ),
            net_flux=(
                measurement.net_flux
                if measurement is not None
                else None
            ),
            instrumental_magnitude=(
                measurement.instrumental_magnitude
                if measurement is not None
                else None
            ),
            aperture_area=(
                measurement.aperture_area
                if measurement is not None
                else None
            ),
            background_area=(
                measurement.background_area
                if measurement is not None
                else None
            ),
            aperture_coverage=(
                measurement.aperture_coverage
                if measurement is not None
                else None
            ),
            annulus_coverage=(
                measurement.annulus_coverage
                if measurement is not None
                else None
            ),
            background_fraction=(
                measurement.background_fraction
                if measurement is not None
                else None
            ),
            valid=(
                measurement.valid
                if measurement is not None
                else False
            ),
            quality_flag=quality_flag,
        )

    # ========================================================================
    # Indexing and consistency validation
    # ========================================================================

    @staticmethod
    def _index_measurements(
        measurements: Sequence[PhotometryMeasurement],
    ) -> dict[str, PhotometryMeasurement]:
        """
        Index photometric measurements by source identifier.

        Raises
        ------
        ValueError
            If duplicate or empty source identifiers are present.
        """

        indexed: dict[str, PhotometryMeasurement] = {}

        for measurement in measurements:
            identifier = measurement.source_identifier

            if not identifier:
                raise ValueError(
                    (
                        "Photometric measurement contains an empty "
                        "source identifier."
                    ),
                )

            if identifier in indexed:
                raise ValueError(
                    (
                        "Duplicate photometric measurement found for "
                        f"source '{identifier}'."
                    ),
                )

            indexed[identifier] = measurement

        return indexed

    @staticmethod
    def _index_source_evaluations(
        evaluations: Sequence[SourceEvaluation],
    ) -> dict[str, SourceEvaluation]:
        """
        Index source evaluations by source identifier.

        Raises
        ------
        ValueError
            If duplicate or empty source identifiers are present.
        """

        indexed: dict[str, SourceEvaluation] = {}

        for evaluation in evaluations:
            identifier = evaluation.source_identifier

            if not identifier:
                raise ValueError(
                    (
                        "Photometry evaluation contains a source "
                        "evaluation with an empty source identifier."
                    ),
                )

            if identifier in indexed:
                raise ValueError(
                    (
                        "Photometry evaluation contains duplicate "
                        f"source identifier '{identifier}'."
                    ),
                )

            indexed[identifier] = evaluation

        return indexed

    @staticmethod
    def _validate_evaluation_completeness(
        *,
        detected_sources: Sequence[DetectedSource],
        evaluation_by_source: dict[str, SourceEvaluation],
    ) -> None:
        """
        Verify that source evaluations correspond exactly to detected
        sources.

        The production evaluator is expected to produce exactly one
        SourceEvaluation for every LocalField source.
        """

        detected_identifiers = {
            source.identifier
            for source in detected_sources
        }

        evaluation_identifiers = set(
            evaluation_by_source,
        )

        missing_identifiers = (
            detected_identifiers
            - evaluation_identifiers
        )

        if missing_identifiers:
            identifiers = ", ".join(
                sorted(missing_identifiers),
            )

            raise ValueError(
                (
                    "PhotometryEvaluation is incomplete. "
                    "No source evaluation exists for detected "
                    f"source(s): {identifiers}."
                ),
            )

        orphan_identifiers = (
            evaluation_identifiers
            - detected_identifiers
        )

        if orphan_identifiers:
            identifiers = ", ".join(
                sorted(orphan_identifiers),
            )

            raise ValueError(
                (
                    "PhotometryEvaluation contains source evaluation "
                    "record(s) that do not correspond to detected "
                    f"source(s): {identifiers}."
                ),
            )

    # ========================================================================
    # Export consistency
    # ========================================================================

    @staticmethod
    def _validate_export_consistency(
        *,
        result: WorkflowResult,
        summary: ExperimentSummary,
        source_records: Sequence[SourceMeasurementRecord],
    ) -> None:
        """
        Verify that transformed export records remain internally
        consistent with the completed workflow result.

        The full detection catalogue is retained on the returned Observation
        for provenance. Source-level export consistency is therefore checked
        against the authoritative LocalField stored on WorkflowResult.
        """

        evaluation = result.evaluation
        local_field = result.local_field

        if evaluation is None:
            raise ValueError(
                "Cannot validate export consistency without evaluation.",
            )

        if local_field is None:
            raise ValueError(
                "Cannot validate export consistency without an "
                "authoritative LocalField.",
            )

        local_field_identifiers = {
            source.identifier
            for source in local_field.sources
        }

        record_identifiers = {
            record.source_identifier
            for record in source_records
        }

        if len(record_identifiers) != len(source_records):
            raise ValueError(
                "Export contains duplicate source measurement records.",
            )

        if record_identifiers != local_field_identifiers:
            missing = sorted(
                local_field_identifiers - record_identifiers,
            )
            orphan = sorted(
                record_identifiers - local_field_identifiers,
            )

            raise ValueError(
                (
                    "Export source population does not match the "
                    "authoritative LocalField: "
                    f"missing={missing}, orphan={orphan}."
                ),
            )

        expected_count = len(local_field.sources)

        if len(source_records) != expected_count:
            raise ValueError(
                (
                    "Export source-record count does not match the "
                    "authoritative LocalField: "
                    f"records={len(source_records)}, "
                    f"local_field={expected_count}."
                ),
            )

        if summary.measured_source_count != evaluation.measured_source_count:
            raise ValueError(
                (
                    "Export measured-source count does not match the "
                    "workflow evaluation: "
                    f"summary={summary.measured_source_count}, "
                    f"evaluation={evaluation.measured_source_count}."
                ),
            )


    # ========================================================================
    # Deterministic experiment identity
    # ========================================================================

    @staticmethod
    def _build_experiment_id(
        *,
        config: ExperimentConfig,
        observation: Observation,
    ) -> str:
        """
        Build a deterministic identifier for an experiment result.

        The identifier is derived from stable experiment and dataset
        provenance rather than from the current wall-clock time.

        Repeated exports of the same workflow result therefore retain
        the same identifier while different datasets and processing
        configurations remain distinguishable.

        Source-selection scope is intentionally not part of the identifier.
        The former ``source_scope`` configuration field was removed by the
        LocalField refactor. The exporter must not reconstruct or invent a
        replacement scope because LocalField selection is a workflow/runtime
        concern rather than an ExperimentConfig field.
        """

        components = (
            observation.provenance.survey.value,
            observation.provenance.dataset_identifier,
            config.target.target_name,
            config.experiment_type.name,
            config.pipeline.name,
        )

        canonical_value = "|".join(
            components,
        )

        digest = hashlib.sha256(
            canonical_value.encode("utf-8"),
        ).hexdigest()[
            :16
        ]

        return (
            f"{config.survey.value.lower()}-"
            f"{config.pipeline.name.lower()}-"
            f"{digest}"
        )

    # ========================================================================
    # Input validation
    # ========================================================================

    @staticmethod
    def _validate_inputs(
        *,
        config: ExperimentConfig,
        result: WorkflowResult,
    ) -> None:
        """
        Validate the minimum state required for export.

        Export is only permitted for a completed workflow containing
        source detection, photometry and evaluation results.
        """

        if config is None:
            raise ValueError(
                "Experiment configuration cannot be None.",
            )

        if result is None:
            raise ValueError(
                "Workflow result cannot be None.",
            )

        observation = result.observation

        if observation is None:
            raise ValueError(
                "Workflow result does not contain an Observation.",
            )

        if observation.detections is None:
            raise ValueError(
                (
                    "Cannot export workflow result without source "
                    "detection results."
                ),
            )

        if observation.photometry is None:
            raise ValueError(
                (
                    "Cannot export workflow result without photometry "
                    "results."
                ),
            )

        if not observation.processing_state.source_detection_completed:
            raise ValueError(
                (
                    "Cannot export workflow result because source "
                    "detection is not marked as completed."
                ),
            )

        if not observation.processing_state.photometry_completed:
            raise ValueError(
                (
                    "Cannot export workflow result because photometry "
                    "is not marked as completed."
                ),
            )

        if result.evaluation is None:
            raise ValueError(
                (
                    "Cannot export workflow result without photometry "
                    "evaluation."
                ),
            )

        if not observation.provenance.dataset_identifier:
            raise ValueError(
                (
                    "Cannot export workflow result without a dataset "
                    "identifier."
                ),
            )

        if not observation.metadata.target_name:
            raise ValueError(
                "Cannot export workflow result without a target name.",
            )