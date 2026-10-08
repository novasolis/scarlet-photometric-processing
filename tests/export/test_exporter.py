"""
Tests for the photometric experiment exporter.

The exporter is responsible for transforming a completed WorkflowResult
into immutable experiment-level and source-level export records.

These tests intentionally do not test scientific calculations. Scientific
quantities are produced by the photometry and evaluation stages and must
be copied into the export layer without modification.

The tests therefore focus on:

* export completeness;
* provenance preservation;
* target-source preservation;
* missing and invalid measurement handling;
* orphan measurement handling;
* duplicate detection;
* incomplete workflow rejection;
* deterministic experiment identifiers;
* preservation of source ordering;
* immutability/non-mutation of the workflow result.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pytest
from astropy.nddata import NDData

from core.models import (
    AcquisitionConfig,
    DetectionConfig,
    EvaluationConfig,
    ExperimentConfig,
    PhotometryConfig,
    PreprocessingConfig,
    RuntimeConfig,
    TargetConfig,
)
from domain.enums import (
    ExperimentType,
    ObservationSource,
    ProcessingPipeline,
    SourceScope,
    Survey,
)
from domain.models import (
    DetectionResult,
    DetectedSource,
    ImageData,
    Observation,
    ObservationMetadata,
    ObservationProvenance,
    PhotometryMeasurement,
    PhotometryResult,
    ProcessingState,
    TargetSourceMatch,
)
from evaluation.models import (
    PhotometryEvaluation,
    SourceEvaluation,
)
from export.exporter import ExperimentExporter
from workflows.models import WorkflowResult


OBSERVATION_TIME = datetime(
    2026,
    8,
    18,
    9,
    30,
    tzinfo=timezone.utc,
)

DOWNLOAD_TIME = datetime(
    2026,
    8,
    18,
    9,
    0,
    tzinfo=timezone.utc,
)

EVALUATION_TIME = datetime(
    2026,
    8,
    18,
    9,
    30,
    30,
    tzinfo=timezone.utc,
)

PHOTOMETRY_TIME = datetime(
    2026,
    8,
    18,
    9,
    30,
    15,
    tzinfo=timezone.utc,
)


def _create_config(
    *,
    experiment_type: ExperimentType = ExperimentType.TARGET_SOURCE,
    source_scope: SourceScope = SourceScope.TARGET,
    pipeline: ProcessingPipeline = ProcessingPipeline.BASELINE,
    target_name: str = "LHS 3844",
    survey: Survey = Survey.TESS,
) -> ExperimentConfig:
    """
    Create a complete deterministic experiment configuration.

    The exporter only consumes configuration/provenance information,
    so the scientific parameter values here are representative fixtures
    rather than test subjects in their own right.
    """

    return ExperimentConfig(
        experiment_type=experiment_type,
        source_scope=source_scope,
        survey=survey,
        target=TargetConfig(
            target_name=target_name,
            right_ascension=340.92941667,
            declination=-69.17210833,
        ),
        acquisition=AcquisitionConfig(
            sector=1,
            author="SPOC",
            cadence_seconds=120,
            cutout_size=15,
        ),
        preprocessing=PreprocessingConfig(
            background_box_size=32,
            background_filter_size=3,
            sigma_clip=3.0,
            exclude_percentile=10.0,
        ),
        detection=DetectionConfig(
            fwhm=2.5,
            detection_threshold_sigma=3.0,
        ),
        photometry=PhotometryConfig(
            aperture_radius=2.0,
            annulus_inner_radius=3.0,
            annulus_outer_radius=5.0,
            minimum_aperture_coverage=0.90,
            minimum_annulus_coverage=0.70,
            minimum_annulus_pixels=20,
            reference_star_count=10,
        ),
        evaluation=EvaluationConfig(
            matching_tolerance=2.0,
            crowding_radius=10.0,
        ),
        runtime=RuntimeConfig(
            save_results=False,
            generate_figures=False,
            show_figures=False,
            overwrite_existing=True,
            logging_level="INFO",
        ),
        pipeline=pipeline,
    )


def _create_source(
    identifier: str,
    *,
    x: float,
    y: float,
    peak: float,
) -> DetectedSource:
    """
    Create a detected-source fixture.
    """

    return DetectedSource(
        identifier=identifier,
        x_centroid=x,
        y_centroid=y,
        peak_value=peak,
    )


def _create_measurement(
    source_identifier: str,
    *,
    valid: bool = True,
    quality_flag: str = "VALID",
    aperture_flux: float | None = 1500.0,
    background_level: float | None = 25.0,
    background_flux: float | None = 100.0,
    net_flux: float | None = 1400.0,
    instrumental_magnitude: float | None = -7.865,
    aperture_area: float | None = 12.56,
    background_area: float | None = 50.27,
    aperture_coverage: float | None = 1.0,
    annulus_coverage: float | None = 1.0,
    background_fraction: float | None = 100.0 / 1500.0,
) -> PhotometryMeasurement:
    """
    Create a representative photometric measurement.
    """

    return PhotometryMeasurement(
        source_identifier=source_identifier,
        x_centroid=10.0,
        y_centroid=10.0,
        aperture_flux=aperture_flux,
        background_level=background_level,
        background_flux=background_flux,
        net_flux=net_flux,
        instrumental_magnitude=instrumental_magnitude,
        aperture_area=aperture_area,
        background_area=background_area,
        aperture_coverage=aperture_coverage,
        annulus_coverage=annulus_coverage,
        background_fraction=background_fraction,
        valid=valid,
        quality_flag=quality_flag,
    )


def _create_source_evaluation(
    source_identifier: str,
    *,
    valid: bool = True,
    quality_flag: str = "VALID",
    net_flux: float | None = 1400.0,
    instrumental_magnitude: float | None = -7.865,
    background_level: float | None = 25.0,
    background_fraction: float | None = 100.0 / 1500.0,
    aperture_coverage: float | None = 1.0,
    annulus_coverage: float | None = 1.0,
    neighbour_count: int = 0,
    crowded: bool = False,
) -> SourceEvaluation:
    """
    Create a representative source-level evaluation.
    """

    return SourceEvaluation(
        source_identifier=source_identifier,
        valid_measurement=valid,
        quality_flag=quality_flag,
        net_flux=net_flux,
        instrumental_magnitude=instrumental_magnitude,
        background_level=background_level,
        background_fraction=background_fraction,
        aperture_coverage=aperture_coverage,
        annulus_coverage=annulus_coverage,
        neighbour_count=neighbour_count,
        crowded=crowded,
    )


def _create_evaluation(
    source_evaluations: tuple[SourceEvaluation, ...],
    *,
    detected_source_count: int | None = None,
    measured_source_count: int | None = None,
    valid_measurement_count: int | None = None,
    measurement_completeness: float | None = None,
    valid_measurement_fraction: float | None = None,
    invalid_measurement_count: int | None = None,
    missing_measurement_count: int | None = None,
    non_positive_flux_count: int | None = None,
    crowded_source_count: int | None = None,
    crowding_fraction: float | None = None,
) -> PhotometryEvaluation:
    """
    Create a representative observation-level evaluation.

    Defaults are derived from the supplied source evaluations so that
    individual tests only need to override the metric under examination.
    """

    if detected_source_count is None:
        detected_source_count = len(source_evaluations)

    if measured_source_count is None:
        measured_source_count = sum(
            evaluation.quality_flag != "MISSING_MEASUREMENT"
            for evaluation in source_evaluations
        )

    if valid_measurement_count is None:
        valid_measurement_count = sum(
            evaluation.valid_measurement
            for evaluation in source_evaluations
        )

    if measurement_completeness is None:
        measurement_completeness = (
            measured_source_count / detected_source_count
            if detected_source_count > 0
            else 0.0
        )

    if valid_measurement_fraction is None:
        valid_measurement_fraction = (
            valid_measurement_count / measured_source_count
            if measured_source_count > 0
            else 0.0
        )

    if invalid_measurement_count is None:
        invalid_measurement_count = (
            measured_source_count - valid_measurement_count
        )

    if missing_measurement_count is None:
        missing_measurement_count = (
            detected_source_count - measured_source_count
        )

    if non_positive_flux_count is None:
        non_positive_flux_count = 0

    if crowded_source_count is None:
        crowded_source_count = sum(
            evaluation.crowded
            for evaluation in source_evaluations
        )

    if crowding_fraction is None:
        crowding_fraction = (
            crowded_source_count / detected_source_count
            if detected_source_count > 0
            else 0.0
        )

    valid_evaluations = tuple(
        evaluation
        for evaluation in source_evaluations
        if evaluation.valid_measurement
    )

    median_net_flux = (
        float(
            np.median(
                [
                    evaluation.net_flux
                    for evaluation in valid_evaluations
                    if evaluation.net_flux is not None
                ]
            )
        )
        if valid_evaluations
        else None
    )

    median_background_level = (
        float(
            np.median(
                [
                    evaluation.background_level
                    for evaluation in valid_evaluations
                    if evaluation.background_level is not None
                ]
            )
        )
        if valid_evaluations
        else None
    )

    median_background_fraction = (
        float(
            np.median(
                [
                    evaluation.background_fraction
                    for evaluation in valid_evaluations
                    if evaluation.background_fraction is not None
                ]
            )
        )
        if valid_evaluations
        else None
    )

    return PhotometryEvaluation(
        detected_source_count=detected_source_count,
        measured_source_count=measured_source_count,
        valid_measurement_count=valid_measurement_count,
        measurement_completeness=measurement_completeness,
        valid_measurement_fraction=valid_measurement_fraction,
        invalid_measurement_count=invalid_measurement_count,
        missing_measurement_count=missing_measurement_count,
        non_positive_flux_count=non_positive_flux_count,
        crowded_source_count=crowded_source_count,
        crowding_fraction=crowding_fraction,
        crowding_radius=10.0,
        median_net_flux=median_net_flux,
        median_background_level=median_background_level,
        median_background_fraction=median_background_fraction,
        algorithm="Photometry Evaluation",
        evaluation_timestamp=EVALUATION_TIME,
        source_evaluations=source_evaluations,
    )


def _create_observation(
    *,
    sources: tuple[DetectedSource, ...],
    measurements: tuple[PhotometryMeasurement, ...],
    target_name: str = "LHS 3844",
    survey: Survey = Survey.TESS,
    dataset_identifier: str = "TEST-DATASET-001",
    source_detection_completed: bool = True,
    photometry_completed: bool = True,
) -> Observation:
    """
    Create a completed Observation fixture.
    """

    metadata = ObservationMetadata(
        target_name=target_name,
        observation_time=OBSERVATION_TIME,
        right_ascension=340.92941667,
        declination=-69.17210833,
        instrument="TESS",
        filter_name="TESS",
        exposure_time=120.0,
        wcs=None,
    )

    provenance = ObservationProvenance(
        source=ObservationSource.SPACE_BASED,
        survey=survey,
        dataset_identifier=dataset_identifier,
        filename="test.fits",
        download_timestamp=DOWNLOAD_TIME,
    )

    image = ImageData(
        image=NDData(
            np.ones(
                (20, 20),
                dtype=float,
            )
        )
    )

    observation = Observation(
        metadata=metadata,
        provenance=provenance,
        original=image,
        active=image,
        images={
            "raw": image,
        },
        detections=DetectionResult(
            sources=sources,
            algorithm="DAOStarFinder",
            detection_threshold=3.0,
            detection_timestamp=OBSERVATION_TIME,
        ),
        photometry=PhotometryResult(
            measurements=measurements,
            algorithm="Classical Aperture Photometry",
            aperture_radius=2.0,
            annulus_inner_radius=3.0,
            annulus_outer_radius=5.0,
            photometry_timestamp=PHOTOMETRY_TIME,
        ),
        processing_state=ProcessingState(
            source_detection_completed=source_detection_completed,
            photometry_completed=photometry_completed,
        ),
    )

    return observation


def _create_result(
    *,
    sources: tuple[DetectedSource, ...],
    measurements: tuple[PhotometryMeasurement, ...],
    source_evaluations: tuple[SourceEvaluation, ...],
    target_match: TargetSourceMatch | None = None,
    config: ExperimentConfig | None = None,
    dataset_identifier: str = "TEST-DATASET-001",
) -> tuple[ExperimentConfig, WorkflowResult]:
    """
    Create a complete configuration/result pair.
    """

    if config is None:
        config = _create_config()

    observation = _create_observation(
        sources=sources,
        measurements=measurements,
        target_name=config.target.target_name,
        survey=config.survey,
        dataset_identifier=dataset_identifier,
    )

    evaluation = _create_evaluation(
        source_evaluations,
    )

    return (
        config,
        WorkflowResult(
            observation=observation,
            evaluation=evaluation,
            target_match=target_match,
        ),
    )


def test_complete_workflow_is_exported_successfully() -> None:
    """
    A completed workflow produces one summary and one record per
    detected source.
    """

    sources = (
        _create_source(
            "1",
            x=10.0,
            y=10.0,
            peak=2000.0,
        ),
        _create_source(
            "2",
            x=14.0,
            y=10.0,
            peak=1000.0,
        ),
    )

    measurements = (
        _create_measurement("1"),
        _create_measurement("2"),
    )

    evaluations = (
        _create_source_evaluation("1"),
        _create_source_evaluation("2"),
    )

    config, result = _create_result(
        sources=sources,
        measurements=measurements,
        source_evaluations=evaluations,
    )

    summary, records = ExperimentExporter().export(
        config=config,
        result=result,
    )

    assert summary is not None
    assert len(records) == 2
    assert [record.source_identifier for record in records] == [
        "1",
        "2",
    ]


def test_experiment_summary_preserves_workflow_provenance() -> None:
    """
    Experiment-level provenance and processing metadata are copied
    without modification.
    """

    config = _create_config(
        pipeline=ProcessingPipeline.SCARLET,
    )

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    config, result = _create_result(
        config=config,
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
        dataset_identifier="TESS-SECTOR-01-TEST",
    )

    summary, _ = ExperimentExporter().export(
        config=config,
        result=result,
    )

    values = asdict(summary)

    assert values["target_name"] == "LHS 3844"
    assert values["survey"] == "TESS"
    assert values["experiment_type"] == (
        ExperimentType.TARGET_SOURCE.value
    )
    assert values["source_scope"] == SourceScope.TARGET.value
    assert values["pipeline"] == ProcessingPipeline.SCARLET.value
    assert values["dataset_identifier"] == "TESS-SECTOR-01-TEST"
    assert values["observation_time"] == OBSERVATION_TIME
    assert values["instrument"] == "TESS"
    assert values["filter_name"] == "TESS"
    assert values["exposure_time"] == 120.0


def test_experiment_summary_preserves_detection_and_photometry_configuration() -> None:
    """
    Detection and aperture-photometry provenance are exported exactly.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    config, result = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
    )

    summary, _ = ExperimentExporter().export(
        config=config,
        result=result,
    )

    values = asdict(summary)

    assert values["detection_algorithm"] == "DAOStarFinder"
    assert values["detection_threshold"] == 3.0
    assert values["photometry_algorithm"] == (
        "Classical Aperture Photometry"
    )
    assert values["aperture_radius"] == 2.0
    assert values["annulus_inner_radius"] == 3.0
    assert values["annulus_outer_radius"] == 5.0


def test_experiment_summary_preserves_evaluation_metrics() -> None:
    """
    Evaluation statistics are copied directly into the export summary.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation_record = _create_source_evaluation("1")

    evaluation = _create_evaluation(
        (evaluation_record,),
        detected_source_count=1,
        measured_source_count=1,
        valid_measurement_count=1,
        measurement_completeness=1.0,
        valid_measurement_fraction=1.0,
        invalid_measurement_count=0,
        missing_measurement_count=0,
        non_positive_flux_count=0,
        crowded_source_count=1,
        crowding_fraction=1.0,
    )

    config, result = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation_record,),
    )

    result = WorkflowResult(
        observation=result.observation,
        evaluation=evaluation,
        target_match=None,
    )

    summary, _ = ExperimentExporter().export(
        config=config,
        result=result,
    )

    values = asdict(summary)

    assert values["detected_source_count"] == 1
    assert values["measured_source_count"] == 1
    assert values["valid_measurement_count"] == 1
    assert values["measurement_completeness"] == 1.0
    assert values["valid_measurement_fraction"] == 1.0
    assert values["invalid_measurement_count"] == 0
    assert values["missing_measurement_count"] == 0
    assert values["non_positive_flux_count"] == 0
    assert values["crowded_source_count"] == 1
    assert values["crowding_fraction"] == 1.0
    assert values["crowding_radius"] == 10.0
    assert values["median_net_flux"] == 1400.0
    assert values["median_background_level"] == 25.0
    assert values["evaluation_algorithm"] == "Photometry Evaluation"
    assert values["evaluation_timestamp"] == EVALUATION_TIME


def test_target_source_match_is_preserved_in_summary_and_source_records() -> None:
    """
    A resolved target-source association is represented both at the
    experiment level and on the matching source record.
    """

    sources = (
        _create_source(
            "1",
            x=10.1,
            y=9.9,
            peak=2000.0,
        ),
        _create_source(
            "2",
            x=15.0,
            y=15.0,
            peak=1000.0,
        ),
    )

    measurements = (
        _create_measurement("1"),
        _create_measurement("2"),
    )

    evaluations = (
        _create_source_evaluation("1"),
        _create_source_evaluation("2"),
    )

    target_match = TargetSourceMatch(
        target_name="LHS 3844",
        source_identifier="1",
        target_x=10.0,
        target_y=10.0,
        source_x=10.1,
        source_y=9.9,
        separation_pixels=0.141421356,
    )

    config, result = _create_result(
        sources=sources,
        measurements=measurements,
        source_evaluations=evaluations,
        target_match=target_match,
    )

    summary, records = ExperimentExporter().export(
        config=config,
        result=result,
    )

    summary_values = asdict(summary)

    assert summary_values["target_source_identifier"] == "1"
    assert summary_values["target_x"] == 10.0
    assert summary_values["target_y"] == 10.0
    assert summary_values["source_x"] == 10.1
    assert summary_values["source_y"] == 9.9
    assert summary_values["target_source_separation_pixels"] == (
        pytest.approx(0.141421356)
    )

    assert records[0].is_target_source is True
    assert records[1].is_target_source is False


def test_field_experiment_has_no_target_source_association() -> None:
    """
    Field-level experiments do not require a target-source association.
    """

    config = _create_config(
        experiment_type=ExperimentType.CROWDED_FIELD,
        source_scope=SourceScope.FIELD,
    )

    sources = (
        _create_source(
            "1",
            x=10.0,
            y=10.0,
            peak=2000.0,
        ),
        _create_source(
            "2",
            x=15.0,
            y=15.0,
            peak=1000.0,
        ),
    )

    measurements = (
        _create_measurement("1"),
        _create_measurement("2"),
    )

    evaluations = (
        _create_source_evaluation("1"),
        _create_source_evaluation("2"),
    )

    config, result = _create_result(
        config=config,
        sources=sources,
        measurements=measurements,
        source_evaluations=evaluations,
    )

    summary, records = ExperimentExporter().export(
        config=config,
        result=result,
    )

    summary_values = asdict(summary)

    assert summary_values["target_source_identifier"] is None
    assert summary_values["target_x"] is None
    assert summary_values["target_y"] is None
    assert summary_values["source_x"] is None
    assert summary_values["source_y"] is None
    assert summary_values["target_source_separation_pixels"] is None

    assert all(
        record.is_target_source is False
        for record in records
    )


def test_source_measurement_record_preserves_all_measurement_values() -> None:
    """
    Source-level export preserves the complete photometric measurement
    without recalculating any scientific quantity.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement(
        "1",
        aperture_flux=1500.0,
        background_level=25.0,
        background_flux=100.0,
        net_flux=1400.0,
        instrumental_magnitude=-7.865,
        aperture_area=12.56,
        background_area=50.27,
        aperture_coverage=0.95,
        annulus_coverage=0.85,
        background_fraction=100.0 / 1500.0,
        valid=True,
        quality_flag="VALID",
    )

    evaluation = _create_source_evaluation(
        "1",
        valid=True,
        quality_flag="VALID",
    )

    config, result = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
    )

    _, records = ExperimentExporter().export(
        config=config,
        result=result,
    )

    values = asdict(records[0])

    assert values["source_identifier"] == "1"
    assert values["x_centroid"] == 10.0
    assert values["y_centroid"] == 10.0
    assert values["peak_value"] == 2000.0

    assert values["aperture_flux"] == 1500.0
    assert values["background_level"] == 25.0
    assert values["background_flux"] == 100.0
    assert values["net_flux"] == 1400.0
    assert values["instrumental_magnitude"] == -7.865
    assert values["aperture_area"] == 12.56
    assert values["background_area"] == 50.27
    assert values["aperture_coverage"] == 0.95
    assert values["annulus_coverage"] == 0.85
    assert values["background_fraction"] == pytest.approx(
        100.0 / 1500.0
    )
    assert values["valid"] is True
    assert values["quality_flag"] == "VALID"


def test_invalid_measurement_is_exported_without_being_discarded() -> None:
    """
    Invalid photometric measurements remain available in the export.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement(
        "1",
        valid=False,
        quality_flag="NON_POSITIVE_FLUX",
        net_flux=-10.0,
        instrumental_magnitude=None,
    )

    evaluation = _create_source_evaluation(
        "1",
        valid=False,
        quality_flag="NON_POSITIVE_FLUX",
        net_flux=-10.0,
        instrumental_magnitude=None,
    )

    config, result = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
    )

    _, records = ExperimentExporter().export(
        config=config,
        result=result,
    )

    values = asdict(records[0])

    assert values["source_identifier"] == "1"
    assert values["valid"] is False
    assert values["quality_flag"] == "NON_POSITIVE_FLUX"
    assert values["net_flux"] == -10.0
    assert values["instrumental_magnitude"] is None


def test_missing_measurement_is_explicitly_exported() -> None:
    """
    A detected source without a photometric measurement still receives
    a source-level export record.
    """

    sources = (
        _create_source(
            "1",
            x=10.0,
            y=10.0,
            peak=2000.0,
        ),
        _create_source(
            "2",
            x=15.0,
            y=15.0,
            peak=1000.0,
        ),
    )

    measurements = (
        _create_measurement("1"),
    )

    evaluations = (
        _create_source_evaluation("1"),
        _create_source_evaluation(
            "2",
            valid=False,
            quality_flag="MISSING_MEASUREMENT",
            net_flux=None,
            instrumental_magnitude=None,
            background_level=None,
            background_fraction=None,
            aperture_coverage=None,
            annulus_coverage=None,
        ),
    )

    config, result = _create_result(
        sources=sources,
        measurements=measurements,
        source_evaluations=evaluations,
    )

    _, records = ExperimentExporter().export(
        config=config,
        result=result,
    )

    assert len(records) == 2

    missing_values = asdict(records[1])

    assert missing_values["source_identifier"] == "2"
    assert missing_values["valid"] is False
    assert missing_values["quality_flag"] == (
        "MISSING_MEASUREMENT"
    )

    assert missing_values["aperture_flux"] is None
    assert missing_values["background_level"] is None
    assert missing_values["background_flux"] is None
    assert missing_values["net_flux"] is None
    assert missing_values["instrumental_magnitude"] is None
    assert missing_values["aperture_area"] is None
    assert missing_values["background_area"] is None
    assert missing_values["aperture_coverage"] is None
    assert missing_values["annulus_coverage"] is None
    assert missing_values["background_fraction"] is None


def test_source_records_preserve_detection_order() -> None:
    """
    Source export records retain the ordering of the detection result.
    """

    sources = (
        _create_source(
            "3",
            x=30.0,
            y=30.0,
            peak=3000.0,
        ),
        _create_source(
            "1",
            x=10.0,
            y=10.0,
            peak=1000.0,
        ),
        _create_source(
            "2",
            x=20.0,
            y=20.0,
            peak=2000.0,
        ),
    )

    measurements = tuple(
        _create_measurement(identifier)
        for identifier in ("1", "2", "3")
    )

    evaluations = tuple(
        _create_source_evaluation(identifier)
        for identifier in ("1", "2", "3")
    )

    config, result = _create_result(
        sources=sources,
        measurements=measurements,
        source_evaluations=evaluations,
    )

    _, records = ExperimentExporter().export(
        config=config,
        result=result,
    )

    assert [record.source_identifier for record in records] == [
        "3",
        "1",
        "2",
    ]


def test_orphan_measurement_is_excluded_and_warning_is_emitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Measurements without corresponding detected sources are excluded
    from the source-level export and reported through logging.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurements = (
        _create_measurement("1"),
        _create_measurement("999"),
    )

    evaluations = (
        _create_source_evaluation("1"),
    )

    config, result = _create_result(
        sources=(source,),
        measurements=measurements,
        source_evaluations=evaluations,
    )

    with caplog.at_level("WARNING"):
        _, records = ExperimentExporter().export(
            config=config,
            result=result,
        )

    assert len(records) == 1
    assert records[0].source_identifier == "1"

    assert "Ignoring 1 orphan photometric measurement" in (
        caplog.text
    )


def test_duplicate_measurement_identifiers_are_rejected() -> None:
    """
    Duplicate photometric measurement identifiers cannot silently
    overwrite one another.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    duplicate_measurements = (
        _create_measurement("1"),
        _create_measurement("1"),
    )

    evaluations = (
        _create_source_evaluation("1"),
    )

    config, result = _create_result(
        sources=(source,),
        measurements=duplicate_measurements,
        source_evaluations=evaluations,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate photometric measurement",
    ):
        ExperimentExporter().export(
            config=config,
            result=result,
        )


def test_missing_source_evaluation_is_rejected() -> None:
    """
    Every detected source must have a corresponding evaluation record.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")

    config, result = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(),
    )

    with pytest.raises(
        ValueError,
        match="No source evaluation exists for detected source",
    ):
        ExperimentExporter().export(
            config=config,
            result=result,
        )


def test_none_configuration_is_rejected() -> None:
    """
    Export requires an experiment configuration.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    _, result = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
    )

    with pytest.raises(
        ValueError,
        match="Experiment configuration cannot be None",
    ):
        ExperimentExporter().export(
            config=None,  # type: ignore[arg-type]
            result=result,
        )


def test_none_workflow_result_is_rejected() -> None:
    """
    Export requires a completed WorkflowResult.
    """

    config = _create_config()

    with pytest.raises(
        ValueError,
        match="Workflow result cannot be None",
    ):
        ExperimentExporter().export(
            config=config,
            result=None,  # type: ignore[arg-type]
        )


def test_missing_detection_result_is_rejected() -> None:
    """
    Export requires source detection results.
    """

    config = _create_config()

    observation = _create_observation(
        sources=(),
        measurements=(),
    )

    observation.detections = None

    evaluation = _create_evaluation(
        (),
        detected_source_count=0,
        measured_source_count=0,
        valid_measurement_count=0,
        measurement_completeness=0.0,
        valid_measurement_fraction=0.0,
        invalid_measurement_count=0,
        missing_measurement_count=0,
    )

    result = WorkflowResult(
        observation=observation,
        evaluation=evaluation,
        target_match=None,
    )

    with pytest.raises(
        ValueError,
        match="without source detection results",
    ):
        ExperimentExporter().export(
            config=config,
            result=result,
        )


def test_missing_photometry_result_is_rejected() -> None:
    """
    Export requires photometry results.
    """

    config = _create_config()

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    observation = _create_observation(
        sources=(source,),
        measurements=(),
    )

    observation.photometry = None

    evaluation = _create_evaluation(
        (
            _create_source_evaluation(
                "1",
                valid=False,
                quality_flag="MISSING_MEASUREMENT",
                net_flux=None,
                instrumental_magnitude=None,
                background_level=None,
                background_fraction=None,
                aperture_coverage=None,
                annulus_coverage=None,
            ),
        ),
    )

    result = WorkflowResult(
        observation=observation,
        evaluation=evaluation,
        target_match=None,
    )

    with pytest.raises(
        ValueError,
        match="without photometry results",
    ):
        ExperimentExporter().export(
            config=config,
            result=result,
        )


@pytest.mark.parametrize(
    (
        "source_detection_completed",
        "photometry_completed",
        "expected_message",
    ),
    (
        (
            False,
            True,
            "source detection is not marked as completed",
        ),
        (
            True,
            False,
            "photometry is not marked as completed",
        ),
    ),
)
def test_incomplete_processing_state_is_rejected(
    source_detection_completed: bool,
    photometry_completed: bool,
    expected_message: str,
) -> None:
    """
    Export requires both source detection and photometry to be marked
    as completed.
    """

    config = _create_config()

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    observation = _create_observation(
        sources=(source,),
        measurements=(measurement,),
        source_detection_completed=source_detection_completed,
        photometry_completed=photometry_completed,
    )

    result = WorkflowResult(
        observation=observation,
        evaluation=_create_evaluation(
            (evaluation,),
        ),
        target_match=None,
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        ExperimentExporter().export(
            config=config,
            result=result,
        )


def test_missing_dataset_identifier_is_rejected() -> None:
    """
    Export provenance requires a stable dataset identifier.
    """

    config = _create_config()

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    observation = _create_observation(
        sources=(source,),
        measurements=(measurement,),
        dataset_identifier="",
    )

    result = WorkflowResult(
        observation=observation,
        evaluation=_create_evaluation(
            (evaluation,),
        ),
        target_match=None,
    )

    with pytest.raises(
        ValueError,
        match="dataset identifier",
    ):
        ExperimentExporter().export(
            config=config,
            result=result,
        )


def test_missing_target_name_is_rejected() -> None:
    """
    Export requires a meaningful target name.
    """

    config = _create_config()

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    observation = _create_observation(
        sources=(source,),
        measurements=(measurement,),
    )

    observation.metadata = ObservationMetadata(
        target_name="",
        observation_time=OBSERVATION_TIME,
        right_ascension=340.92941667,
        declination=-69.17210833,
        instrument="TESS",
        filter_name="TESS",
        exposure_time=120.0,
        wcs=None,
    )

    result = WorkflowResult(
        observation=observation,
        evaluation=_create_evaluation(
            (evaluation,),
        ),
        target_match=None,
    )

    with pytest.raises(
        ValueError,
        match="target name",
    ):
        ExperimentExporter().export(
            config=config,
            result=result,
        )


def test_experiment_identifier_is_deterministic() -> None:
    """
    Exporting identical experiment provenance twice produces the same
    experiment identifier.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    config, result = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
        dataset_identifier="DATASET-001",
    )

    exporter = ExperimentExporter()

    first_summary, _ = exporter.export(
        config=config,
        result=result,
    )

    second_summary, _ = exporter.export(
        config=config,
        result=result,
    )

    assert first_summary.experiment_id == (
        second_summary.experiment_id
    )


def test_experiment_identifier_changes_when_pipeline_changes() -> None:
    """
    Baseline and SCARLET processing branches receive different
    experiment identifiers even when the astronomical dataset is the
    same.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    baseline_config, baseline_result = _create_result(
        config=_create_config(
            pipeline=ProcessingPipeline.BASELINE,
        ),
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
        dataset_identifier="DATASET-001",
    )

    scarlet_config, scarlet_result = _create_result(
        config=_create_config(
            pipeline=ProcessingPipeline.SCARLET,
        ),
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
        dataset_identifier="DATASET-001",
    )

    exporter = ExperimentExporter()

    baseline_summary, _ = exporter.export(
        config=baseline_config,
        result=baseline_result,
    )

    scarlet_summary, _ = exporter.export(
        config=scarlet_config,
        result=scarlet_result,
    )

    assert baseline_summary.experiment_id != (
        scarlet_summary.experiment_id
    )


def test_experiment_identifier_changes_when_dataset_changes() -> None:
    """
    Different datasets receive different experiment identifiers.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    config_a, result_a = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
        dataset_identifier="DATASET-A",
    )

    config_b, result_b = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
        dataset_identifier="DATASET-B",
    )

    exporter = ExperimentExporter()

    summary_a, _ = exporter.export(
        config=config_a,
        result=result_a,
    )

    summary_b, _ = exporter.export(
        config=config_b,
        result=result_b,
    )

    assert summary_a.experiment_id != summary_b.experiment_id


def test_export_does_not_modify_workflow_observation() -> None:
    """
    Export is a read-only transformation and must not mutate the
    Observation or its processing products.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    config, result = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
    )

    observation = result.observation

    original_detections = observation.detections
    original_photometry = observation.photometry
    original_processing_state = (
        observation.processing_state
    )

    ExperimentExporter().export(
        config=config,
        result=result,
    )

    assert observation.detections is original_detections
    assert observation.photometry is original_photometry
    assert observation.processing_state is original_processing_state

    assert observation.processing_state.source_detection_completed is True
    assert observation.processing_state.photometry_completed is True


def test_export_records_are_immutable() -> None:
    """
    Export DTOs are immutable value objects.
    """

    source = _create_source(
        "1",
        x=10.0,
        y=10.0,
        peak=2000.0,
    )

    measurement = _create_measurement("1")
    evaluation = _create_source_evaluation("1")

    config, result = _create_result(
        sources=(source,),
        measurements=(measurement,),
        source_evaluations=(evaluation,),
    )

    summary, records = ExperimentExporter().export(
        config=config,
        result=result,
    )

    with pytest.raises(
        AttributeError,
    ):
        summary.target_name = "Modified"  # type: ignore[misc]

    with pytest.raises(
        AttributeError,
    ):
        records[0].source_identifier = "Modified"  # type: ignore[misc]