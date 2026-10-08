from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from config.experiment_config import create_experiment_config
from core.exceptions import PhotometricAnalysisError
from core.models import DifferentialPhotometryConfig, ReferenceStarConfig
from domain.enums import ExperimentType, ProcessingPipeline, Survey
from domain.models import (
    DetectedSource,
    DetectionResult,
    LocalField,
    Observation,
    TargetSourceMatch,
)
from evaluation.models import PhotometryEvaluation
from target.local_field import LocalFieldSelector
from workflows.models import WorkflowResult
from workflows.photometric_workflow import PhotometricWorkflow


def config(
    *,
    experiment_type=ExperimentType.TARGET_PHOTOMETRY,
    pipeline=ProcessingPipeline.BASELINE,
    differential_photometry=None,
):
    # The current experiment_config.py deliberately uses separate TESS
    # registries: normal target-centred experiments use TESS_BENCHMARK_TARGETS
    # (including lhs_3844), while differential photometry uses
    # TESS_DIFFERENTIAL_PHOTOMETRY_TARGETS (including gv_car).
    target_name = (
        "gv_car"
        if experiment_type is ExperimentType.DIFFERENTIAL_PHOTOMETRY
        else "lhs_3844"
    )

    return create_experiment_config(
        survey=Survey.TESS,
        experiment_type=experiment_type,
        pipeline=pipeline,
        target_name=target_name,
        differential_photometry=differential_photometry,
    )


def detections():
    return DetectionResult(
        sources=(
            DetectedSource(identifier="1", x_centroid=5.1, y_centroid=5.1, peak_value=100.0),
            DetectedSource(identifier="2", x_centroid=7.0, y_centroid=5.0, peak_value=80.0),
            DetectedSource(identifier="3", x_centroid=5.0, y_centroid=8.0, peak_value=60.0),
        ),
        algorithm="DAOStarFinder",
        detection_threshold=3.0,
        detection_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def target_match():
    return TargetSourceMatch(
        target_name="GV Car",
        source_identifier="1",
        target_x=5.0,
        target_y=5.0,
        source_x=5.1,
        source_y=5.1,
        separation_pixels=0.1414213562,
    )


def components(observation):
    acquirer = MagicMock()
    acquirer.acquire.return_value = object()

    builder = MagicMock()
    builder.build.return_value = observation

    preprocessor = MagicMock()
    preprocessor.process.side_effect = lambda x: x

    detector = MagicMock()
    detection = detections()
    detector.detect.side_effect = lambda x: (setattr(x, "detections", detection) or x)

    photometer = MagicMock()
    photometer.measure.side_effect = lambda x: x

    evaluator = MagicMock()
    evaluator.evaluate.return_value = MagicMock(spec=PhotometryEvaluation)

    resolver = MagicMock()
    resolver.resolve.return_value = target_match()

    selector = MagicMock(spec=LocalFieldSelector)
    field = LocalField(
        target_source=detection.sources[0],
        neighbour_sources=detection.sources[1:],
        neighbour_distances_pixels=(1.902629759, 2.901723625),
        available_source_count=3,
        max_sources=50,
    )
    selector.select.return_value = field

    scarlet = MagicMock()
    scarlet.process.side_effect = lambda x, **kwargs: x

    references = MagicMock()
    differential = MagicMock()

    return locals()


def make_workflow(
    observation,
    c,
    *,
    experiment_type=ExperimentType.TARGET_PHOTOMETRY,
    pipeline=ProcessingPipeline.BASELINE,
):
    return PhotometricWorkflow(
        config=config(experiment_type=experiment_type, pipeline=pipeline),
        acquirer=c["acquirer"],
        builder=c["builder"],
        preprocessor=c["preprocessor"],
        detector=c["detector"],
        photometer=c["photometer"],
        evaluator=c["evaluator"],
        target_resolver=c["resolver"],
        reference_resolver=(
            c["references"]
            if experiment_type is ExperimentType.DIFFERENTIAL_PHOTOMETRY
            else None
        ),
        differential_photometer=(
            c["differential"]
            if experiment_type is ExperimentType.DIFFERENTIAL_PHOTOMETRY
            else None
        ),
        scarlet_preprocessor=(
            c["scarlet"]
            if pipeline is ProcessingPipeline.SCARLET
            else None
        ),
        local_field_selector=c["selector"],
    )


def test_target_workflow_resolves_target_and_returns_match():
    observation = MagicMock(spec=Observation)
    c = components(observation)

    result = make_workflow(observation, c).run()

    assert isinstance(result, WorkflowResult)
    assert result.target_match is c["resolver"].resolve.return_value


def test_workflow_constructs_and_returns_local_field():
    observation = MagicMock(spec=Observation)
    c = components(observation)

    result = make_workflow(observation, c).run()

    c["selector"].select.assert_called_once_with(
        detections=observation.detections,
        target_match=c["resolver"].resolve.return_value,
    )
    assert result.local_field is c["selector"].select.return_value


def test_local_field_is_selected_before_photometry():
    observation = MagicMock(spec=Observation)
    c = components(observation)
    events = []

    c["resolver"].resolve.side_effect = (
        lambda x: events.append("resolve") or target_match()
    )
    c["selector"].select.side_effect = (
        lambda **kwargs: events.append("field") or c["selector"].return_value
    )
    c["photometer"].measure.side_effect = (
        lambda x: events.append("photometry") or x
    )
    c["evaluator"].evaluate.side_effect = (
        lambda x: events.append("evaluation")
        or MagicMock(spec=PhotometryEvaluation)
    )

    make_workflow(observation, c).run()

    assert (
        events.index("resolve")
        < events.index("field")
        < events.index("photometry")
        < events.index("evaluation")
    )


def test_scarlet_receives_the_same_local_field_source_centres():
    observation = MagicMock(spec=Observation)
    c = components(observation)

    result = make_workflow(
        observation,
        c,
        pipeline=ProcessingPipeline.SCARLET,
    ).run()

    c["scarlet"].process.assert_called_once_with(
        observation,
        source_centers=c["selector"].select.return_value.source_centers,
    )
    assert result.local_field is c["selector"].select.return_value


def test_baseline_does_not_invoke_scarlet():
    observation = MagicMock(spec=Observation)
    c = components(observation)

    result = make_workflow(observation, c).run()

    c["scarlet"].process.assert_not_called()
    assert result.local_field is c["selector"].select.return_value


def test_differential_photometry_remains_target_centred():
    observation = MagicMock(spec=Observation)
    c = components(observation)

    cfg = config(
        experiment_type=ExperimentType.DIFFERENTIAL_PHOTOMETRY,
        differential_photometry=DifferentialPhotometryConfig(
            reference_stars=(
                ReferenceStarConfig(
                    identifier="Reference A",
                    right_ascension=100.01,
                    declination=-30.01,
                ),
            ),
        ),
    )

    refs = cfg.differential_photometry.reference_stars

    c["references"].resolve.side_effect = (
        lambda current, reference: MagicMock(
            reference_identifier=reference.identifier
        )
    )

    workflow = PhotometricWorkflow(
        config=cfg,
        acquirer=c["acquirer"],
        builder=c["builder"],
        preprocessor=c["preprocessor"],
        detector=c["detector"],
        photometer=c["photometer"],
        evaluator=c["evaluator"],
        target_resolver=c["resolver"],
        reference_resolver=c["references"],
        differential_photometer=c["differential"],
        local_field_selector=c["selector"],
    )

    result = workflow.run()

    assert result.local_field is c["selector"].select.return_value
    assert c["references"].resolve.call_count == len(refs)
    c["differential"].calculate.assert_called_once()


def test_missing_detection_result_prevents_local_field_selection():
    observation = MagicMock(spec=Observation)
    observation.detections = None
    c = components(observation)

    c["detector"].detect.side_effect = None
    c["detector"].detect.return_value = observation

    with pytest.raises(
        PhotometricAnalysisError,
        match="LocalField selection requires a completed source-detection result",
    ):
        make_workflow(observation, c).run()

    c["selector"].select.assert_not_called()
    c["photometer"].measure.assert_not_called()


def test_missing_target_resolver_fails_before_local_field_selection():
    observation = MagicMock(spec=Observation)
    c = components(observation)

    workflow = PhotometricWorkflow(
        config=config(),
        acquirer=c["acquirer"],
        builder=c["builder"],
        preprocessor=c["preprocessor"],
        detector=c["detector"],
        photometer=c["photometer"],
        evaluator=c["evaluator"],
        target_resolver=None,
        local_field_selector=c["selector"],
    )

    with pytest.raises(
        PhotometricAnalysisError,
        match="Every target-centred experiment requires a TargetSourceResolver",
    ):
        workflow.run()

    c["selector"].select.assert_not_called()


def test_scarlet_requires_injected_preprocessor():
    observation = MagicMock(spec=Observation)
    c = components(observation)

    workflow = PhotometricWorkflow(
        config=config(pipeline=ProcessingPipeline.SCARLET),
        acquirer=c["acquirer"],
        builder=c["builder"],
        preprocessor=c["preprocessor"],
        detector=c["detector"],
        photometer=c["photometer"],
        evaluator=c["evaluator"],
        target_resolver=c["resolver"],
        local_field_selector=c["selector"],
        scarlet_preprocessor=None,
    )

    with pytest.raises(
        PhotometricAnalysisError,
        match="no ScarletPreprocessor",
    ):
        workflow.run()

    c["selector"].select.assert_called_once()


def test_unexpected_photometry_error_is_wrapped():
    observation = MagicMock(spec=Observation)
    c = components(observation)

    c["photometer"].measure.side_effect = RuntimeError(
        "Synthetic photometry failure."
    )

    with pytest.raises(PhotometricAnalysisError) as exc:
        make_workflow(observation, c).run()

    assert isinstance(exc.value.__cause__, RuntimeError)
