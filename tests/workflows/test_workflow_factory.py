from __future__ import annotations

from typing import Any, cast
from unittest.mock import ANY, MagicMock

import pytest

import workflows.workflow_factory as factory
from config.experiment_config import create_experiment_config
from domain.enums import ExperimentType, ProcessingPipeline, Survey
from target.local_field import LocalFieldSelector


def cfg(*, survey: Survey = Survey.TESS, experiment_type: ExperimentType = ExperimentType.TARGET_PHOTOMETRY, pipeline: ProcessingPipeline = ProcessingPipeline.BASELINE) -> Any:
    return create_experiment_config(
        survey=survey, experiment_type=experiment_type, pipeline=pipeline,
        target_name="lhs_3844" if survey is Survey.TESS else "m67",
    )


@pytest.fixture
def patched(monkeypatch):
    c = {name: MagicMock(name=name) for name in (
        "tess_acquirer", "tess_builder", "sdss_acquirer", "sdss_builder",
        "background", "scarlet", "detector", "target_resolver",
        "reference_resolver", "photometer", "differential_photometer",
        "evaluator", "local_field_selector", "workflow",
    )}
    constructors = {
        "TESSAcquirer": MagicMock(return_value=c["tess_acquirer"]),
        "TESSObservationBuilder": MagicMock(return_value=c["tess_builder"]),
        "SDSSAcquirer": MagicMock(return_value=c["sdss_acquirer"]),
        "SDSSObservationBuilder": MagicMock(return_value=c["sdss_builder"]),
        "BackgroundSubtractor": MagicMock(return_value=c["background"]),
        "ScarletPreprocessor": MagicMock(return_value=c["scarlet"]),
        "DAOStarFinderDetector": MagicMock(return_value=c["detector"]),
        "TargetSourceResolver": MagicMock(return_value=c["target_resolver"]),
        "ReferenceSourceResolver": MagicMock(return_value=c["reference_resolver"]),
        "AperturePhotometer": MagicMock(return_value=c["photometer"]),
        "DifferentialPhotometer": MagicMock(return_value=c["differential_photometer"]),
        "PhotometryEvaluator": MagicMock(return_value=c["evaluator"]),
        "LocalFieldSelector": MagicMock(spec=LocalFieldSelector, return_value=c["local_field_selector"]),
        "PhotometricWorkflow": MagicMock(return_value=c["workflow"]),
        "get_logger": MagicMock(return_value=MagicMock(name="logger")),
    }
    for name, value in constructors.items():
        monkeypatch.setattr(factory, name, value, raising=False)
    return c, constructors


def test_factory_returns_workflow(patched):
    c, constructors = patched
    assert factory.create_workflow(cfg()) is c["workflow"]


def test_factory_selects_tess_components(patched):
    c, _ = patched
    factory.create_workflow(cfg(survey=Survey.TESS))
    workflow_mock = cast(MagicMock, factory.PhotometricWorkflow)
    kwargs = workflow_mock.call_args.kwargs
    assert kwargs["acquirer"] is c["tess_acquirer"]
    assert kwargs["builder"] is c["tess_builder"]


def test_factory_selects_sdss_components(patched):
    c, _ = patched
    factory.create_workflow(cfg(survey=Survey.SDSS))
    workflow_mock = cast(MagicMock, factory.PhotometricWorkflow)
    kwargs = workflow_mock.call_args.kwargs
    assert kwargs["acquirer"] is c["sdss_acquirer"]
    assert kwargs["builder"] is c["sdss_builder"]


def test_factory_always_constructs_target_resolver(patched):
    c, constructors = patched
    config = cfg()
    factory.create_workflow(config)
    constructors["TargetSourceResolver"].assert_called_once_with(
        matching_tolerance=config.evaluation.matching_tolerance, logger=ANY,
    )
    workflow_mock = cast(MagicMock, factory.PhotometricWorkflow)
    assert workflow_mock.call_args.kwargs["target_resolver"] is c["target_resolver"]


def test_factory_constructs_local_field_selector_and_injects_it(patched):
    c, constructors = patched
    factory.create_workflow(cfg())
    constructors["LocalFieldSelector"].assert_called_once_with(max_sources=50, logger=ANY)
    workflow_mock = cast(MagicMock, factory.PhotometricWorkflow)
    assert workflow_mock.call_args.kwargs["local_field_selector"] is c["local_field_selector"]


def test_factory_does_not_create_differential_services_for_target_photometry(patched):
    _, constructors = patched
    factory.create_workflow(cfg())
    constructors["ReferenceSourceResolver"].assert_not_called()
    constructors["DifferentialPhotometer"].assert_not_called()
    workflow_mock = cast(MagicMock, factory.PhotometricWorkflow)
    kwargs = workflow_mock.call_args.kwargs
    assert kwargs["reference_resolver"] is None
    assert kwargs["differential_photometer"] is None


def test_factory_creates_differential_services_for_differential_experiment(patched):
    from core.models import DifferentialPhotometryConfig, ReferenceStarConfig
    c, constructors = patched
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.DIFFERENTIAL_PHOTOMETRY,
        pipeline=ProcessingPipeline.BASELINE,
        target_name="gv_car",
        differential_photometry=DifferentialPhotometryConfig(
            reference_stars=(ReferenceStarConfig(identifier="Reference A", right_ascension=100.01, declination=-30.01),),
        ),
    )
    factory.create_workflow(config)
    constructors["ReferenceSourceResolver"].assert_called_once_with(
        matching_tolerance=config.evaluation.matching_tolerance, logger=ANY,
    )
    constructors["DifferentialPhotometer"].assert_called_once_with()
    workflow_mock = cast(MagicMock, factory.PhotometricWorkflow)
    kwargs = workflow_mock.call_args.kwargs
    assert kwargs["reference_resolver"] is c["reference_resolver"]
    assert kwargs["differential_photometer"] is c["differential_photometer"]


def test_scarlet_is_injected_separately(patched):
    c, constructors = patched
    factory.create_workflow(cfg(pipeline=ProcessingPipeline.SCARLET))
    constructors["ScarletPreprocessor"].assert_called_once_with(logger=ANY)
    workflow_mock = cast(MagicMock, factory.PhotometricWorkflow)
    assert workflow_mock.call_args.kwargs["scarlet_preprocessor"] is c["scarlet"]


def test_baseline_does_not_create_scarlet(patched):
    _, constructors = patched
    factory.create_workflow(cfg())
    constructors["ScarletPreprocessor"].assert_not_called()
    workflow_mock = cast(MagicMock, factory.PhotometricWorkflow)
    assert workflow_mock.call_args.kwargs["scarlet_preprocessor"] is None


def test_factory_rejects_non_experiment_config():
    with pytest.raises(TypeError, match="requires an ExperimentConfig"):
        factory.create_workflow(object())
