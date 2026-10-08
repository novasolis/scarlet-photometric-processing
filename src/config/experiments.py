"""
Final production experiment registry for the SCARLET photometry study.

Experiment sets:

    FIELD_ANALYSIS
        Manual single-field reconnaissance. Change the target_name below
        when a different field needs to be analysed.

    PHOTOMETRY_TEST
        Manual single-target Baseline-vs-SCARLET test. Change the survey,
        target_name and, where necessary, acquisition/time-series settings
        below. This is the controlled sandbox for validating SCARLET
        experiments before production.

    TARGET_PHOTOMETRY
        Final production target-centred aperture photometry across both
        SDSS and TESS. Every configured target is run under both BASELINE
        and SCARLET without time-series cadence processing.

    TIMESERIES_PHOTOMETRY
        Final production TESS target-centred time-series photometry.
        Every configured TESS target is run under both BASELINE and SCARLET
        using the same 300 quality-filtered cadence indices.

    DIFFERENTIAL_PHOTOMETRY
        Final production TESS time-series differential-photometry
        experiments using the configured reference stars and the same
        300 quality-filtered cadence indices.

Baseline and SCARLET are processing conditions of the same scientific
experiment. They are deliberately not separate experiment sets.

Survey-specific target registries and reusable configuration remain in
config.experiment_config. This module defines the experiment selection
and intentionally manual test/field entries.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

from config.experiment_config import (
    SDSS_ACQUISITION_CONFIG,
    SDSS_FIELD_TARGETS,
    SDSS_PHOTOMETRY_CONFIG,
    SDSS_PHOTOMETRY_TARGETS,
    TESS_ACQUISITION_CONFIG,
    TESS_FIELD_ACQUISITION_CONFIG,
    TESS_FIELD_TARGETS,
    TESS_LOCAL_FIELD_ACQUISITION_CONFIG,
    TESS_PHOTOMETRY_TARGETS,
    DIFFERENTIAL_PHOTOMETRY_REFERENCE_STARS,
    DIFFERENTIAL_PHOTOMETRY_TARGETS,
    create_experiment_config,
)
from core.models import (
    DifferentialPhotometryConfig,
    ExperimentConfig,
    TimeSeriesConfig,
)
from domain.enums import ExperimentType, ProcessingPipeline, Survey


class ExperimentSet(str, Enum):
    """The experiment-set categories used by the production runner."""

    FIELD_ANALYSIS = "field_analysis" #TEST
    PHOTOMETRY_TEST = "photometry_test" #TEST
    TARGET_PHOTOMETRY = "target_photometry" #PRODUCTION
    TIMESERIES_PHOTOMETRY = "timeseries_photometry" #PRODUCTION
    DIFFERENTIAL_PHOTOMETRY = "differential_photometry" #PRODUCTION


# Change only this value to select what the runner executes.
ACTIVE_EXPERIMENT_SET = ExperimentSet.PHOTOMETRY_TEST


PROCESSING_COMPARISON_PIPELINES: tuple[
    ProcessingPipeline,
    ...,
] = (
    ProcessingPipeline.BASELINE,
    ProcessingPipeline.SCARLET,
)


def with_processing_pipeline(
    experiment: ExperimentConfig,
    pipeline: ProcessingPipeline,
) -> ExperimentConfig:
    """Return a copy of an experiment using a different processing pipeline."""

    if not isinstance(experiment, ExperimentConfig):
        raise TypeError("experiment must be an ExperimentConfig.")

    if not isinstance(pipeline, ProcessingPipeline):
        raise TypeError("pipeline must be a ProcessingPipeline.")

    return replace(experiment, pipeline=pipeline)


def build_processing_comparison(
    experiment: ExperimentConfig,
    pipelines: tuple[
        ProcessingPipeline,
        ...,
    ] = PROCESSING_COMPARISON_PIPELINES,
) -> tuple[ExperimentConfig, ...]:
    """
    Expand one scientific experiment into controlled processing variants.

    All scientific and observational settings remain unchanged. Only the
    processing pipeline is changed.
    """

    if not isinstance(experiment, ExperimentConfig):
        raise TypeError("experiment must be an ExperimentConfig.")

    if not pipelines:
        raise ValueError("At least one processing pipeline is required.")

    if len(set(pipelines)) != len(pipelines):
        raise ValueError("Processing comparison pipelines must be unique.")

    for pipeline in pipelines:
        if not isinstance(pipeline, ProcessingPipeline):
            raise TypeError(
                "All processing comparison pipelines must be "
                "ProcessingPipeline instances.",
            )

    return tuple(
        with_processing_pipeline(experiment, pipeline)
        for pipeline in pipelines
    )


def expand_processing_comparison_set(
    experiments: tuple[ExperimentConfig, ...],
    pipelines: tuple[
        ProcessingPipeline,
        ...,
    ] = PROCESSING_COMPARISON_PIPELINES,
) -> tuple[ExperimentConfig, ...]:
    """Expand a collection of experiments into processing variants."""

    if not experiments:
        raise ValueError("At least one experiment is required.")

    expanded: list[ExperimentConfig] = []

    for experiment in experiments:
        expanded.extend(build_processing_comparison(experiment, pipelines))

    return tuple(expanded)


# ============================================================================
# Shared TESS production time-series configuration
# ============================================================================

TESS_PRODUCTION_CADENCE_INDICES: tuple[int, ...] = tuple(range(300))

TESS_PRODUCTION_TIME_SERIES_CONFIG = TimeSeriesConfig(
    enabled=True,
    cadence_indices=TESS_PRODUCTION_CADENCE_INDICES,
)


# ============================================================================
# FIELD_ANALYSIS -- intentionally manual
# ============================================================================

# Change these two values when a different configured field needs to be
# analysed. FIELD_ANALYSIS deliberately runs ONE field, not the entire
# field registry.
FIELD_ANALYSIS_SURVEY = Survey.SDSS
FIELD_ANALYSIS_TARGET = "ngc2420"


if FIELD_ANALYSIS_SURVEY is Survey.TESS:
    if FIELD_ANALYSIS_TARGET not in TESS_FIELD_TARGETS:
        raise ValueError(
            f"FIELD_ANALYSIS_TARGET {FIELD_ANALYSIS_TARGET!r} is not present "
            "in TESS_FIELD_TARGETS.",
        )

elif FIELD_ANALYSIS_SURVEY is Survey.SDSS:
    if FIELD_ANALYSIS_TARGET not in SDSS_FIELD_TARGETS:
        raise ValueError(
            f"FIELD_ANALYSIS_TARGET {FIELD_ANALYSIS_TARGET!r} is not present "
            "in SDSS_FIELD_TARGETS.",
        )

else:
    raise ValueError(
        "FIELD_ANALYSIS_SURVEY must be Survey.TESS or Survey.SDSS.",
    )


FIELD_ANALYSIS: tuple[ExperimentConfig, ...] = (
    create_experiment_config(
        survey=FIELD_ANALYSIS_SURVEY,
        experiment_type=ExperimentType.FIELD_ANALYSIS,
        target_name=FIELD_ANALYSIS_TARGET,
    ),
)


# ============================================================================
# PHOTOMETRY_TEST -- intentionally manual
# ============================================================================

PHOTOMETRY_TEST: tuple[ExperimentConfig, ...] = (
    *build_processing_comparison(
        create_experiment_config(
            survey=Survey.SDSS,
            experiment_type=ExperimentType.TARGET_PHOTOMETRY,
            target_name="m67_sand978",
            acquisition=SDSS_ACQUISITION_CONFIG,
            #time_series=TESS_PRODUCTION_TIME_SERIES_CONFIG
        ),
    ),
)


# ============================================================================
# TARGET_PHOTOMETRY -- final production
# ============================================================================
#
# Target-centred aperture photometry without time-series processing.
# This experiment set intentionally spans both surveys so that survey/source
# is an experimental factor rather than the experiment-set definition.
#
# TESS and SDSS retain their survey-specific acquisition configurations.

def _build_target_photometry_experiments() -> tuple[ExperimentConfig, ...]:
    """Build Baseline + SCARLET target-centred experiments for SDSS and TESS."""

    canonical_experiments: list[ExperimentConfig] = []

    for target_id in SDSS_PHOTOMETRY_TARGETS:
        canonical_experiments.append(
            create_experiment_config(
                survey=Survey.SDSS,
                experiment_type=ExperimentType.TARGET_PHOTOMETRY,
                target_name=target_id,
                acquisition=SDSS_ACQUISITION_CONFIG,
            ),
        )

    for target_id in TESS_PHOTOMETRY_TARGETS:
        canonical_experiments.append(
            create_experiment_config(
                survey=Survey.TESS,
                experiment_type=ExperimentType.TARGET_PHOTOMETRY,
                target_name=target_id,
                acquisition=TESS_LOCAL_FIELD_ACQUISITION_CONFIG,
            ),
        )

    return expand_processing_comparison_set(
        tuple(canonical_experiments),
    )


TARGET_PHOTOMETRY: tuple[ExperimentConfig, ...] = (
    _build_target_photometry_experiments()
)


# ============================================================================
# TIMESERIES_PHOTOMETRY -- final production
# ============================================================================

def _build_timeseries_photometry_experiments() -> tuple[ExperimentConfig, ...]:
    """Build Baseline + SCARLET TESS target time-series experiments."""

    canonical_experiments: list[ExperimentConfig] = []

    for target_id in TESS_PHOTOMETRY_TARGETS:
        canonical_experiments.append(
            create_experiment_config(
                survey=Survey.TESS,
                experiment_type=ExperimentType.TARGET_PHOTOMETRY,
                target_name=target_id,
                acquisition=TESS_LOCAL_FIELD_ACQUISITION_CONFIG,
                time_series=TESS_PRODUCTION_TIME_SERIES_CONFIG,
            ),
        )

    return expand_processing_comparison_set(
        tuple(canonical_experiments),
    )


TIMESERIES_PHOTOMETRY: tuple[ExperimentConfig, ...] = (
    _build_timeseries_photometry_experiments()
)


# ============================================================================
# DIFFERENTIAL_PHOTOMETRY -- intentionally staged
# ============================================================================

def _build_differential_photometry_experiments() -> tuple[ExperimentConfig, ...]:
    """
    Build staged TESS differential-photometry experiments.

    The target identifiers come from the shared TESS experiment configuration
    registry. Reference-star selections are maintained independently for each
    target and are supplied by DIFFERENTIAL_PHOTOMETRY_REFERENCE_STARS.
    """

    canonical_experiments: list[ExperimentConfig] = []

    for target_id in DIFFERENTIAL_PHOTOMETRY_TARGETS:
        canonical_experiments.append(
            create_experiment_config(
                survey=Survey.TESS,
                experiment_type=ExperimentType.DIFFERENTIAL_PHOTOMETRY,
                target_name=target_id,
                acquisition=TESS_LOCAL_FIELD_ACQUISITION_CONFIG,
                time_series=TESS_PRODUCTION_TIME_SERIES_CONFIG,
                differential_photometry=DifferentialPhotometryConfig(
                    reference_stars=DIFFERENTIAL_PHOTOMETRY_REFERENCE_STARS[
                        target_id
                    ],
                ),
            ),
        )

    return expand_processing_comparison_set(
        tuple(canonical_experiments),
    )


DIFFERENTIAL_PHOTOMETRY: tuple[ExperimentConfig, ...] = (
    _build_differential_photometry_experiments()
)


# ============================================================================
# Experiment-set registry and public runner interface
# ============================================================================

EXPERIMENT_SETS: dict[
    ExperimentSet,
    tuple[ExperimentConfig, ...],
] = {
    ExperimentSet.FIELD_ANALYSIS: FIELD_ANALYSIS,
    ExperimentSet.PHOTOMETRY_TEST: PHOTOMETRY_TEST,
    ExperimentSet.TARGET_PHOTOMETRY: TARGET_PHOTOMETRY,
    ExperimentSet.TIMESERIES_PHOTOMETRY: TIMESERIES_PHOTOMETRY,
    ExperimentSet.DIFFERENTIAL_PHOTOMETRY: DIFFERENTIAL_PHOTOMETRY,
}


def get_active_experiments() -> tuple[ExperimentConfig, ...]:
    """Return the experiments selected by ACTIVE_EXPERIMENT_SET."""

    return EXPERIMENT_SETS[ACTIVE_EXPERIMENT_SET]


EXPERIMENTS: tuple[ExperimentConfig, ...] = get_active_experiments()


def describe_active_experiment_set() -> str:
    """Return a concise description of the active experiment set."""

    return (
        f"Active experiment set: {ACTIVE_EXPERIMENT_SET.value} | "
        f"experiments: {len(get_active_experiments())}"
    )