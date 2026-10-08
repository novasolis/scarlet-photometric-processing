"""
Tests for experiment configuration construction and validation.

These tests verify the current target-centred configuration factory:

    - default target selection;
    - benchmark target resolution;
    - survey-specific acquisition defaults;
    - explicit acquisition overrides;
    - processing-pipeline selection;
    - time-series compatibility; and
    - differential-photometry configuration requirements.

The current architecture does not use SourceScope.  Source selection is
resolved later by the workflow from the detected catalogue and LocalField.
"""

from __future__ import annotations

import pytest

from config.experiment_config import (
    DEFAULT_SDSS_TARGET,
    DEFAULT_TESS_TARGET,
    TESS_ACQUISITION_CONFIG,
    TESS_LOCAL_FIELD_ACQUISITION_CONFIG,
    create_experiment_config,
)
from core.models import (
    AcquisitionConfig,
    DifferentialPhotometryConfig,
    ReferenceStarConfig,
    TimeSeriesConfig,
)
from domain.enums import ExperimentType, ProcessingPipeline, Survey


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _reference_config() -> DifferentialPhotometryConfig:
    """Create a deterministic differential-photometry configuration."""
    return DifferentialPhotometryConfig(
        reference_stars=(
            ReferenceStarConfig(
                identifier="Reference A",
                right_ascension=97.630000,
                declination=29.670000,
            ),
            ReferenceStarConfig(
                identifier="Reference B",
                right_ascension=97.640000,
                declination=29.680000,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Differential-photometry configuration
# ---------------------------------------------------------------------------


def test_differential_photometry_requires_configuration() -> None:
    """A differential experiment without references must be rejected."""
    with pytest.raises(
        ValueError,
        match="DIFFERENTIAL_PHOTOMETRY experiments require",
    ):
        create_experiment_config(
            survey=Survey.TESS,
            experiment_type=ExperimentType.DIFFERENTIAL_PHOTOMETRY,
            target_name="wasp_12",
        )


def test_valid_differential_photometry_configuration_is_preserved() -> None:
    """A valid differential configuration must be retained unchanged."""
    differential_config = _reference_config()

    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.DIFFERENTIAL_PHOTOMETRY,
        target_name="wasp_12",
        differential_photometry=differential_config,
    )

    assert config.experiment_type is ExperimentType.DIFFERENTIAL_PHOTOMETRY
    assert config.differential_photometry is differential_config
    assert config.differential_photometry is not None
    assert (
        config.differential_photometry.reference_stars
        == differential_config.reference_stars
    )


def test_differential_configuration_is_rejected_for_target_photometry() -> None:
    """Differential configuration must not leak into normal target experiments."""
    with pytest.raises(
        ValueError,
        match="only valid for DIFFERENTIAL_PHOTOMETRY",
    ):
        create_experiment_config(
            survey=Survey.TESS,
            experiment_type=ExperimentType.TARGET_PHOTOMETRY,
            target_name="wasp_12",
            differential_photometry=_reference_config(),
        )


def test_differential_configuration_is_none_for_normal_experiment() -> None:
    """Normal experiments must not acquire an implicit differential config."""
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
        target_name="wasp_12",
    )

    assert config.differential_photometry is None


# ---------------------------------------------------------------------------
# Default target selection
# ---------------------------------------------------------------------------


def test_tess_default_target_is_selected() -> None:
    """TESS configuration uses the configured TESS benchmark by default."""
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
    )

    assert DEFAULT_TESS_TARGET == "lhs_3844"
    assert config.target.target_name == "LHS 3844"


def test_sdss_default_target_is_selected() -> None:
    """SDSS configuration uses the configured SDSS benchmark by default."""
    config = create_experiment_config(
        survey=Survey.SDSS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
    )

    assert DEFAULT_SDSS_TARGET == "m67"
    assert config.target.target_name == "M67"


# ---------------------------------------------------------------------------
# Benchmark target resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target_name", "expected_name"),
    (
        ("hd_209458", "HD 209458"),
        ("lhs_3844", "LHS 3844"),
        ("wasp_12", "WASP-12"),
        ("hd_189733", "HD 189733"),
    ),
)
def test_named_tess_benchmark_is_resolved(
    target_name: str,
    expected_name: str,
) -> None:
    """Named TESS benchmark identifiers must resolve to target models."""
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
        target_name=target_name,
    )

    assert config.target.target_name == expected_name
    assert config.target.right_ascension is not None
    assert config.target.declination is not None


def test_unknown_benchmark_target_is_rejected() -> None:
    """An unknown target identifier must produce a useful configuration error."""
    with pytest.raises(
        ValueError,
        match="Unknown TESS benchmark target",
    ):
        create_experiment_config(
            survey=Survey.TESS,
            experiment_type=ExperimentType.TARGET_PHOTOMETRY,
            target_name="does_not_exist",
        )


# ---------------------------------------------------------------------------
# Survey-specific acquisition configuration
# ---------------------------------------------------------------------------


def test_target_experiment_uses_tess_target_acquisition() -> None:
    """Normal target-centred TESS experiments use the SPOC acquisition path."""
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
        target_name="wasp_12",
    )

    assert config.acquisition == TESS_ACQUISITION_CONFIG
    assert config.acquisition.author == "SPOC"
    assert config.acquisition.cadence_seconds == 120
    assert config.acquisition.cutout_size == 25


def test_differential_experiment_uses_target_centric_tess_acquisition() -> None:
    """Differential TESS experiments use the target-centric acquisition path."""
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.DIFFERENTIAL_PHOTOMETRY,
        target_name="wasp_12",
        differential_photometry=_reference_config(),
    )

    assert config.acquisition == TESS_ACQUISITION_CONFIG
    assert config.acquisition.author == "SPOC"
    assert config.acquisition.cadence_seconds == 120
    assert config.acquisition.cutout_size == 25


def test_local_field_acquisition_can_be_explicitly_selected() -> None:
    """
    Target-centred experiments requiring additional spatial context can
    explicitly select the extended LocalField acquisition configuration.
    """
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
        target_name="ngc_3532",
        acquisition=TESS_LOCAL_FIELD_ACQUISITION_CONFIG,
    )

    assert config.acquisition is TESS_LOCAL_FIELD_ACQUISITION_CONFIG
    assert config.acquisition.cutout_size == 50
    assert config.acquisition.author is None
    assert config.acquisition.cadence_seconds is None


def test_explicit_acquisition_overrides_survey_default() -> None:
    """An explicitly supplied acquisition configuration must be preserved."""
    custom_acquisition = AcquisitionConfig(
        sector=42,
        author="SPOC",
        cadence_seconds=120,
        cutout_size=30,
    )

    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
        target_name="wasp_12",
        acquisition=custom_acquisition,
    )

    assert config.acquisition is custom_acquisition


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------


def test_baseline_pipeline_is_default() -> None:
    """Normal configuration defaults to the Baseline processing pipeline."""
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
        target_name="wasp_12",
    )

    assert config.pipeline is ProcessingPipeline.BASELINE


def test_scarlet_pipeline_is_preserved() -> None:
    """SCARLET must be retained when explicitly selected."""
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
        pipeline=ProcessingPipeline.SCARLET,
        target_name="wasp_12",
    )

    assert config.pipeline is ProcessingPipeline.SCARLET


# ---------------------------------------------------------------------------
# Time-series configuration
# ---------------------------------------------------------------------------


def test_time_series_is_rejected_for_sdss() -> None:
    """Enabled time-series execution is currently TESS-only."""
    with pytest.raises(
        ValueError,
        match="Time-series execution is currently supported only for TESS",
    ):
        create_experiment_config(
            survey=Survey.SDSS,
            experiment_type=ExperimentType.TARGET_PHOTOMETRY,
            target_name="m67",
            time_series=TimeSeriesConfig(enabled=True),
        )


def test_disabled_time_series_is_accepted_for_sdss() -> None:
    """Disabled time-series configuration is harmless for SDSS."""
    config = create_experiment_config(
        survey=Survey.SDSS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
        target_name="m67",
        time_series=TimeSeriesConfig(enabled=False),
    )

    assert config.time_series.enabled is False


def test_enabled_tess_time_series_is_preserved() -> None:
    """Enabled TESS time-series configuration must be retained."""
    time_series = TimeSeriesConfig(
        enabled=True,
        cadence_indices=(0, 1, 2, 3),
    )

    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
        target_name="wasp_12",
        time_series=time_series,
    )

    assert config.time_series is time_series
    assert config.time_series.enabled is True
    assert config.time_series.cadence_indices == (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Target coordinate contract
# ---------------------------------------------------------------------------


def test_target_configuration_contains_celestial_coordinates() -> None:
    """
    Target-centred experiments must resolve a benchmark with celestial
    coordinates before entering the production workflow.
    """
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.TARGET_PHOTOMETRY,
        target_name="wasp_12",
    )

    assert config.target.right_ascension is not None
    assert config.target.declination is not None


def test_differential_experiment_contains_target_coordinates() -> None:
    """Differential experiments must retain target celestial coordinates."""
    config = create_experiment_config(
        survey=Survey.TESS,
        experiment_type=ExperimentType.DIFFERENTIAL_PHOTOMETRY,
        target_name="wasp_12",
        differential_photometry=_reference_config(),
    )

    assert config.target.right_ascension is not None
    assert config.target.declination is not None


# ---------------------------------------------------------------------------
# Basic factory validation
# ---------------------------------------------------------------------------


def test_invalid_experiment_type_is_rejected() -> None:
    """The factory must reject values that are not ExperimentType members."""
    with pytest.raises(
        ValueError,
        match="Unsupported experiment type",
    ):
        create_experiment_config(
            survey=Survey.TESS,
            experiment_type="not-an-experiment",  # type: ignore[arg-type]
            target_name="wasp_12",
        )


def test_invalid_pipeline_is_rejected() -> None:
    """The factory must reject values that are not ProcessingPipeline members."""
    with pytest.raises(
        ValueError,
        match="Unsupported processing pipeline",
    ):
        create_experiment_config(
            survey=Survey.TESS,
            experiment_type=ExperimentType.TARGET_PHOTOMETRY,
            pipeline="not-a-pipeline",  # type: ignore[arg-type]
            target_name="wasp_12",
        )