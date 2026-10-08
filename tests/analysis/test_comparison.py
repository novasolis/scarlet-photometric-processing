"""
Tests for the Baseline-versus-SCARLET experiment comparison boundary.

These tests verify that the comparison layer:

* constructs a pipeline-independent experiment identity;
* accepts valid Baseline-versus-SCARLET pairs;
* requires the correct processing pipeline for each side;
* rejects mismatched scientific/acquisition identity;
* rejects uncontrolled detection configuration;
* rejects uncontrolled aperture-photometry configuration;
* preserves immutable comparison objects; and
* does not compare experimental outcomes as if they were configuration.

Scientific measurements are intentionally not tested here.  Those belong
to the downstream source/cadence comparison and analysis layers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace

import pytest

from analysis.comparison import (
    ComparisonCompatibilityError,
    ExperimentComparisonKey,
    ExperimentComparisonPair,
    build_comparison_key,
    pair_experiments,
    validate_comparison_pair,
)
from export.models import ExperimentSummary


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def baseline_summary() -> ExperimentSummary:
    """Return a valid Baseline experiment summary."""
    return _make_summary(
        experiment_id="TESS-LC-001-B",
        pipeline="Baseline",
    )


@pytest.fixture
def scarlet_summary() -> ExperimentSummary:
    """Return a valid SCARLET experiment summary."""
    return _make_summary(
        experiment_id="TESS-LC-001-S",
        pipeline="SCARLET",
    )


def _make_summary(
    *,
    experiment_id: str = "EXP-001",
    target_name: str = "HD 209458",
    survey: str = "TESS",
    experiment_type: str = "Target-source photometry",
    source_scope: str = "TARGET",
    pipeline: str = "Baseline",
    observation_time: datetime | None = None,
    instrument: str | None = "TESS",
    filter_name: str | None = "TESS",
    exposure_time: float | None = 120.0,
    dataset_identifier: str = "tess:hd_209458:sector-001",
    detected_source_count: int = 25,
    detection_algorithm: str = "DAOStarFinder",
    detection_threshold: float = 5.0,
    target_source_identifier: str | None = "source-001",
    target_x: float | None = 100.0,
    target_y: float | None = 100.0,
    source_x: float | None = 100.2,
    source_y: float | None = 99.8,
    target_source_separation_pixels: float | None = 0.2828427,
    photometry_algorithm: str = "AperturePhotometry",
    aperture_radius: float = 3.0,
    annulus_inner_radius: float = 5.0,
    annulus_outer_radius: float = 8.0,
    measured_source_count: int = 25,
    valid_measurement_count: int = 24,
    measurement_completeness: float = 0.96,
    valid_measurement_fraction: float = 0.96,
    invalid_measurement_count: int = 1,
    missing_measurement_count: int = 0,
    non_positive_flux_count: int = 0,
    crowded_source_count: int = 4,
    crowding_fraction: float = 0.16,
    crowding_radius: float = 5.0,
    median_net_flux: float | None = 1000.0,
    median_background_level: float | None = 10.0,
    median_background_fraction: float | None = 0.01,
    evaluation_algorithm: str = "PhotometryEvaluator",
    evaluation_timestamp: datetime | None = None,
) -> ExperimentSummary:
    """
    Build a fully populated ExperimentSummary for comparison tests.

    The fixture deliberately keeps the scientific output values independent
    from the controlled-comparison fields so tests can demonstrate that
    different outcomes remain valid.
    """
    if observation_time is None:
        observation_time = datetime(
            2026,
            1,
            1,
            0,
            0,
            tzinfo=timezone.utc,
        )

    if evaluation_timestamp is None:
        evaluation_timestamp = datetime(
            2026,
            1,
            1,
            0,
            1,
            tzinfo=timezone.utc,
        )

    return ExperimentSummary(
        experiment_id=experiment_id,
        target_name=target_name,
        survey=survey,
        experiment_type=experiment_type,
        source_scope=source_scope,
        pipeline=pipeline,
        observation_time=observation_time,
        instrument=instrument,
        filter_name=filter_name,
        exposure_time=exposure_time,
        dataset_identifier=dataset_identifier,
        detected_source_count=detected_source_count,
        detection_algorithm=detection_algorithm,
        detection_threshold=detection_threshold,
        target_source_identifier=target_source_identifier,
        target_x=target_x,
        target_y=target_y,
        source_x=source_x,
        source_y=source_y,
        target_source_separation_pixels=target_source_separation_pixels,
        photometry_algorithm=photometry_algorithm,
        aperture_radius=aperture_radius,
        annulus_inner_radius=annulus_inner_radius,
        annulus_outer_radius=annulus_outer_radius,
        measured_source_count=measured_source_count,
        valid_measurement_count=valid_measurement_count,
        measurement_completeness=measurement_completeness,
        valid_measurement_fraction=valid_measurement_fraction,
        invalid_measurement_count=invalid_measurement_count,
        missing_measurement_count=missing_measurement_count,
        non_positive_flux_count=non_positive_flux_count,
        crowded_source_count=crowded_source_count,
        crowding_fraction=crowding_fraction,
        crowding_radius=crowding_radius,
        median_net_flux=median_net_flux,
        median_background_level=median_background_level,
        median_background_fraction=median_background_fraction,
        evaluation_algorithm=evaluation_algorithm,
        evaluation_timestamp=evaluation_timestamp,
    )


# ============================================================================
# ExperimentComparisonKey
# ============================================================================


def test_build_comparison_key_excludes_pipeline(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """
    Baseline and SCARLET executions of the same experiment share one key.

    Pipeline must not be part of comparison identity because Baseline and
    SCARLET are deliberately the two processing alternatives being compared.
    """
    baseline_key = build_comparison_key(baseline_summary)
    scarlet_key = build_comparison_key(scarlet_summary)

    assert baseline_key == scarlet_key

    assert baseline_key.survey == "TESS"
    assert baseline_key.target_name == "HD 209458"
    assert baseline_key.experiment_type == "Target-source photometry"
    assert baseline_key.source_scope == "TARGET"
    assert (
        baseline_key.dataset_identifier
        == "tess:hd_209458:sector-001"
    )


def test_comparison_key_is_immutable(
    baseline_summary: ExperimentSummary,
) -> None:
    """ExperimentComparisonKey is an immutable value object."""
    key = build_comparison_key(baseline_summary)

    with pytest.raises(AttributeError):
        key.pipeline = "SCARLET"  # type: ignore[attr-defined]


def test_comparison_key_rejects_empty_identity_fields() -> None:
    """Comparison identity fields must not be empty."""
    with pytest.raises(ValueError, match="target_name"):
        ExperimentComparisonKey(
            survey="TESS",
            target_name="",
            experiment_type="Target-source photometry",
            source_scope="TARGET",
            dataset_identifier="dataset-001",
        )


def test_comparison_key_rejects_non_string_identity_fields() -> None:
    """Comparison identity fields must be strings."""
    with pytest.raises(TypeError, match="survey"):
        ExperimentComparisonKey(
            survey=123,  # type: ignore[arg-type]
            target_name="HD 209458",
            experiment_type="Target-source photometry",
            source_scope="TARGET",
            dataset_identifier="dataset-001",
        )


# ============================================================================
# build_comparison_key input validation
# ============================================================================


def test_build_comparison_key_rejects_non_summary() -> None:
    """Comparison keys can only be constructed from ExperimentSummary."""
    with pytest.raises(TypeError, match="ExperimentSummary"):
        build_comparison_key(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", ""),
        ("target_name", ""),
        ("survey", ""),
        ("experiment_type", ""),
        ("source_scope", ""),
        ("pipeline", ""),
        ("dataset_identifier", ""),
        ("detection_algorithm", ""),
        ("photometry_algorithm", ""),
    ],
)
def test_build_comparison_key_rejects_empty_summary_fields(
    baseline_summary: ExperimentSummary,
    field: str,
    value: str,
) -> None:
    """Required exported summary identity/configuration fields cannot be empty."""
    summary = replace(
        baseline_summary,
        **{field: value},
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match=field,
    ):
        build_comparison_key(summary)


# ============================================================================
# Valid comparison
# ============================================================================


def test_validate_comparison_pair_accepts_valid_pair(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """A controlled Baseline-versus-SCARLET pair is accepted."""
    key = validate_comparison_pair(
        baseline=baseline_summary,
        scarlet=scarlet_summary,
    )

    assert isinstance(key, ExperimentComparisonKey)
    assert key == build_comparison_key(baseline_summary)


def test_pair_experiments_returns_validated_pair(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """pair_experiments returns the immutable comparison pair."""
    result = pair_experiments(
        baseline=baseline_summary,
        scarlet=scarlet_summary,
    )

    assert isinstance(result, ExperimentComparisonPair)
    assert result.baseline is baseline_summary
    assert result.scarlet is scarlet_summary
    assert result.key == build_comparison_key(baseline_summary)


def test_comparison_pair_is_immutable(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """ExperimentComparisonPair is immutable."""
    result = pair_experiments(
        baseline=baseline_summary,
        scarlet=scarlet_summary,
    )

    with pytest.raises(AttributeError):
        result.baseline = scarlet_summary  # type: ignore[misc]


# ============================================================================
# Pipeline validation
# ============================================================================


def test_baseline_side_requires_baseline_pipeline(
    scarlet_summary: ExperimentSummary,
) -> None:
    """The baseline side must actually use the Baseline pipeline."""
    invalid_baseline = replace(
        scarlet_summary,
        experiment_id="TESS-LC-001-B",
        pipeline="SCARLET",
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="expected 'Baseline'",
    ):
        validate_comparison_pair(
            baseline=invalid_baseline,
            scarlet=scarlet_summary,
        )


def test_scarlet_side_requires_scarlet_pipeline(
    baseline_summary: ExperimentSummary,
) -> None:
    """The SCARLET side must actually use the SCARLET pipeline."""
    invalid_scarlet = replace(
        baseline_summary,
        experiment_id="TESS-LC-001-S",
        pipeline="Baseline",
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="expected 'SCARLET'",
    ):
        validate_comparison_pair(
            baseline=baseline_summary,
            scarlet=invalid_scarlet,
        )


def test_unknown_pipeline_is_rejected(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """An unsupported pipeline cannot participate in the comparison."""
    invalid_scarlet = replace(
        scarlet_summary,
        pipeline="Experimental",
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="expected 'SCARLET'",
    ):
        validate_comparison_pair(
            baseline=baseline_summary,
            scarlet=invalid_scarlet,
        )


# ============================================================================
# Scientific/acquisition identity validation
# ============================================================================


@pytest.mark.parametrize(
    ("field", "baseline_value", "scarlet_value", "message"),
    [
        (
            "survey",
            "TESS",
            "SDSS",
            "survey differs",
        ),
        (
            "target_name",
            "HD 209458",
            "HD 189733",
            "target differs",
        ),
        (
            "experiment_type",
            "Target-source photometry",
            "Source-field photometry",
            "experiment_type differs",
        ),
        (
            "source_scope",
            "TARGET",
            "FIELD",
            "source_scope differs",
        ),
        (
            "dataset_identifier",
            "tess:hd_209458:sector-001",
            "tess:hd_209458:sector-002",
            "dataset_identifier differs",
        ),
    ],
)
def test_comparison_rejects_identity_mismatch(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
    field: str,
    baseline_value: str,
    scarlet_value: str,
    message: str,
) -> None:
    """Controlled comparisons require identical experiment identity."""
    baseline = replace(
        baseline_summary,
        **{field: baseline_value},
    )

    scarlet = replace(
        scarlet_summary,
        **{field: scarlet_value},
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match=message,
    ):
        validate_comparison_pair(
            baseline=baseline,
            scarlet=scarlet,
        )


# ============================================================================
# Detection configuration validation
# ============================================================================


def test_comparison_rejects_detection_algorithm_mismatch(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """Baseline and SCARLET must use the same detector."""
    scarlet = replace(
        scarlet_summary,
        detection_algorithm="DifferentDetector",
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="detection_algorithm differs",
    ):
        validate_comparison_pair(
            baseline=baseline_summary,
            scarlet=scarlet,
        )


def test_comparison_rejects_detection_threshold_mismatch(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """Baseline and SCARLET must use the same detection threshold."""
    scarlet = replace(
        scarlet_summary,
        detection_threshold=7.0,
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="detection_threshold differs",
    ):
        validate_comparison_pair(
            baseline=baseline_summary,
            scarlet=scarlet,
        )


# ============================================================================
# Aperture-photometry configuration validation
# ============================================================================


def test_comparison_rejects_photometry_algorithm_mismatch(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """Baseline and SCARLET must use the same photometry implementation."""
    scarlet = replace(
        scarlet_summary,
        photometry_algorithm="DifferentPhotometer",
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="photometry_algorithm differs",
    ):
        validate_comparison_pair(
            baseline=baseline_summary,
            scarlet=scarlet,
        )


def test_comparison_rejects_aperture_radius_mismatch(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """Baseline and SCARLET must use identical aperture geometry."""
    scarlet = replace(
        scarlet_summary,
        aperture_radius=4.0,
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="aperture_radius differs",
    ):
        validate_comparison_pair(
            baseline=baseline_summary,
            scarlet=scarlet,
        )


def test_comparison_rejects_inner_annulus_radius_mismatch(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """Baseline and SCARLET must use identical inner annulus geometry."""
    scarlet = replace(
        scarlet_summary,
        annulus_inner_radius=6.0,
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="annulus_inner_radius differs",
    ):
        validate_comparison_pair(
            baseline=baseline_summary,
            scarlet=scarlet,
        )


def test_comparison_rejects_outer_annulus_radius_mismatch(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """Baseline and SCARLET must use identical outer annulus geometry."""
    scarlet = replace(
        scarlet_summary,
        annulus_outer_radius=10.0,
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="annulus_outer_radius differs",
    ):
        validate_comparison_pair(
            baseline=baseline_summary,
            scarlet=scarlet,
        )


# ============================================================================
# Experimental outcomes are deliberately not comparison constraints
# ============================================================================


def test_different_scientific_outcomes_do_not_invalidate_pair(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """
    Different experimental outcomes remain a valid comparison.

    SCARLET is expected to change source-level and evaluation outcomes.
    Those differences are precisely what the downstream analysis is intended
    to investigate.
    """
    scarlet = replace(
        scarlet_summary,
        detected_source_count=31,
        measured_source_count=29,
        valid_measurement_count=28,
        measurement_completeness=0.90,
        valid_measurement_fraction=0.9655,
        invalid_measurement_count=1,
        missing_measurement_count=1,
        non_positive_flux_count=0,
        crowded_source_count=7,
        crowding_fraction=0.2414,
        median_net_flux=1250.0,
        median_background_level=7.5,
        median_background_fraction=0.006,
    )

    key = validate_comparison_pair(
        baseline=baseline_summary,
        scarlet=scarlet,
    )

    assert key == build_comparison_key(baseline_summary)


def test_different_target_match_does_not_invalidate_pair(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """
    Different target-source association outcomes remain comparable.

    Target matching is an experimental result, not a controlled input.
    """
    scarlet = replace(
        scarlet_summary,
        target_source_identifier="source-009",
        source_x=101.5,
        source_y=98.5,
        target_source_separation_pixels=2.1213203,
    )

    key = validate_comparison_pair(
        baseline=baseline_summary,
        scarlet=scarlet,
    )

    assert key == build_comparison_key(baseline_summary)


# ============================================================================
# Input validation
# ============================================================================


def test_validate_comparison_pair_rejects_none_baseline(
    scarlet_summary: ExperimentSummary,
) -> None:
    """A missing Baseline summary is invalid."""
    with pytest.raises(TypeError, match="ExperimentSummary"):
        validate_comparison_pair(
            baseline=None,  # type: ignore[arg-type]
            scarlet=scarlet_summary,
        )


def test_validate_comparison_pair_rejects_none_scarlet(
    baseline_summary: ExperimentSummary,
) -> None:
    """A missing SCARLET summary is invalid."""
    with pytest.raises(TypeError, match="ExperimentSummary"):
        validate_comparison_pair(
            baseline=baseline_summary,
            scarlet=None,  # type: ignore[arg-type]
        )


def test_pair_experiments_rejects_invalid_pair(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """pair_experiments must not bypass comparison validation."""
    invalid_scarlet = replace(
        scarlet_summary,
        dataset_identifier="different-dataset",
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="dataset_identifier differs",
    ):
        pair_experiments(
            baseline=baseline_summary,
            scarlet=invalid_scarlet,
        )


# ============================================================================
# Explicit key consistency
# ============================================================================


def test_comparison_pair_rejects_incorrect_supplied_key(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """
    ExperimentComparisonPair must reject a key inconsistent with its summaries.
    """
    incorrect_key = ExperimentComparisonKey(
        survey="TESS",
        target_name="HD 189733",
        experiment_type="Target-source photometry",
        source_scope="TARGET",
        dataset_identifier="tess:hd_189733:sector-001",
    )

    with pytest.raises(
        ComparisonCompatibilityError,
        match="supplied comparison key",
    ):
        ExperimentComparisonPair(
            key=incorrect_key,
            baseline=baseline_summary,
            scarlet=scarlet_summary,
        )


def test_comparison_pair_accepts_correct_supplied_key(
    baseline_summary: ExperimentSummary,
    scarlet_summary: ExperimentSummary,
) -> None:
    """A pair accepts the exact key derived from its summaries."""
    key = build_comparison_key(baseline_summary)

    result = ExperimentComparisonPair(
        key=key,
        baseline=baseline_summary,
        scarlet=scarlet_summary,
    )

    assert result.key == key