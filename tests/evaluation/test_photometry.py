"""
Unit tests for survey-independent photometry evaluation.

These tests exercise the evaluation layer using deterministic domain
objects. No survey data, FITS files, Photutils, Lightkurve, MAST or
network access is required.

The tests verify that PhotometryEvaluator correctly:

    * evaluates valid measurements;
    * distinguishes measured, valid, invalid and missing measurements;
    * counts non-positive net flux;
    * calculates source crowding;
    * calculates aggregate statistics;
    * handles non-finite measurement values;
    * rejects malformed domain state; and
    * enforces evaluation prerequisites.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from astropy.nddata import NDData

from core.models import EvaluationConfig
from domain.enums import ObservationSource, Survey
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
)
from evaluation.photometry import PhotometryEvaluator


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _create_evaluation_config(
    *,
    crowding_radius: float = 10.0,
) -> EvaluationConfig:
    """
    Create a deterministic evaluation configuration for tests.
    """

    return EvaluationConfig(
        matching_tolerance=2.0,
        crowding_radius=crowding_radius,
    )


def _create_source(
    identifier: str,
    x: float,
    y: float,
    *,
    peak: float = 100.0,
) -> DetectedSource:
    """
    Create a deterministic detected source.
    """

    return DetectedSource(
        identifier=identifier,
        x_centroid=x,
        y_centroid=y,
        peak_value=peak,
    )


def _create_measurement(
    identifier: str,
    *,
    net_flux: float | None = 100.0,
    instrumental_magnitude: float | None = 10.0,
    background_level: float | None = 10.0,
    background_fraction: float | None = 0.1,
    aperture_coverage: float | None = 1.0,
    annulus_coverage: float | None = 1.0,
    valid: bool = True,
    quality_flag: str = "VALID",
) -> PhotometryMeasurement:
    """
    Create a deterministic photometric measurement.
    """

    return PhotometryMeasurement(
        source_identifier=identifier,
        x_centroid=0.0,
        y_centroid=0.0,
        aperture_flux=110.0,
        background_level=background_level,
        background_flux=10.0,
        net_flux=net_flux,
        instrumental_magnitude=instrumental_magnitude,
        aperture_area=12.0,
        background_area=20.0,
        aperture_coverage=aperture_coverage,
        annulus_coverage=annulus_coverage,
        background_fraction=background_fraction,
        valid=valid,
        quality_flag=quality_flag,
    )


def _create_observation(
    sources: tuple[DetectedSource, ...],
    measurements: tuple[PhotometryMeasurement, ...],
    *,
    source_detection_completed: bool = True,
    photometry_completed: bool = True,
) -> Observation:
    """
    Create a minimal valid Observation containing detection and
    photometry results.

    The image itself is deliberately trivial because the evaluator
    operates only on source and measurement domain objects.
    """

    timestamp = datetime.now(timezone.utc)

    image = ImageData(
        image=NDData(
            data=np.zeros(
                (11, 11),
                dtype=float,
            ),
        ),
    )

    metadata = ObservationMetadata(
        target_name="Test Target",
        observation_time=timestamp,
        right_ascension=100.0,
        declination=-30.0,
        instrument="Test Instrument",
        filter_name="Test Filter",
        exposure_time=120.0,
        wcs=None,
    )

    provenance = ObservationProvenance(
        source=ObservationSource.SPACE_BASED,
        survey=Survey.TESS,
        dataset_identifier="TEST",
        filename=None,
        download_timestamp=timestamp,
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
            detection_threshold=10.0,
            detection_timestamp=timestamp,
        ),
        photometry=PhotometryResult(
            measurements=measurements,
            algorithm="Classical Aperture Photometry",
            aperture_radius=2.0,
            annulus_inner_radius=3.0,
            annulus_outer_radius=4.0,
            photometry_timestamp=timestamp,
        ),
        processing_state=ProcessingState(
            source_detection_completed=source_detection_completed,
            photometry_completed=photometry_completed,
        ),
    )

    return observation


def _create_evaluator(
    *,
    crowding_radius: float = 10.0,
) -> PhotometryEvaluator:
    """
    Create a PhotometryEvaluator using deterministic test settings.
    """

    return PhotometryEvaluator(
        config=_create_evaluation_config(
            crowding_radius=crowding_radius,
        ),
    )


# ---------------------------------------------------------------------------
# Core evaluation behaviour
# ---------------------------------------------------------------------------


def test_all_valid_measurements_produce_complete_evaluation() -> None:
    """
    All detected sources have valid photometric measurements.

    Expected:

        detected = 3
        measured = 3
        valid = 3
        invalid = 0
        missing = 0
        completeness = 1.0
        valid fraction = 1.0
    """

    sources = (
        _create_source("1", 0.0, 0.0),
        _create_source("2", 3.0, 4.0),
        _create_source("3", 30.0, 0.0),
    )

    measurements = (
        _create_measurement(
            "1",
            net_flux=100.0,
            background_level=10.0,
            background_fraction=0.10,
        ),
        _create_measurement(
            "2",
            net_flux=200.0,
            background_level=20.0,
            background_fraction=0.20,
        ),
        _create_measurement(
            "3",
            net_flux=300.0,
            background_level=30.0,
            background_fraction=0.30,
        ),
    )

    observation = _create_observation(
        sources,
        measurements,
    )

    evaluation = _create_evaluator().evaluate(
        observation,
    )

    assert evaluation.detected_source_count == 3
    assert evaluation.measured_source_count == 3
    assert evaluation.valid_measurement_count == 3
    assert evaluation.invalid_measurement_count == 0
    assert evaluation.missing_measurement_count == 0

    assert evaluation.measurement_completeness == pytest.approx(
        1.0,
    )

    assert evaluation.valid_measurement_fraction == pytest.approx(
        1.0,
    )

    assert evaluation.non_positive_flux_count == 0

    #
    # Sources 1 and 2 are 5 pixels apart and therefore crowded.
    # Source 3 is isolated.
    #
    assert evaluation.crowded_source_count == 2
    assert evaluation.crowding_fraction == pytest.approx(
        2.0 / 3.0,
    )

    assert evaluation.median_net_flux == pytest.approx(
        200.0,
    )

    assert evaluation.median_background_level == pytest.approx(
        20.0,
    )

    assert evaluation.median_background_fraction == pytest.approx(
        0.20,
    )

    assert len(evaluation.source_evaluations) == 3


def test_invalid_measurement_is_counted_but_not_valid() -> None:
    """
    Invalid measurement records remain part of the evaluation but
    are excluded from the valid measurement population.
    """

    sources = (
        _create_source("1", 0.0, 0.0),
        _create_source("2", 30.0, 0.0),
    )

    measurements = (
        _create_measurement(
            "1",
            net_flux=100.0,
            valid=True,
            quality_flag="VALID",
        ),
        _create_measurement(
            "2",
            net_flux=50.0,
            valid=False,
            quality_flag="INSUFFICIENT_ANNULUS_COVERAGE",
        ),
    )

    observation = _create_observation(
        sources,
        measurements,
    )

    evaluation = _create_evaluator().evaluate(
        observation,
    )

    assert evaluation.detected_source_count == 2
    assert evaluation.measured_source_count == 2
    assert evaluation.valid_measurement_count == 1
    assert evaluation.invalid_measurement_count == 1
    assert evaluation.missing_measurement_count == 0

    assert evaluation.measurement_completeness == pytest.approx(
        1.0,
    )

    assert evaluation.valid_measurement_fraction == pytest.approx(
        0.5,
    )

    source_evaluation = evaluation.source_evaluations[1]

    assert source_evaluation.source_identifier == "2"
    assert source_evaluation.valid_measurement is False
    assert (
        source_evaluation.quality_flag
        == "INSUFFICIENT_ANNULUS_COVERAGE"
    )


def test_missing_measurement_is_explicitly_represented() -> None:
    """
    A detected source without a measurement record is represented
    using MISSING_MEASUREMENT.
    """

    sources = (
        _create_source("1", 0.0, 0.0),
        _create_source("2", 30.0, 0.0),
    )

    measurements = (
        _create_measurement(
            "1",
            net_flux=100.0,
        ),
    )

    observation = _create_observation(
        sources,
        measurements,
    )

    evaluation = _create_evaluator().evaluate(
        observation,
    )

    assert evaluation.detected_source_count == 2
    assert evaluation.measured_source_count == 1
    assert evaluation.valid_measurement_count == 1
    assert evaluation.invalid_measurement_count == 0
    assert evaluation.missing_measurement_count == 1

    assert evaluation.measurement_completeness == pytest.approx(
        0.5,
    )

    assert evaluation.valid_measurement_fraction == pytest.approx(
        1.0,
    )

    missing = evaluation.source_evaluations[1]

    assert missing.source_identifier == "2"
    assert missing.valid_measurement is False
    assert missing.quality_flag == "MISSING_MEASUREMENT"

    assert missing.net_flux is None
    assert missing.instrumental_magnitude is None
    assert missing.background_level is None


def test_non_positive_flux_is_counted_separately() -> None:
    """
    Zero and negative net flux are counted independently from the
    measurement validity flag.
    """

    sources = (
        _create_source("1", 0.0, 0.0),
        _create_source("2", 30.0, 0.0),
        _create_source("3", 60.0, 0.0),
    )

    measurements = (
        _create_measurement(
            "1",
            net_flux=100.0,
            valid=True,
            quality_flag="VALID",
        ),
        _create_measurement(
            "2",
            net_flux=0.0,
            valid=False,
            quality_flag="NON_POSITIVE_NET_FLUX",
        ),
        _create_measurement(
            "3",
            net_flux=-25.0,
            valid=False,
            quality_flag="NON_POSITIVE_NET_FLUX",
        ),
    )

    observation = _create_observation(
        sources,
        measurements,
    )

    evaluation = _create_evaluator().evaluate(
        observation,
    )

    assert evaluation.non_positive_flux_count == 2

    assert evaluation.valid_measurement_count == 1
    assert evaluation.invalid_measurement_count == 2


# ---------------------------------------------------------------------------
# Crowding
# ---------------------------------------------------------------------------


def test_crowding_is_calculated_from_source_positions() -> None:
    """
    Crowding is determined independently from photometric validity.
    """

    sources = (
        _create_source("1", 0.0, 0.0),
        _create_source("2", 6.0, 8.0),
        _create_source("3", 30.0, 30.0),
    )

    measurements = tuple(
        _create_measurement(
            identifier,
        )
        for identifier in (
            "1",
            "2",
            "3",
        )
    )

    observation = _create_observation(
        sources,
        measurements,
    )

    evaluation = _create_evaluator(
        crowding_radius=10.0,
    ).evaluate(
        observation,
    )

    #
    # Distance between sources 1 and 2 is exactly 10 pixels.
    #
    assert evaluation.source_evaluations[0].neighbour_count == 1
    assert evaluation.source_evaluations[1].neighbour_count == 1
    assert evaluation.source_evaluations[2].neighbour_count == 0

    assert evaluation.source_evaluations[0].crowded is True
    assert evaluation.source_evaluations[1].crowded is True
    assert evaluation.source_evaluations[2].crowded is False

    assert evaluation.crowded_source_count == 2
    assert evaluation.crowding_fraction == pytest.approx(
        2.0 / 3.0,
    )


def test_crowding_radius_is_respected() -> None:
    """
    Sources outside the configured crowding radius are not classified
    as neighbours.
    """

    sources = (
        _create_source("1", 0.0, 0.0),
        _create_source("2", 6.0, 8.0),
    )

    measurements = (
        _create_measurement("1"),
        _create_measurement("2"),
    )

    observation = _create_observation(
        sources,
        measurements,
    )

    evaluation = _create_evaluator(
        crowding_radius=9.99,
    ).evaluate(
        observation,
    )

    assert evaluation.crowded_source_count == 0
    assert evaluation.crowding_fraction == pytest.approx(
        0.0,
    )

    assert evaluation.source_evaluations[0].neighbour_count == 0
    assert evaluation.source_evaluations[1].neighbour_count == 0


# ---------------------------------------------------------------------------
# Numerical robustness
# ---------------------------------------------------------------------------


def test_non_finite_measurement_values_are_sanitised() -> None:
    """
    NaN and infinity must not enter evaluation statistics.
    """

    sources = (
        _create_source("1", 0.0, 0.0),
        _create_source("2", 30.0, 0.0),
    )

    measurements = (
        _create_measurement(
            "1",
            net_flux=float("nan"),
            instrumental_magnitude=float("inf"),
            background_level=float("nan"),
            background_fraction=float("inf"),
            aperture_coverage=float("nan"),
            annulus_coverage=float("inf"),
            valid=True,
        ),
        _create_measurement(
            "2",
            net_flux=200.0,
            instrumental_magnitude=12.0,
            background_level=20.0,
            background_fraction=0.2,
            valid=True,
        ),
    )

    observation = _create_observation(
        sources,
        measurements,
    )

    evaluation = _create_evaluator().evaluate(
        observation,
    )

    first = evaluation.source_evaluations[0]
    second = evaluation.source_evaluations[1]

    assert first.net_flux is None
    assert first.instrumental_magnitude is None
    assert first.background_level is None
    assert first.background_fraction is None
    assert first.aperture_coverage is None
    assert first.annulus_coverage is None

    assert second.net_flux == pytest.approx(
        200.0,
    )

    #
    # Median uses only finite values.
    #
    assert evaluation.median_net_flux == pytest.approx(
        200.0,
    )

    assert evaluation.median_background_level == pytest.approx(
        20.0,
    )

    assert evaluation.median_background_fraction == pytest.approx(
        0.2,
    )


# ---------------------------------------------------------------------------
# Measurement integrity
# ---------------------------------------------------------------------------


def test_duplicate_measurement_identifiers_are_rejected() -> None:
    """
    Duplicate measurement records for the same source are invalid.
    """

    sources = (
        _create_source("1", 0.0, 0.0),
    )

    measurements = (
        _create_measurement("1"),
        _create_measurement("1"),
    )

    observation = _create_observation(
        sources,
        measurements,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate photometric measurement",
    ):
        _create_evaluator().evaluate(
            observation,
        )


def test_duplicate_detected_source_identifiers_are_rejected() -> None:
    """
    Duplicate detected-source identifiers are invalid.
    """

    sources = (
        _create_source("1", 0.0, 0.0),
        _create_source("1", 10.0, 10.0),
    )

    measurements = (
        _create_measurement("1"),
    )

    observation = _create_observation(
        sources,
        measurements,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate detected source identifier",
    ):
        _create_evaluator().evaluate(
            observation,
        )


def test_orphan_measurement_does_not_inflate_measured_source_count() -> None:
    """
    A measurement whose identifier does not correspond to a detected
    source is not counted as a measured detected source.
    """

    sources = (
        _create_source("1", 0.0, 0.0),
    )

    measurements = (
        _create_measurement("1"),
        _create_measurement("orphan"),
    )

    observation = _create_observation(
        sources,
        measurements,
    )

    evaluation = _create_evaluator().evaluate(
        observation,
    )

    assert evaluation.detected_source_count == 1
    assert evaluation.measured_source_count == 1
    assert evaluation.valid_measurement_count == 1
    assert evaluation.missing_measurement_count == 0


# ---------------------------------------------------------------------------
# Observation validation
# ---------------------------------------------------------------------------


def test_none_observation_is_rejected() -> None:
    """
    Evaluation requires a real Observation.
    """

    with pytest.raises(
        ValueError,
        match="Observation cannot be None",
    ):
        _create_evaluator().evaluate(
            None,  # type: ignore[arg-type]
        )


def test_missing_detection_result_is_rejected() -> None:
    """
    Evaluation cannot proceed without completed source detection.
    """

    observation = _create_observation(
        sources=(),
        measurements=(),
    )

    observation.detections = None

    with pytest.raises(
        ValueError,
        match="no source detections",
    ):
        _create_evaluator().evaluate(
            observation,
        )


def test_missing_photometry_result_is_rejected() -> None:
    """
    Evaluation cannot proceed without photometry results.
    """

    observation = _create_observation(
        sources=(
            _create_source("1", 0.0, 0.0),
        ),
        measurements=(
            _create_measurement("1"),
        ),
    )

    observation.photometry = None

    with pytest.raises(
        ValueError,
        match="no photometric result",
    ):
        _create_evaluator().evaluate(
            observation,
        )


def test_source_detection_must_be_completed() -> None:
    """
    The source-detection processing flag is part of the evaluation
    contract.
    """

    observation = _create_observation(
        sources=(
            _create_source("1", 0.0, 0.0),
        ),
        measurements=(
            _create_measurement("1"),
        ),
        source_detection_completed=False,
        photometry_completed=True,
    )

    with pytest.raises(
        ValueError,
        match="Source detection is not marked as completed",
    ):
        _create_evaluator().evaluate(
            observation,
        )


def test_photometry_must_be_completed() -> None:
    """
    The photometry processing flag is part of the evaluation contract.
    """

    observation = _create_observation(
        sources=(
            _create_source("1", 0.0, 0.0),
        ),
        measurements=(
            _create_measurement("1"),
        ),
        source_detection_completed=True,
        photometry_completed=False,
    )

    with pytest.raises(
        ValueError,
        match="Photometry is not marked as completed",
    ):
        _create_evaluator().evaluate(
            observation,
        )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_invalid_crowding_radius_is_rejected() -> None:
    """
    Crowding radius must be finite and greater than zero.
    """

    with pytest.raises(
        ValueError,
        match="Crowding radius must be a finite value",
    ):
        PhotometryEvaluator(
            config=_create_evaluation_config(
                crowding_radius=0.0,
            ),
        )


@pytest.mark.parametrize(
    "radius",
    [
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_positive_or_non_finite_crowding_radius_is_rejected(
    radius: float,
) -> None:
    """
    Non-positive and non-finite crowding radii are invalid.
    """

    with pytest.raises(
        ValueError,
        match="Crowding radius must be a finite value",
    ):
        PhotometryEvaluator(
            config=_create_evaluation_config(
                crowding_radius=radius,
            ),
        )