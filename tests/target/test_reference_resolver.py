"""
Unit tests for ReferenceSourceResolver.

The resolver performs source-level reference association:

    configured RA/Dec
          ↓
       WCS transform
          ↓
    reference pixel position
          ↓
    nearest detected source
          ↓
    ReferenceSourceMatch

These tests use a synthetic celestial WCS and synthetic detected sources.
No external astronomical data is downloaded.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.wcs import WCS

from core.exceptions import ObservationError
from core.models import ReferenceStarConfig
from domain.enums import ObservationSource, Survey
from domain.models import (
    DetectedSource,
    DetectionResult,
    ImageData,
    Observation,
    ObservationMetadata,
    ObservationProvenance,
    ReferenceSourceMatch,
)
from target.reference_resolver import ReferenceSourceResolver


# ---------------------------------------------------------------------------
# Test constants.
# ---------------------------------------------------------------------------

REFERENCE_IDENTIFIER = "TIC 123456789"

REFERENCE_RA = 100.0
REFERENCE_DEC = -30.0

TIMESTAMP = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)

# Sentinel used to distinguish:
#
#     _create_observation()
#
# from:
#
#     _create_observation(wcs=None)
#
# The former creates the normal synthetic WCS.
_DEFAULT_WCS = object()


# ---------------------------------------------------------------------------
# WCS helpers.
# ---------------------------------------------------------------------------


def _create_celestial_wcs() -> WCS:
    """
    Create a deterministic two-dimensional celestial WCS.

    The reference coordinates are placed at the WCS reference position
    so that the expected reference pixel coordinates are deterministic.
    """

    wcs = WCS(
        naxis=2,
    )

    wcs.wcs.crpix = [
        5.0,
        5.0,
    ]

    wcs.wcs.crval = [
        REFERENCE_RA,
        REFERENCE_DEC,
    ]

    wcs.wcs.cdelt = [
        -0.01,
        0.01,
    ]

    wcs.wcs.ctype = [
        "RA---TAN",
        "DEC--TAN",
    ]

    return wcs


def _reference_pixel_position(
    wcs: WCS,
) -> tuple[float, float]:
    """
    Return the WCS-derived pixel position of the configured reference.
    """

    x, y = wcs.world_to_pixel_values(
        REFERENCE_RA,
        REFERENCE_DEC,
    )

    return (
        float(x),
        float(y),
    )


# ---------------------------------------------------------------------------
# Domain-model helpers.
# ---------------------------------------------------------------------------


def _create_reference(
    *,
    identifier: str = REFERENCE_IDENTIFIER,
    right_ascension: float = REFERENCE_RA,
    declination: float = REFERENCE_DEC,
) -> ReferenceStarConfig:
    """
    Create a synthetic reference-star configuration.
    """

    return ReferenceStarConfig(
        identifier=identifier,
        right_ascension=right_ascension,
        declination=declination,
    )


def _create_source(
    *,
    identifier: str,
    x: float,
    y: float,
    peak: float = 100.0,
) -> DetectedSource:
    """
    Create a synthetic detected source.
    """

    return DetectedSource(
        identifier=identifier,
        x_centroid=x,
        y_centroid=y,
        peak_value=peak,
    )


def _create_observation(
    *,
    sources: tuple[DetectedSource, ...] = (),
    wcs: WCS | None | object = _DEFAULT_WCS,
) -> Observation:
    """
    Create a minimal Observation for reference-resolver tests.

    Omitting ``wcs`` creates the normal synthetic celestial WCS.

    Explicitly passing ``wcs=None`` creates an Observation with no WCS.

    Passing a WCS object uses that WCS directly.
    """

    if wcs is _DEFAULT_WCS:
        actual_wcs: WCS | None = _create_celestial_wcs()
    else:
        actual_wcs = wcs  # type: ignore[assignment]

    metadata = ObservationMetadata(
        target_name="Test Target",
        observation_time=TIMESTAMP,
        right_ascension=REFERENCE_RA,
        declination=REFERENCE_DEC,
        wcs=actual_wcs,
    )

    provenance = ObservationProvenance(
        source=ObservationSource.SPACE_BASED,
        survey=Survey.TESS,
        dataset_identifier="TEST-DATASET",
        filename=None,
        download_timestamp=TIMESTAMP,
    )

    image = ImageData(
        image=NDData(
            data=np.zeros(
                (20, 20),
                dtype=np.float64,
            ),
            wcs=actual_wcs,
        ),
    )

    detections = DetectionResult(
        sources=sources,
        algorithm="Test Detector",
        detection_threshold=1.0,
        detection_timestamp=TIMESTAMP,
    )

    return Observation(
        metadata=metadata,
        provenance=provenance,
        original=image,
        active=image,
        images={
            "raw": image,
        },
        detections=detections,
    )


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matching_tolerance",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_matching_tolerance_is_rejected(
    matching_tolerance: float,
) -> None:
    """
    Matching tolerance must be finite and greater than zero.
    """

    with pytest.raises(
        ValueError,
        match="matching tolerance",
    ):
        ReferenceSourceResolver(
            matching_tolerance=matching_tolerance,
        )


def test_valid_matching_tolerance_is_accepted() -> None:
    """
    A finite positive matching tolerance is accepted.
    """

    resolver = ReferenceSourceResolver(
        matching_tolerance=2.0,
    )

    assert resolver is not None


# ---------------------------------------------------------------------------
# Successful resolution.
# ---------------------------------------------------------------------------


def test_reference_is_resolved_at_exact_reference_position() -> None:
    """
    A source exactly at the WCS-derived reference position is matched.
    """

    wcs = _create_celestial_wcs()

    reference_x, reference_y = _reference_pixel_position(
        wcs,
    )

    source = _create_source(
        identifier="reference",
        x=reference_x,
        y=reference_y,
    )

    observation = _create_observation(
        sources=(source,),
        wcs=wcs,
    )

    reference = _create_reference()

    match = ReferenceSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
        reference,
    )

    assert isinstance(
        match,
        ReferenceSourceMatch,
    )

    assert match.reference_identifier == REFERENCE_IDENTIFIER
    assert match.source_identifier == "reference"

    assert match.reference_x == pytest.approx(
        reference_x,
    )

    assert match.reference_y == pytest.approx(
        reference_y,
    )

    assert match.source_x == pytest.approx(
        reference_x,
    )

    assert match.source_y == pytest.approx(
        reference_y,
    )

    assert match.separation_pixels == pytest.approx(
        0.0,
    )


def test_reference_is_resolved_within_matching_tolerance() -> None:
    """
    A source within the configured matching tolerance is accepted.
    """

    wcs = _create_celestial_wcs()

    reference_x, reference_y = _reference_pixel_position(
        wcs,
    )

    source = _create_source(
        identifier="reference",
        x=reference_x + 1.0,
        y=reference_y,
    )

    observation = _create_observation(
        sources=(source,),
        wcs=wcs,
    )

    match = ReferenceSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
        _create_reference(),
    )

    assert match.source_identifier == "reference"

    assert match.separation_pixels == pytest.approx(
        1.0,
    )


def test_source_exactly_at_matching_tolerance_is_accepted() -> None:
    """
    A source exactly on the matching-tolerance boundary is accepted.
    """

    wcs = _create_celestial_wcs()

    reference_x, reference_y = _reference_pixel_position(
        wcs,
    )

    source = _create_source(
        identifier="reference",
        x=reference_x + 2.0,
        y=reference_y,
    )

    observation = _create_observation(
        sources=(source,),
        wcs=wcs,
    )

    match = ReferenceSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
        _create_reference(),
    )

    assert match.source_identifier == "reference"

    assert match.separation_pixels == pytest.approx(
        2.0,
    )


# ---------------------------------------------------------------------------
# Nearest-source selection.
# ---------------------------------------------------------------------------


def test_nearest_detected_source_is_selected() -> None:
    """
    When several sources are detected, the nearest source is selected.
    """

    wcs = _create_celestial_wcs()

    reference_x, reference_y = _reference_pixel_position(
        wcs,
    )

    distant_source = _create_source(
        identifier="distant",
        x=reference_x + 1.5,
        y=reference_y,
    )

    nearest_source = _create_source(
        identifier="nearest",
        x=reference_x + 0.25,
        y=reference_y,
    )

    observation = _create_observation(
        sources=(
            distant_source,
            nearest_source,
        ),
        wcs=wcs,
    )

    match = ReferenceSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
        _create_reference(),
    )

    assert match.source_identifier == "nearest"

    assert match.separation_pixels == pytest.approx(
        0.25,
    )


def test_nearest_source_is_selected_independent_of_detection_order() -> None:
    """
    Detection order must not determine the selected reference source.
    """

    wcs = _create_celestial_wcs()

    reference_x, reference_y = _reference_pixel_position(
        wcs,
    )

    nearest_source = _create_source(
        identifier="nearest",
        x=reference_x + 0.4,
        y=reference_y + 0.3,
    )

    distant_source = _create_source(
        identifier="distant",
        x=reference_x + 1.5,
        y=reference_y,
    )

    observation = _create_observation(
        sources=(
            distant_source,
            nearest_source,
        ),
        wcs=wcs,
    )

    match = ReferenceSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
        _create_reference(),
    )

    assert match.source_identifier == "nearest"

    assert match.separation_pixels == pytest.approx(
        np.hypot(
            0.4,
            0.3,
        ),
    )


# ---------------------------------------------------------------------------
# Resolution failures.
# ---------------------------------------------------------------------------


def test_source_outside_matching_tolerance_is_rejected() -> None:
    """
    A source farther away than the configured tolerance is rejected.
    """

    wcs = _create_celestial_wcs()

    reference_x, reference_y = _reference_pixel_position(
        wcs,
    )

    source = _create_source(
        identifier="too-far",
        x=reference_x + 2.1,
        y=reference_y,
    )

    observation = _create_observation(
        sources=(source,),
        wcs=wcs,
    )

    with pytest.raises(
        ObservationError,
        match="matching tolerance",
    ):
        ReferenceSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
            _create_reference(),
        )


def test_no_detected_sources_is_rejected() -> None:
    """
    Resolution fails when no sources were detected.
    """

    observation = _create_observation(
        sources=(),
    )

    with pytest.raises(
        ObservationError,
        match="detected sources",
    ):
        ReferenceSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
            _create_reference(),
        )


def test_missing_detection_result_is_rejected() -> None:
    """
    Resolution requires a completed DetectionResult.
    """

    observation = _create_observation()

    observation.detections = None

    with pytest.raises(
        ObservationError,
        match="detection",
    ):
        ReferenceSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
            _create_reference(),
        )


# ---------------------------------------------------------------------------
# WCS validation.
# ---------------------------------------------------------------------------


def test_missing_wcs_is_rejected() -> None:
    """
    Resolution requires celestial WCS information.
    """

    observation = _create_observation(
        wcs=None,
    )

    with pytest.raises(
        ObservationError,
        match="WCS",
    ):
        ReferenceSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
            _create_reference(),
        )


def test_non_celestial_wcs_is_rejected() -> None:
    """
    A WCS without celestial coordinate information is rejected.
    """

    wcs = WCS(
        naxis=2,
    )

    observation = _create_observation(
        wcs=wcs,
    )

    with pytest.raises(
        ObservationError,
        match="celestial",
    ):
        ReferenceSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
            _create_reference(),
        )


# ---------------------------------------------------------------------------
# Reference-coordinate validation.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Detected-source validation.
# ---------------------------------------------------------------------------


def test_non_finite_source_centroid_does_not_override_valid_source() -> None:
    """
    An invalid source centroid must not prevent a valid source from
    being selected.
    """

    wcs = _create_celestial_wcs()

    reference_x, reference_y = _reference_pixel_position(
        wcs,
    )

    invalid_source = _create_source(
        identifier="invalid",
        x=float("nan"),
        y=reference_y,
    )

    valid_source = _create_source(
        identifier="valid",
        x=reference_x + 0.5,
        y=reference_y,
    )

    observation = _create_observation(
        sources=(
            invalid_source,
            valid_source,
        ),
        wcs=wcs,
    )

    match = ReferenceSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
        _create_reference(),
    )

    assert match.source_identifier == "valid"

    assert match.separation_pixels == pytest.approx(
        0.5,
    )


def test_all_invalid_source_centroids_are_rejected() -> None:
    """
    Resolution fails when no detected source has usable centroid
    coordinates.
    """

    observation = _create_observation(
        sources=(
            _create_source(
                identifier="invalid-x",
                x=float("nan"),
                y=1.0,
            ),
            _create_source(
                identifier="invalid-y",
                x=2.0,
                y=float("inf"),
            ),
        ),
    )

    with pytest.raises(
        ObservationError,
        match="centroid",
    ):
        ReferenceSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
            _create_reference(),
        )


# ---------------------------------------------------------------------------
# ReferenceSourceMatch integrity.
# ---------------------------------------------------------------------------


def test_match_contains_reference_and_source_coordinates() -> None:
    """
    A successful match preserves both the expected reference position
    and the actual detected-source position.
    """

    wcs = _create_celestial_wcs()

    reference_x, reference_y = _reference_pixel_position(
        wcs,
    )

    source_x = reference_x + 0.6
    source_y = reference_y - 0.8

    source = _create_source(
        identifier="reference-source",
        x=source_x,
        y=source_y,
    )

    observation = _create_observation(
        sources=(source,),
        wcs=wcs,
    )

    match = ReferenceSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
        _create_reference(),
    )

    assert match.reference_identifier == REFERENCE_IDENTIFIER
    assert match.source_identifier == "reference-source"

    assert match.reference_x == pytest.approx(
        reference_x,
    )

    assert match.reference_y == pytest.approx(
        reference_y,
    )

    assert match.source_x == pytest.approx(
        source_x,
    )

    assert match.source_y == pytest.approx(
        source_y,
    )

    assert match.separation_pixels == pytest.approx(
        1.0,
    )