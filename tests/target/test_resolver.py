"""
Unit tests for TargetSourceResolver.

The resolver performs source-level target association:

    configured RA/Dec
          ↓
      WCS transform
          ↓
    target pixel position
          ↓
    nearest detected source
          ↓
    TargetSourceMatch

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
from domain.enums import ObservationSource, Survey
from domain.models import (
    DetectedSource,
    DetectionResult,
    ImageData,
    Observation,
    ObservationMetadata,
    ObservationProvenance,
)
from target.resolver import TargetSourceResolver


# ---------------------------------------------------------------------------
# Test constants.
# ---------------------------------------------------------------------------

TARGET_NAME = "Test Target"

TARGET_RA = 100.0
TARGET_DEC = -30.0

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
# The former should create the normal synthetic WCS.
# The latter should explicitly create an Observation with no WCS.
_DEFAULT_WCS = object()


# ---------------------------------------------------------------------------
# WCS helpers.
# ---------------------------------------------------------------------------


def _create_celestial_wcs() -> WCS:
    """
    Create a deterministic two-dimensional celestial WCS.

    The target coordinates are placed at the WCS reference position so
    that the expected target pixel coordinates are deterministic.
    """

    wcs = WCS(
        naxis=2,
    )

    wcs.wcs.crpix = [
        5.0,
        5.0,
    ]

    wcs.wcs.crval = [
        TARGET_RA,
        TARGET_DEC,
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


def _target_pixel_position(
    wcs: WCS,
) -> tuple[float, float]:
    """
    Return the WCS-derived pixel position of the configured target.
    """

    x, y = wcs.world_to_pixel_values(
        TARGET_RA,
        TARGET_DEC,
    )

    return (
        float(x),
        float(y),
    )


# ---------------------------------------------------------------------------
# Domain-model helpers.
# ---------------------------------------------------------------------------


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
    right_ascension: float | None = TARGET_RA,
    declination: float | None = TARGET_DEC,
) -> Observation:
    """
    Create a minimal Observation for resolver tests.

    Omitting ``wcs`` creates the normal synthetic celestial WCS.

    Explicitly passing ``wcs=None`` creates an Observation with
    no WCS, allowing missing-WCS validation to be tested.

    Passing a WCS object uses that WCS directly.
    """

    if wcs is _DEFAULT_WCS:
        actual_wcs: WCS | None = _create_celestial_wcs()
    else:
        actual_wcs = wcs  # type: ignore[assignment]

    metadata = ObservationMetadata(
        target_name=TARGET_NAME,
        observation_time=TIMESTAMP,
        right_ascension=right_ascension,
        declination=declination,
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
        TargetSourceResolver(
            matching_tolerance=matching_tolerance,
        )


def test_valid_matching_tolerance_is_accepted() -> None:
    """
    A finite positive matching tolerance is accepted.
    """

    resolver = TargetSourceResolver(
        matching_tolerance=2.0,
    )

    assert resolver is not None


# ---------------------------------------------------------------------------
# Successful resolution.
# ---------------------------------------------------------------------------


def test_target_is_resolved_at_exact_target_position() -> None:
    """
    A source exactly at the WCS-derived target position is matched.
    """

    wcs = _create_celestial_wcs()

    target_x, target_y = _target_pixel_position(
        wcs,
    )

    source = _create_source(
        identifier="target",
        x=target_x,
        y=target_y,
    )

    observation = _create_observation(
        sources=(source,),
        wcs=wcs,
    )

    match = TargetSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
    )

    assert match.target_name == TARGET_NAME
    assert match.source_identifier == "target"

    assert match.target_x == pytest.approx(
        target_x,
    )

    assert match.target_y == pytest.approx(
        target_y,
    )

    assert match.source_x == pytest.approx(
        target_x,
    )

    assert match.source_y == pytest.approx(
        target_y,
    )

    assert match.separation_pixels == pytest.approx(
        0.0,
    )


def test_target_is_resolved_within_matching_tolerance() -> None:
    """
    A source within the configured matching tolerance is accepted.
    """

    wcs = _create_celestial_wcs()

    target_x, target_y = _target_pixel_position(
        wcs,
    )

    source = _create_source(
        identifier="target",
        x=target_x + 1.0,
        y=target_y,
    )

    observation = _create_observation(
        sources=(source,),
        wcs=wcs,
    )

    match = TargetSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
    )

    assert match.source_identifier == "target"

    assert match.separation_pixels == pytest.approx(
        1.0,
    )


def test_source_exactly_at_matching_tolerance_is_accepted() -> None:
    """
    A source exactly on the matching-tolerance boundary is accepted.
    """

    wcs = _create_celestial_wcs()

    target_x, target_y = _target_pixel_position(
        wcs,
    )

    source = _create_source(
        identifier="target",
        x=target_x + 2.0,
        y=target_y,
    )

    observation = _create_observation(
        sources=(source,),
        wcs=wcs,
    )

    match = TargetSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
    )

    assert match.source_identifier == "target"

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

    target_x, target_y = _target_pixel_position(
        wcs,
    )

    distant_source = _create_source(
        identifier="distant",
        x=target_x + 1.5,
        y=target_y,
    )

    nearest_source = _create_source(
        identifier="nearest",
        x=target_x + 0.25,
        y=target_y,
    )

    observation = _create_observation(
        sources=(
            distant_source,
            nearest_source,
        ),
        wcs=wcs,
    )

    match = TargetSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
    )

    assert match.source_identifier == "nearest"

    assert match.separation_pixels == pytest.approx(
        0.25,
    )


def test_nearest_source_is_selected_independent_of_detection_order() -> None:
    """
    Detection order must not determine the selected target source.
    """

    wcs = _create_celestial_wcs()

    target_x, target_y = _target_pixel_position(
        wcs,
    )

    nearest_source = _create_source(
        identifier="nearest",
        x=target_x + 0.4,
        y=target_y + 0.3,
    )

    distant_source = _create_source(
        identifier="distant",
        x=target_x + 1.5,
        y=target_y,
    )

    observation = _create_observation(
        sources=(
            distant_source,
            nearest_source,
        ),
        wcs=wcs,
    )

    match = TargetSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
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

    target_x, target_y = _target_pixel_position(
        wcs,
    )

    source = _create_source(
        identifier="too-far",
        x=target_x + 2.1,
        y=target_y,
    )

    observation = _create_observation(
        sources=(source,),
        wcs=wcs,
    )

    with pytest.raises(
        ObservationError,
        match="matching tolerance",
    ):
        TargetSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
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
        TargetSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
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
        TargetSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
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
        TargetSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
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
        TargetSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
        )


# ---------------------------------------------------------------------------
# Target-coordinate validation.
# ---------------------------------------------------------------------------


def test_missing_right_ascension_is_rejected() -> None:
    """
    Resolution requires a target right ascension.
    """

    observation = _create_observation(
        right_ascension=None,
    )

    with pytest.raises(
        ObservationError,
        match="right ascension",
    ):
        TargetSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
        )


def test_missing_declination_is_rejected() -> None:
    """
    Resolution requires a target declination.
    """

    observation = _create_observation(
        declination=None,
    )

    with pytest.raises(
        ObservationError,
        match="declination",
    ):
        TargetSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
        )


@pytest.mark.parametrize(
    "right_ascension",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_right_ascension_is_rejected(
    right_ascension: float,
) -> None:
    """
    Non-finite right ascension must not be passed into the WCS transform.
    """

    observation = _create_observation(
        right_ascension=right_ascension,
    )

    with pytest.raises(
        ObservationError,
        match="right ascension",
    ):
        TargetSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
        )


@pytest.mark.parametrize(
    "declination",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_declination_is_rejected(
    declination: float,
) -> None:
    """
    Non-finite declination must not be passed into the WCS transform.
    """

    observation = _create_observation(
        declination=declination,
    )

    with pytest.raises(
        ObservationError,
        match="declination",
    ):
        TargetSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
        )


# ---------------------------------------------------------------------------
# Detected-source validation.
# ---------------------------------------------------------------------------


def test_non_finite_source_centroid_does_not_override_valid_source() -> None:
    """
    An invalid source centroid must not prevent a valid source from
    being selected.
    """

    wcs = _create_celestial_wcs()

    target_x, target_y = _target_pixel_position(
        wcs,
    )

    invalid_source = _create_source(
        identifier="invalid",
        x=float("nan"),
        y=target_y,
    )

    valid_source = _create_source(
        identifier="valid",
        x=target_x + 0.5,
        y=target_y,
    )

    observation = _create_observation(
        sources=(
            invalid_source,
            valid_source,
        ),
        wcs=wcs,
    )

    match = TargetSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
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
        TargetSourceResolver(
            matching_tolerance=2.0,
        ).resolve(
            observation,
        )


# ---------------------------------------------------------------------------
# TargetSourceMatch integrity.
# ---------------------------------------------------------------------------


def test_match_contains_target_and_source_coordinates() -> None:
    """
    A successful match preserves both the expected target position and
    the actual detected-source position.
    """

    wcs = _create_celestial_wcs()

    target_x, target_y = _target_pixel_position(
        wcs,
    )

    source_x = target_x + 0.6
    source_y = target_y - 0.8

    source = _create_source(
        identifier="target-source",
        x=source_x,
        y=source_y,
    )

    observation = _create_observation(
        sources=(source,),
        wcs=wcs,
    )

    match = TargetSourceResolver(
        matching_tolerance=2.0,
    ).resolve(
        observation,
    )

    assert match.target_name == TARGET_NAME
    assert match.source_identifier == "target-source"

    assert match.target_x == pytest.approx(
        target_x,
    )

    assert match.target_y == pytest.approx(
        target_y,
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