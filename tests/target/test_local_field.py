"""
Unit tests for target-centred local-field selection.

The tests verify the scientific contract of LocalFieldSelector:

    complete detection catalogue
            +
       resolved target
            ↓
       target-centred
        LocalField

The selector must:

    - always include the target;
    - exclude the target from the neighbour list;
    - select nearest neighbours by pixel distance;
    - respect max_sources including the target;
    - remain deterministic;
    - tolerate fewer available sources than max_sources;
    - reject invalid or inconsistent inputs.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.exceptions import ObservationError
from domain.models import (
    DetectedSource,
    DetectionResult,
    TargetSourceMatch,
)
from target.local_field import (
    LocalField,
    LocalFieldSelector,
)


# ============================================================================
# Test helpers.
# ============================================================================


def make_source(
    identifier: str,
    x: float,
    y: float,
    peak: float = 100.0,
) -> DetectedSource:
    """Create a DetectedSource for testing."""

    return DetectedSource(
        identifier=identifier,
        x_centroid=x,
        y_centroid=y,
        peak_value=peak,
    )


def make_detection_result(
    *sources: DetectedSource,
) -> DetectionResult:
    """Create a DetectionResult containing the supplied sources."""

    return DetectionResult(
        sources=tuple(sources),
        algorithm="test",
        detection_threshold=5.0,
        detection_timestamp=datetime.now(
            timezone.utc,
        ),
    )


def make_target_match(
    source_identifier: str,
    *,
    target_name: str = "Test Target",
    target_x: float = 0.0,
    target_y: float = 0.0,
    source_x: float = 0.0,
    source_y: float = 0.0,
    separation_pixels: float = 0.0,
) -> TargetSourceMatch:
    """Create a TargetSourceMatch for testing."""

    return TargetSourceMatch(
        target_name=target_name,
        source_identifier=source_identifier,
        target_x=target_x,
        target_y=target_y,
        source_x=source_x,
        source_y=source_y,
        separation_pixels=separation_pixels,
    )


# ============================================================================
# Constructor validation.
# ============================================================================


class TestLocalFieldSelectorConstruction:
    """Tests for LocalFieldSelector construction."""

    def test_accepts_positive_max_sources(self) -> None:
        selector = LocalFieldSelector(
            max_sources=50,
        )

        assert selector.max_sources == 50

    @pytest.mark.parametrize(
        "max_sources",
        [
            0,
            -1,
            -10,
        ],
    )
    def test_rejects_non_positive_max_sources(
        self,
        max_sources: int,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="max_sources must be greater than zero",
        ):
            LocalFieldSelector(
                max_sources=max_sources,
            )

    def test_rejects_non_integer_max_sources(self) -> None:
        with pytest.raises(
            TypeError,
            match="max_sources must be an integer",
        ):
            LocalFieldSelector(
                max_sources=50.0,  # type: ignore[arg-type]
            )


# ============================================================================
# Basic selection behaviour.
# ============================================================================


class TestLocalFieldSelection:
    """Tests for target-centred source selection."""

    def test_selects_target_and_nearest_neighbours(self) -> None:
        """
        The target must be included and the remaining slots filled by
        the nearest detected neighbours.
        """

        target = make_source(
            "target",
            x=0.0,
            y=0.0,
        )

        source_a = make_source(
            "source_a",
            x=1.0,
            y=0.0,
        )

        source_b = make_source(
            "source_b",
            x=2.0,
            y=0.0,
        )

        source_c = make_source(
            "source_c",
            x=5.0,
            y=0.0,
        )

        source_d = make_source(
            "source_d",
            x=10.0,
            y=0.0,
        )

        detections = make_detection_result(
            source_d,
            source_c,
            target,
            source_b,
            source_a,
        )

        target_match = make_target_match(
            "target",
        )

        selector = LocalFieldSelector(
            max_sources=3,
        )

        local_field = selector.select(
            detections=detections,
            target_match=target_match,
        )

        assert local_field.sources == (
            target,
            source_a,
            source_b,
        )

        assert local_field.selected_source_count == 3
        assert local_field.selected_neighbour_count == 2

    def test_target_is_always_first(self) -> None:
        """The resolved target must be the first source in the field."""

        target = make_source(
            "target",
            x=50.0,
            y=50.0,
        )

        neighbour = make_source(
            "neighbour",
            x=51.0,
            y=50.0,
        )

        detections = make_detection_result(
            neighbour,
            target,
        )

        target_match = make_target_match(
            "target",
            source_x=50.0,
            source_y=50.0,
        )

        local_field = LocalFieldSelector(
            max_sources=10,
        ).select(
            detections=detections,
            target_match=target_match,
        )

        assert local_field.sources[0] == target
        assert local_field.target_source == target

    def test_target_is_not_duplicated_as_neighbour(self) -> None:
        """The target must not also appear in the neighbour collection."""

        target = make_source(
            "target",
            x=0.0,
            y=0.0,
        )

        neighbour = make_source(
            "neighbour",
            x=1.0,
            y=0.0,
        )

        detections = make_detection_result(
            target,
            neighbour,
        )

        target_match = make_target_match(
            "target",
        )

        local_field = LocalFieldSelector(
            max_sources=10,
        ).select(
            detections=detections,
            target_match=target_match,
        )

        assert local_field.neighbour_sources == (
            neighbour,
        )

        assert local_field.sources.count(target) == 1


# ============================================================================
# max_sources semantics.
# ============================================================================


class TestLocalFieldSize:
    """Tests for max_sources semantics."""

    def test_max_sources_includes_target(self) -> None:
        """
        max_sources=1 means target only.

        It does not mean target + one neighbour.
        """

        target = make_source(
            "target",
            x=0.0,
            y=0.0,
        )

        neighbour = make_source(
            "neighbour",
            x=1.0,
            y=0.0,
        )

        detections = make_detection_result(
            target,
            neighbour,
        )

        target_match = make_target_match(
            "target",
        )

        local_field = LocalFieldSelector(
            max_sources=1,
        ).select(
            detections=detections,
            target_match=target_match,
        )

        assert local_field.sources == (
            target,
        )

        assert local_field.selected_source_count == 1
        assert local_field.selected_neighbour_count == 0
        assert local_field.selection_radius_pixels == 0.0

    def test_selects_all_available_sources_when_fewer_than_maximum(
        self,
    ) -> None:
        """The selector must not require max_sources sources."""

        target = make_source(
            "target",
            x=0.0,
            y=0.0,
        )

        source_a = make_source(
            "source_a",
            x=1.0,
            y=0.0,
        )

        source_b = make_source(
            "source_b",
            x=0.0,
            y=2.0,
        )

        detections = make_detection_result(
            target,
            source_a,
            source_b,
        )

        target_match = make_target_match(
            "target",
        )

        local_field = LocalFieldSelector(
            max_sources=50,
        ).select(
            detections=detections,
            target_match=target_match,
        )

        assert local_field.selected_source_count == 3
        assert local_field.selected_neighbour_count == 2
        assert local_field.available_source_count == 3

    def test_maximum_of_fifty_means_target_plus_49_neighbours(
        self,
    ) -> None:
        """Verify the intended 50-source experimental configuration."""

        sources = [
            make_source(
                "source_00",
                x=0.0,
                y=0.0,
            )
        ]

        for index in range(1, 61):
            sources.append(
                make_source(
                    f"source_{index:02d}",
                    x=float(index),
                    y=0.0,
                )
            )

        detections = make_detection_result(
            *sources,
        )

        target_match = make_target_match(
            "source_00",
        )

        local_field = LocalFieldSelector(
            max_sources=50,
        ).select(
            detections=detections,
            target_match=target_match,
        )

        assert local_field.selected_source_count == 50
        assert local_field.selected_neighbour_count == 49

        assert local_field.sources[0].identifier == "source_00"

        assert (
            local_field.sources[-1].identifier
            == "source_49"
        )


# ============================================================================
# Distance ordering.
# ============================================================================


class TestLocalFieldDistanceOrdering:
    """Tests for target-relative pixel-distance ordering."""

    def test_neighbours_are_ordered_by_pixel_distance(self) -> None:
        target = make_source(
            "target",
            x=10.0,
            y=10.0,
        )

        source_far = make_source(
            "far",
            x=16.0,
            y=10.0,
        )

        source_near = make_source(
            "near",
            x=11.0,
            y=10.0,
        )

        source_middle = make_source(
            "middle",
            x=13.0,
            y=10.0,
        )

        detections = make_detection_result(
            source_far,
            source_middle,
            target,
            source_near,
        )

        target_match = make_target_match(
            "target",
            source_x=10.0,
            source_y=10.0,
        )

        local_field = LocalFieldSelector(
            max_sources=10,
        ).select(
            detections=detections,
            target_match=target_match,
        )

        assert tuple(
            source.identifier
            for source in local_field.neighbour_sources
        ) == (
            "near",
            "middle",
            "far",
        )

        assert local_field.neighbour_distances_pixels == (
            1.0,
            3.0,
            6.0,
        )

    def test_uses_euclidean_pixel_distance(self) -> None:
        """
        Verify that distance is calculated as:

            sqrt(dx² + dy²)
        """

        target = make_source(
            "target",
            x=0.0,
            y=0.0,
        )

        neighbour = make_source(
            "neighbour",
            x=3.0,
            y=4.0,
        )

        detections = make_detection_result(
            target,
            neighbour,
        )

        target_match = make_target_match(
            "target",
        )

        local_field = LocalFieldSelector(
            max_sources=10,
        ).select(
            detections=detections,
            target_match=target_match,
        )

        assert local_field.neighbour_distances_pixels == (
            5.0,
        )

        assert local_field.selection_radius_pixels == 5.0


# ============================================================================
# Determinism.
# ============================================================================


class TestLocalFieldDeterminism:
    """Tests ensuring selection is reproducible."""

    def test_equal_distance_sources_use_identifier_as_tie_breaker(
        self,
    ) -> None:
        """
        Sources at equal distance must have a deterministic ordering.
        """

        target = make_source(
            "target",
            x=0.0,
            y=0.0,
        )

        source_b = make_source(
            "source_b",
            x=1.0,
            y=0.0,
        )

        source_a = make_source(
            "source_a",
            x=-1.0,
            y=0.0,
        )

        detections = make_detection_result(
            source_b,
            target,
            source_a,
        )

        target_match = make_target_match(
            "target",
        )

        local_field = LocalFieldSelector(
            max_sources=3,
        ).select(
            detections=detections,
            target_match=target_match,
        )

        assert tuple(
            source.identifier
            for source in local_field.neighbour_sources
        ) == (
            "source_a",
            "source_b",
        )

    def test_selection_is_independent_of_detection_catalogue_order(
        self,
    ) -> None:
        """Reordering detections must not change the selected field."""

        target = make_source(
            "target",
            x=0.0,
            y=0.0,
        )

        source_a = make_source(
            "source_a",
            x=3.0,
            y=0.0,
        )

        source_b = make_source(
            "source_b",
            x=1.0,
            y=0.0,
        )

        source_c = make_source(
            "source_c",
            x=2.0,
            y=0.0,
        )

        target_match = make_target_match(
            "target",
        )

        first = LocalFieldSelector(
            max_sources=3,
        ).select(
            detections=make_detection_result(
                source_c,
                target,
                source_a,
                source_b,
            ),
            target_match=target_match,
        )

        second = LocalFieldSelector(
            max_sources=3,
        ).select(
            detections=make_detection_result(
                source_b,
                source_a,
                target,
                source_c,
            ),
            target_match=target_match,
        )

        assert first.sources == second.sources
        assert (
            first.neighbour_distances_pixels
            == second.neighbour_distances_pixels
        )


# ============================================================================
# Target resolution consistency.
# ============================================================================


class TestLocalFieldTargetResolution:
    """Tests for consistency with TargetSourceMatch."""

    def test_uses_resolved_source_identifier(self) -> None:
        """
        LocalFieldSelector must use the source identified by
        TargetSourceMatch rather than assuming the target is the first
        detection.
        """

        unrelated = make_source(
            "source_1",
            x=100.0,
            y=100.0,
        )

        target = make_source(
            "target_source",
            x=10.0,
            y=10.0,
        )

        detections = make_detection_result(
            unrelated,
            target,
        )

        target_match = make_target_match(
            "target_source",
            source_x=10.0,
            source_y=10.0,
        )

        local_field = LocalFieldSelector(
            max_sources=10,
        ).select(
            detections=detections,
            target_match=target_match,
        )

        assert (
            local_field.target_source.identifier
            == "target_source"
        )

        assert (
            local_field.target_source.x_centroid
            == 10.0
        )

        assert (
            local_field.target_source.y_centroid
            == 10.0
        )


# ============================================================================
# Invalid inputs.
# ============================================================================


class TestLocalFieldValidation:
    """Tests for invalid or inconsistent selection inputs."""

    def test_rejects_none_detection_result(self) -> None:
        selector = LocalFieldSelector(
            max_sources=10,
        )

        target_match = make_target_match(
            "target",
        )

        with pytest.raises(
            ObservationError,
            match="requires completed source detection",
        ):
            selector.select(
                detections=None,  # type: ignore[arg-type]
                target_match=target_match,
            )

    def test_rejects_none_target_match(self) -> None:
        target = make_source(
            "target",
            x=0.0,
            y=0.0,
        )

        detections = make_detection_result(
            target,
        )

        selector = LocalFieldSelector(
            max_sources=10,
        )

        with pytest.raises(
            ObservationError,
            match="requires a resolved target source",
        ):
            selector.select(
                detections=detections,
                target_match=None,  # type: ignore[arg-type]
            )

    def test_rejects_empty_detection_catalogue(self) -> None:
        detections = make_detection_result()

        target_match = make_target_match(
            "target",
        )

        selector = LocalFieldSelector(
            max_sources=10,
        )

        with pytest.raises(
            ObservationError,
            match="detection catalogue is empty",
        ):
            selector.select(
                detections=detections,
                target_match=target_match,
            )

    def test_rejects_target_missing_from_detection_catalogue(
        self,
    ) -> None:
        detections = make_detection_result(
            make_source(
                "source_1",
                x=1.0,
                y=1.0,
            ),
        )

        target_match = make_target_match(
            "missing_target",
        )

        selector = LocalFieldSelector(
            max_sources=10,
        )

        with pytest.raises(
            ObservationError,
            match="is not present in the current detection catalogue",
        ):
            selector.select(
                detections=detections,
                target_match=target_match,
            )

    def test_rejects_duplicate_target_identifier(self) -> None:
        target_a = make_source(
            "target",
            x=0.0,
            y=0.0,
        )

        target_b = make_source(
            "target",
            x=1.0,
            y=1.0,
        )

        detections = make_detection_result(
            target_a,
            target_b,
        )

        target_match = make_target_match(
            "target",
        )

        selector = LocalFieldSelector(
            max_sources=10,
        )

        with pytest.raises(
            ObservationError,
            match="contains multiple sources",
        ):
            selector.select(
                detections=detections,
                target_match=target_match,
            )

    def test_ignores_non_finite_neighbour_coordinates(self) -> None:
        target = make_source(
            "target",
            x=0.0,
            y=0.0,
        )

        valid_neighbour = make_source(
            "valid",
            x=2.0,
            y=0.0,
        )

        invalid_neighbour = make_source(
            "invalid",
            x=float("nan"),
            y=1.0,
        )

        detections = make_detection_result(
            target,
            invalid_neighbour,
            valid_neighbour,
        )

        target_match = make_target_match(
            "target",
        )

        local_field = LocalFieldSelector(
            max_sources=10,
        ).select(
            detections=detections,
            target_match=target_match,
        )

        assert local_field.sources == (
            target,
            valid_neighbour,
        )

        assert local_field.available_source_count == 2