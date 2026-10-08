"""
Target-centred local-field selection.

This module defines the survey-independent experimental source population
used by both Baseline and SCARLET processing.

The LocalFieldSelector does not perform source detection, target resolution,
photometry, or SCARLET processing. It operates only on an existing detection
catalogue and an already-resolved target source.

Selection semantics
-------------------
The local field always contains:

    target + nearest neighbouring detected sources

The configured ``max_sources`` value includes the target itself.

For example:

    max_sources = 50

means:

    1 target + up to 49 neighbours

Source distances are measured in image pixels using the detected target
centroid as the centre of the local field.

The result is deterministic:

* the target is always first;
* neighbours are sorted by increasing distance;
* equal-distance neighbours are ordered by source identifier.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from core.exceptions import ObservationError
from domain.models import (
    DetectedSource,
    DetectionResult,
    LocalField,
    TargetSourceMatch,
)
from utils.logger import get_logger


# ---------------------------------------------------------------------------
# Local-field selector
# ---------------------------------------------------------------------------


class LocalFieldSelector:
    """
    Select a deterministic target-centred local field.

    The selector operates on the complete source-detection catalogue and
    an already-resolved target. It does not perform target resolution itself.

    Parameters
    ----------
    max_sources
        Maximum number of sources in the local field, including the target.

    logger
        Optional logger used for diagnostics.

    Selection algorithm
    -------------------
    1. Validate the detection result.
    2. Locate the resolved target source by identifier.
    3. Validate source coordinates.
    4. Exclude the target from neighbour candidates.
    5. Calculate Euclidean pixel distance from every candidate to the
       detected target centroid.
    6. Sort by distance, with source identifier as a deterministic tie-break.
    7. Select at most ``max_sources - 1`` neighbours.
    8. Return the target plus selected neighbours.
    """

    def __init__(
        self,
        max_sources: int,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(max_sources, int):
            raise TypeError(
                "Local-field max_sources must be an integer."
            )

        if max_sources <= 0:
            raise ValueError(
                "Local-field max_sources must be greater than zero."
            )

        self._max_sources = max_sources
        self._logger = logger or get_logger(__name__)

    @property
    def max_sources(self) -> int:
        """Return the configured maximum local-field size."""

        return self._max_sources

    def select(
        self,
        *,
        detections: DetectionResult,
        target_match: TargetSourceMatch,
    ) -> LocalField:
        """
        Construct the target-centred local field.

        Parameters
        ----------
        detections
            Complete source-detection result for the observation.

        target_match
            Previously resolved association between the configured target
            and its detected source.

        Returns
        -------
        LocalField
            Deterministic target-centred source population.

        Raises
        ------
        ObservationError
            If the detection catalogue is empty, the resolved target cannot
            be found, or no valid source coordinates are available.
        """

        if detections is None:
            raise ObservationError(
                "Local-field selection requires completed source detection."
            )

        if target_match is None:
            raise ObservationError(
                "Local-field selection requires a resolved target source."
            )

        if not detections.sources:
            raise ObservationError(
                "Local-field selection cannot proceed because the "
                "detection catalogue is empty."
            )

        target_source = self._find_target_source(
            detections,
            target_match,
        )

        target_x = float(target_source.x_centroid)
        target_y = float(target_source.y_centroid)

        if not math.isfinite(target_x) or not math.isfinite(target_y):
            raise ObservationError(
                (
                    "Resolved target source "
                    f"'{target_source.identifier}' has non-finite "
                    "centroid coordinates."
                )
            )

        candidates: list[
            tuple[float, str, DetectedSource]
        ] = []

        valid_source_count = 0

        for source in detections.sources:
            x = float(source.x_centroid)
            y = float(source.y_centroid)

            if not math.isfinite(x) or not math.isfinite(y):
                self._logger.debug(
                    (
                        "Skipping source '%s' from local-field "
                        "selection because its centroid coordinates "
                        "are not finite."
                    ),
                    source.identifier,
                )
                continue

            valid_source_count += 1

            # The resolved target is included explicitly and must not
            # also appear as a neighbour.
            if source.identifier == target_source.identifier:
                continue

            distance = math.hypot(
                x - target_x,
                y - target_y,
            )

            candidates.append(
                (
                    distance,
                    source.identifier,
                    source,
                )
            )

        if valid_source_count == 0:
            raise ObservationError(
                "Local-field selection failed because no detected "
                "source has valid centroid coordinates."
            )

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

        neighbour_limit = max(
            0,
            self._max_sources - 1,
        )

        selected_candidates = candidates[
            :neighbour_limit
        ]

        neighbour_sources = tuple(
            candidate[2]
            for candidate in selected_candidates
        )

        neighbour_distances = tuple(
            float(candidate[0])
            for candidate in selected_candidates
        )

        local_field = LocalField(
            target_source=target_source,
            neighbour_sources=neighbour_sources,
            neighbour_distances_pixels=neighbour_distances,
            available_source_count=valid_source_count,
            max_sources=self._max_sources,
        )

        self._logger.info(
            (
                "Local field selected for target '%s': "
                "available=%d, selected=%d, neighbours=%d, "
                "max_sources=%d, radius=%.3f pixels."
            ),
            target_match.target_name,
            local_field.available_source_count,
            local_field.selected_source_count,
            local_field.selected_neighbour_count,
            local_field.max_sources,
            local_field.selection_radius_pixels,
        )

        return local_field

    @staticmethod
    def _find_target_source(
        detections: DetectionResult,
        target_match: TargetSourceMatch,
    ) -> DetectedSource:
        """
        Locate the detected source identified by TargetSourceMatch.
        """

        matches = [
            source
            for source in detections.sources
            if source.identifier == target_match.source_identifier
        ]

        if not matches:
            raise ObservationError(
                (
                    "The resolved target source "
                    f"'{target_match.source_identifier}' is not present "
                    "in the current detection catalogue."
                )
            )

        if len(matches) > 1:
            raise ObservationError(
                (
                    "The detection catalogue contains multiple sources "
                    f"with target identifier "
                    f"'{target_match.source_identifier}'."
                )
            )

        return matches[0]