"""
Target-source resolution.

Provides survey-independent resolution of the configured astronomical
target to a detected source using celestial coordinates and image WCS.

The resolver does not perform source detection or photometry. It only
determines which detected source most closely corresponds to the
configured astronomical target.
"""

from __future__ import annotations

import logging
import math

from core.exceptions import ObservationError
from domain.models import Observation, TargetSourceMatch
from utils.logger import get_logger


class TargetSourceResolver:
    """
    Resolve the configured astronomical target to a detected source.

    The target's right ascension and declination are transformed into
    image pixel coordinates using the Observation WCS. The nearest
    detected source is then selected if its positional separation is
    within the configured matching tolerance.

    Parameters
    ----------
    matching_tolerance
        Maximum permitted separation between the expected target
        position and a detected source centroid, in pixels.

    logger
        Optional logger used for diagnostic messages.
    """

    def __init__(
        self,
        matching_tolerance: float,
        logger: logging.Logger | None = None,
    ) -> None:
        if (
            not math.isfinite(matching_tolerance)
            or matching_tolerance <= 0
        ):
            raise ValueError(
                "Target-source matching tolerance must be a "
                "finite value greater than zero."
            )

        self._matching_tolerance = matching_tolerance
        self._logger = logger or get_logger(__name__)

    def resolve(
        self,
        observation: Observation,
    ) -> TargetSourceMatch:
        """
        Resolve the configured target to the nearest detected source.

        Parameters
        ----------
        observation
            Observation containing target coordinates, WCS and
            detected source positions.

        Returns
        -------
        TargetSourceMatch
            Association between the configured target and the
            nearest detected source within the configured tolerance.

        Raises
        ------
        ObservationError
            If the observation does not contain the information
            required for target resolution, or if no valid detected
            source lies within the permitted matching tolerance.
        """

        self._logger.info(
            "Resolving target '%s' against detected sources.",
            observation.metadata.target_name,
        )

        target_x, target_y = self._target_pixel_position(
            observation,
        )

        detection_result = observation.detections

        if detection_result is None:
            raise ObservationError(
                "Target resolution requires completed source detection."
            )

        if not detection_result.sources:
            raise ObservationError(
                "Target resolution cannot proceed because no "
                "detected sources are available."
            )

        nearest_source = None
        nearest_separation = math.inf

        for source in detection_result.sources:
            if (
                not math.isfinite(source.x_centroid)
                or not math.isfinite(source.y_centroid)
            ):
                self._logger.debug(
                    "Skipping source '%s' because its centroid "
                    "coordinates are not finite.",
                    source.identifier,
                )
                continue

            separation = math.hypot(
                source.x_centroid - target_x,
                source.y_centroid - target_y,
            )

            if separation < nearest_separation:
                nearest_separation = separation
                nearest_source = source

        if nearest_source is None:
            raise ObservationError(
                "Target resolution failed because no detected source "
                "with valid centroid coordinates was available."
            )

        self._logger.info(
            "Expected target position: (%.3f, %.3f) pixels.",
            target_x,
            target_y,
        )

        self._logger.info(
            "Nearest detected source: '%s'.",
            nearest_source.identifier,
        )

        self._logger.info(
            "Target-source separation: %.3f pixels.",
            nearest_separation,
        )

        if nearest_separation > self._matching_tolerance:
            raise ObservationError(
                "No detected source was found within the configured "
                f"target matching tolerance of "
                f"{self._matching_tolerance:.3f} pixels. "
                f"Nearest source '{nearest_source.identifier}' is "
                f"{nearest_separation:.3f} pixels from the expected "
                "target position."
            )

        match = TargetSourceMatch(
            target_name=observation.metadata.target_name,
            source_identifier=nearest_source.identifier,
            target_x=target_x,
            target_y=target_y,
            source_x=nearest_source.x_centroid,
            source_y=nearest_source.y_centroid,
            separation_pixels=nearest_separation,
        )

        self._logger.info(
            "Target '%s' resolved to source '%s' "
            "(separation=%.3f pixels).",
            match.target_name,
            match.source_identifier,
            match.separation_pixels,
        )

        return match

    def _target_pixel_position(
        self,
        observation: Observation,
    ) -> tuple[float, float]:
        """
        Convert the target celestial coordinates to image pixels.

        Raises
        ------
        ObservationError
            If target coordinates or a suitable celestial WCS are
            unavailable, or if the coordinate transformation fails.
        """

        right_ascension = observation.metadata.right_ascension
        declination = observation.metadata.declination
        wcs = observation.metadata.wcs

        if right_ascension is None:
            raise ObservationError(
                f"Target '{observation.metadata.target_name}' has no "
                "right ascension."
            )

        if declination is None:
            raise ObservationError(
                f"Target '{observation.metadata.target_name}' has no "
                "declination."
            )

        if not math.isfinite(right_ascension):
            raise ObservationError(
                f"Target '{observation.metadata.target_name}' has an "
                "invalid right ascension."
            )

        if not math.isfinite(declination):
            raise ObservationError(
                f"Target '{observation.metadata.target_name}' has an "
                "invalid declination."
            )

        if wcs is None:
            raise ObservationError(
                f"Observation for target "
                f"'{observation.metadata.target_name}' has no WCS."
            )

        if not wcs.has_celestial:
            raise ObservationError(
                f"Observation WCS for target "
                f"'{observation.metadata.target_name}' does not "
                "contain celestial coordinate information."
            )

        try:
            x, y = wcs.world_to_pixel_values(
                right_ascension,
                declination,
            )
        except Exception as exc:
            raise ObservationError(
                f"Failed to transform target coordinates for "
                f"'{observation.metadata.target_name}' into "
                "image pixel coordinates."
            ) from exc

        x = float(x)
        y = float(y)

        if not math.isfinite(x) or not math.isfinite(y):
            raise ObservationError(
                f"Target coordinate transformation produced an "
                f"invalid pixel position for "
                f"'{observation.metadata.target_name}'."
            )

        return x, y