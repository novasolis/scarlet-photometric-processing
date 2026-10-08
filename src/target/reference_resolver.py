"""
Reference-source resolution.

Provides survey-independent resolution of a configured differential-
photometry reference star to a detected source using celestial
coordinates and image WCS.

The resolver does not perform source detection or photometry. It only
determines which detected source most closely corresponds to the
configured reference star.
"""

from __future__ import annotations

import logging
import math

from core.exceptions import ObservationError
from core.models import ReferenceStarConfig
from domain.models import Observation, ReferenceSourceMatch
from utils.logger import get_logger


class ReferenceSourceResolver:
    """
    Resolve a configured reference star to a detected source.

    The reference star's right ascension and declination are transformed
    into image pixel coordinates using the Observation WCS. The nearest
    detected source is then selected if its positional separation is
    within the configured matching tolerance.

    Parameters
    ----------
    matching_tolerance
        Maximum permitted separation between the expected reference-star
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
                "Reference-source matching tolerance must be a "
                "finite value greater than zero."
            )

        self._matching_tolerance = matching_tolerance
        self._logger = logger or get_logger(__name__)

    def resolve(
        self,
        observation: Observation,
        reference: ReferenceStarConfig,
    ) -> ReferenceSourceMatch:
        """
        Resolve a configured reference star to the nearest detected source.

        Parameters
        ----------
        observation
            Observation containing the reference-star coordinates,
            WCS and detected source positions.

        reference
            Configured differential-photometry reference star.

        Returns
        -------
        ReferenceSourceMatch
            Association between the configured reference star and the
            nearest detected source within the configured tolerance.

        Raises
        ------
        ObservationError
            If the observation does not contain the information
            required for reference resolution, or if no valid detected
            source lies within the permitted matching tolerance.
        """

        self._logger.info(
            "Resolving reference star '%s' against detected sources.",
            reference.identifier,
        )

        reference_x, reference_y = self._reference_pixel_position(
            observation,
            reference,
        )
        self._logger.info(
            "REFERENCE DIAGNOSTIC | '%s' | "
            "RA=%.8f deg | Dec=%.8f deg | "
            "expected pixel=(%.3f, %.3f)",
            reference.identifier,
            reference.right_ascension,
            reference.declination,
            reference_x,
            reference_y,
        )

        detection_result = observation.detections

        if detection_result is None:
            raise ObservationError(
                "Reference resolution requires completed source detection."
            )

        if not detection_result.sources:
            raise ObservationError(
                "Reference resolution cannot proceed because no "
                "detected sources are available."
            )

        self._logger.info(
            "REFERENCE DIAGNOSTIC | Detected sources for '%s':",
            reference.identifier,
        )

        for source in detection_result.sources:
            self._logger.info(
                "  source=%s | x=%.3f | y=%.3f | "
                "distance=%.3f px",
                source.identifier,
                source.x_centroid,
                source.y_centroid,
                math.hypot(
                    source.x_centroid - reference_x,
                    source.y_centroid - reference_y,
                ),
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
                source.x_centroid - reference_x,
                source.y_centroid - reference_y,
            )

            if separation < nearest_separation:
                nearest_separation = separation
                nearest_source = source

        if nearest_source is None:
            raise ObservationError(
                "Reference resolution failed because no detected source "
                "with valid centroid coordinates was available."
            )

        self._logger.info(
            "Expected reference position: (%.3f, %.3f) pixels.",
            reference_x,
            reference_y,
        )

        self._logger.info(
            "Nearest detected source: '%s'.",
            nearest_source.identifier,
        )

        self._logger.info(
            "Reference-source separation: %.3f pixels.",
            nearest_separation,
        )

        if nearest_separation > self._matching_tolerance:
            raise ObservationError(
                "No detected source was found within the configured "
                "reference matching tolerance of "
                f"{self._matching_tolerance:.3f} pixels. "
                f"Nearest source '{nearest_source.identifier}' is "
                f"{nearest_separation:.3f} pixels from the expected "
                "reference position."
            )

        match = ReferenceSourceMatch(
            reference_identifier=reference.identifier,
            source_identifier=nearest_source.identifier,
            reference_x=reference_x,
            reference_y=reference_y,
            source_x=nearest_source.x_centroid,
            source_y=nearest_source.y_centroid,
            separation_pixels=nearest_separation,
        )

        self._logger.info(
            "Reference star '%s' resolved to source '%s' "
            "(separation=%.3f pixels).",
            match.reference_identifier,
            match.source_identifier,
            match.separation_pixels,
        )

        return match

    def _reference_pixel_position(
        self,
        observation: Observation,
        reference: ReferenceStarConfig,
    ) -> tuple[float, float]:
        """
        Convert reference-star celestial coordinates to image pixels.

        Parameters
        ----------
        observation
            Observation containing the image WCS.

        reference
            Configured reference star containing celestial coordinates.

        Returns
        -------
        tuple[float, float]
            Expected reference-star position as ``(x, y)`` pixel
            coordinates.

        Raises
        ------
        ObservationError
            If the reference coordinates or a suitable celestial WCS
            are unavailable, or if the coordinate transformation fails.
        """

        right_ascension = reference.right_ascension
        declination = reference.declination
        wcs = observation.metadata.wcs

        if not math.isfinite(right_ascension):
            raise ObservationError(
                f"Reference star '{reference.identifier}' has an "
                "invalid right ascension."
            )

        if not math.isfinite(declination):
            raise ObservationError(
                f"Reference star '{reference.identifier}' has an "
                "invalid declination."
            )

        if wcs is None:
            raise ObservationError(
                "Observation has no WCS required for reference "
                "source resolution."
            )

        if not wcs.has_celestial:
            raise ObservationError(
                "Observation WCS does not contain celestial "
                "coordinate information required for reference "
                "source resolution."
            )

        try:
            x, y = wcs.world_to_pixel_values(
                right_ascension,
                declination,
            )
        except Exception as exc:
            raise ObservationError(
                f"Failed to transform reference star "
                f"'{reference.identifier}' coordinates into "
                "image pixel coordinates."
            ) from exc

        x = float(x)
        y = float(y)

        if not math.isfinite(x) or not math.isfinite(y):
            raise ObservationError(
                f"Reference star '{reference.identifier}' coordinate "
                "transformation produced an invalid pixel position."
            )

        return x, y