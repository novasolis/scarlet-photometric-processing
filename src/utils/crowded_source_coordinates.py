"""
Crowded-source sky-coordinate conversion utility.

Provides a production-safe diagnostic utility for converting detected
crowded-source pixel coordinates into astronomical sky coordinates using
the WCS associated with a completed Observation.

This module performs NO source detection, photometry, crowding analysis,
target matching, or catalogue identification.

The production workflow remains authoritative for all scientific
processing. This utility only consumes an existing Observation and
performs the deterministic coordinate transformation:

    pixel coordinates
        ↓
    Observation.metadata.wcs
        ↓
    sky coordinates (RA, Dec)

The utility is intended for:

    * investigating crowded-source candidates;
    * identifying astronomical catalogue objects;
    * preparing target-specific follow-up experiments;
    * producing thesis diagnostic tables;
    * validating WCS behaviour.

The WCS is taken directly from the Observation's scientific metadata.
The utility never reconstructs WCS information from the FITS file or
from target metadata.

Scientific responsibility
-------------------------

The separation of responsibilities is deliberately maintained:

    CrowdedSourceExtractor
        → identifies crowded candidates

    CrowdedSourceCoordinates
        → converts candidate pixel coordinates to RA/Dec

    Catalogue lookup
        → identifies astronomical objects

    Target configuration
        → stores astronomical identity and RA/Dec

    Production TargetSourceResolver
        → resolves the configured astronomical target against detected
          sources using the Observation WCS

No target identity is inferred by this utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import cast

from astropy.coordinates import Angle, SkyCoord
from astropy.wcs import WCS

from domain.models import Observation


@dataclass(frozen=True, slots=True)
class CrowdedSourceCoordinate:
    """
    Immutable sky-coordinate record for one crowded-source candidate.

    Parameters
    ----------
    source_identifier
        Identifier assigned to the detected source.

    x_pixel
        X-coordinate of the detected source centroid in pixel coordinates.

    y_pixel
        Y-coordinate of the detected source centroid in pixel coordinates.

    right_ascension
        Right ascension of the source centroid in decimal degrees.

    declination
        Declination of the source centroid in decimal degrees.
    """

    source_identifier: str
    x_pixel: float
    y_pixel: float
    right_ascension: float
    declination: float


class CrowdedSourceCoordinates:
    """
    Convert crowded-source pixel coordinates into sky coordinates.

    This utility is deliberately read-only and diagnostic. It does not
    alter the Observation or any workflow result.

    WCS information is obtained exclusively from:

        observation.metadata.wcs

    The utility does not:

        * perform source detection;
        * calculate source centroids;
        * calculate crowding;
        * perform catalogue searches;
        * identify astronomical objects;
        * create target definitions;
        * resolve targets against detected sources;
        * modify the Observation.
    """

    def convert(
        self,
        *,
        observation: Observation,
        sources: tuple[
            tuple[str, float, float],
            ...,
        ],
    ) -> tuple[CrowdedSourceCoordinate, ...]:
        """
        Convert crowded-source pixel coordinates to RA/Dec.

        Parameters
        ----------
        observation
            Observation containing the authoritative celestial WCS in
            ``observation.metadata.wcs``.

        sources
            Source records represented as:

                (source_identifier, x_pixel, y_pixel)

            Coordinates must use the same pixel coordinate system as the
            WCS associated with the Observation.

        Returns
        -------
        tuple[CrowdedSourceCoordinate, ...]
            Immutable coordinate records preserving input order.

        Raises
        ------
        ValueError
            If the Observation is invalid, WCS information is unavailable,
            the WCS is not celestial, source data are invalid, or the WCS
            transformation produces invalid coordinates.
        """

        self._validate_observation(
            observation,
        )

        wcs = self._get_celestial_wcs(
            observation,
        )

        self._validate_sources(
            sources,
        )

        coordinates: list[CrowdedSourceCoordinate] = []

        for (
            source_identifier,
            x_pixel,
            y_pixel,
        ) in sources:
            sky_coordinate = self._pixel_to_sky(
                wcs=wcs,
                x_pixel=x_pixel,
                y_pixel=y_pixel,
                source_identifier=source_identifier,
            )

            (
                right_ascension,
                declination,
            ) = self._extract_equatorial_coordinates(
                sky_coordinate=sky_coordinate,
                source_identifier=source_identifier,
            )

            record = CrowdedSourceCoordinate(
                source_identifier=source_identifier,
                x_pixel=x_pixel,
                y_pixel=y_pixel,
                right_ascension=right_ascension,
                declination=declination,
            )

            self._validate_coordinate_record(
                record,
            )

            coordinates.append(
                record,
            )

        return tuple(
            coordinates,
        )

    # ------------------------------------------------------------------
    # Observation and WCS validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_observation(
        observation: Observation,
    ) -> None:
        """
        Validate the supplied Observation.

        Raises
        ------
        ValueError
            If the Observation is unavailable.
        """

        if observation is None:
            raise ValueError(
                "Observation cannot be None."
            )

    @staticmethod
    def _get_celestial_wcs(
        observation: Observation,
    ) -> WCS:
        """
        Retrieve and validate the Observation's celestial WCS.

        Parameters
        ----------
        observation
            Observation containing the WCS.

        Returns
        -------
        WCS
            Validated celestial WCS.

        Raises
        ------
        ValueError
            If WCS information is unavailable, invalid, or non-celestial.
        """

        wcs = observation.metadata.wcs

        if wcs is None:
            raise ValueError(
                "Observation does not contain WCS information."
            )

        if not isinstance(
            wcs,
            WCS,
        ):
            raise ValueError(
                "Observation metadata contains an invalid WCS object."
            )

        if not wcs.has_celestial:
            raise ValueError(
                "Observation WCS does not contain celestial "
                "coordinate information."
            )

        return wcs

    # ------------------------------------------------------------------
    # Source validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_sources(
        sources: tuple[
            tuple[str, float, float],
            ...,
        ],
    ) -> None:
        """
        Validate supplied source coordinates.

        Raises
        ------
        ValueError
            If a source identifier is empty or coordinates are non-finite.
        """

        if sources is None:
            raise ValueError(
                "Sources cannot be None."
            )

        for (
            source_identifier,
            x_pixel,
            y_pixel,
        ) in sources:
            if not source_identifier.strip():
                raise ValueError(
                    "Crowded-source coordinate input contains "
                    "an empty source identifier."
                )

            if not isfinite(
                x_pixel,
            ):
                raise ValueError(
                    f"Source '{source_identifier}' has a non-finite "
                    f"X coordinate: {x_pixel!r}."
                )

            if not isfinite(
                y_pixel,
            ):
                raise ValueError(
                    f"Source '{source_identifier}' has a non-finite "
                    f"Y coordinate: {y_pixel!r}."
                )

    # ------------------------------------------------------------------
    # WCS transformation
    # ------------------------------------------------------------------

    @staticmethod
    def _pixel_to_sky(
        *,
        wcs: WCS,
        x_pixel: float,
        y_pixel: float,
        source_identifier: str,
    ) -> SkyCoord:
        """
        Transform one pixel coordinate into a celestial coordinate.

        Parameters
        ----------
        wcs
            Validated celestial WCS.

        x_pixel
            X-coordinate in pixel coordinates.

        y_pixel
            Y-coordinate in pixel coordinates.

        source_identifier
            Source identifier used for diagnostic error reporting.

        Returns
        -------
        SkyCoord
            Celestial coordinate corresponding to the supplied pixel
            position.

        Raises
        ------
        ValueError
            If the WCS transformation fails or does not return a
            ``SkyCoord``.
        """

        try:
            result = wcs.pixel_to_world(
                x_pixel,
                y_pixel,
            )
        except Exception as exc:
            raise ValueError(
                "WCS transformation failed for source "
                f"'{source_identifier}' at pixel coordinates "
                f"({x_pixel}, {y_pixel})."
            ) from exc

        if not isinstance(
            result,
            SkyCoord,
        ):
            raise ValueError(
                "WCS transformation returned an unexpected coordinate "
                f"type for source '{source_identifier}': "
                f"{type(result).__name__}."
            )

        return result

    @staticmethod
    def _extract_equatorial_coordinates(
        *,
        sky_coordinate: SkyCoord,
        source_identifier: str,
    ) -> tuple[float, float]:
        """
        Extract right ascension and declination in decimal degrees.

        Astropy's type definitions permit angular components to be
        scalar or array-valued. This utility performs a scalar
        pixel-to-world transformation, so the resulting values are
        expected to be scalar.

        ``cast(float, ...)`` communicates this established scalar
        contract to the static type checker without changing runtime
        behaviour.

        Parameters
        ----------
        sky_coordinate
            Validated celestial SkyCoord.

        source_identifier
            Source identifier used for diagnostic error reporting.

        Returns
        -------
        tuple[float, float]
            ``(right_ascension, declination)`` in decimal degrees.

        Raises
        ------
        ValueError
            If the coordinate does not contain valid RA/Dec values.
        """

        right_ascension = sky_coordinate.ra
        declination = sky_coordinate.dec

        if not isinstance(
            right_ascension,
            Angle,
        ):
            raise ValueError(
                "WCS transformation did not provide a valid right "
                f"ascension for source '{source_identifier}'."
            )

        if not isinstance(
            declination,
            Angle,
        ):
            raise ValueError(
                "WCS transformation did not provide a valid "
                f"declination for source '{source_identifier}'."
            )

        right_ascension_degrees = float(
            cast(
                float,
                right_ascension.deg,
            ),
        )

        declination_degrees = float(
            cast(
                float,
                declination.deg,
            ),
        )

        if not isfinite(
            right_ascension_degrees,
        ):
            raise ValueError(
                "WCS transformation produced a non-finite right "
                f"ascension for source '{source_identifier}'."
            )

        if not isfinite(
            declination_degrees,
        ):
            raise ValueError(
                "WCS transformation produced a non-finite "
                f"declination for source '{source_identifier}'."
            )

        return (
            right_ascension_degrees,
            declination_degrees,
        )

    # ------------------------------------------------------------------
    # Converted-coordinate validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_coordinate_record(
        record: CrowdedSourceCoordinate,
    ) -> None:
        """
        Validate one converted coordinate record.

        Raises
        ------
        ValueError
            If the record contains invalid coordinate values.
        """

        if not record.source_identifier.strip():
            raise ValueError(
                "Crowded-source coordinate record has an empty "
                "source identifier."
            )

        if not isfinite(
            record.x_pixel,
        ):
            raise ValueError(
                f"Source '{record.source_identifier}' has a non-finite "
                f"X coordinate: {record.x_pixel!r}."
            )

        if not isfinite(
            record.y_pixel,
        ):
            raise ValueError(
                f"Source '{record.source_identifier}' has a non-finite "
                f"Y coordinate: {record.y_pixel!r}."
            )

        if not isfinite(
            record.right_ascension,
        ):
            raise ValueError(
                f"Source '{record.source_identifier}' has a non-finite "
                f"right ascension: {record.right_ascension!r}."
            )

        if not isfinite(
            record.declination,
        ):
            raise ValueError(
                f"Source '{record.source_identifier}' has a non-finite "
                f"declination: {record.declination!r}."
            )

        if not (
            0.0
            <= record.right_ascension
            < 360.0
        ):
            raise ValueError(
                f"Source '{record.source_identifier}' has an invalid "
                f"right ascension: "
                f"{record.right_ascension!r} degrees."
            )

        if not (
            -90.0
            <= record.declination
            <= 90.0
        ):
            raise ValueError(
                f"Source '{record.source_identifier}' has an invalid "
                f"declination: "
                f"{record.declination!r} degrees."
            )

    # ------------------------------------------------------------------
    # Serialisation / presentation
    # ------------------------------------------------------------------

    @staticmethod
    def to_rows(
        records: tuple[CrowdedSourceCoordinate, ...],
    ) -> tuple[dict[str, object], ...]:
        """
        Convert coordinate records into serialisable row dictionaries.

        Parameters
        ----------
        records
            Coordinate records returned by :meth:`convert`.

        Returns
        -------
        tuple[dict[str, object], ...]
            Immutable tuple of dictionaries suitable for CSV, JSON or
            dataframe conversion.

        Raises
        ------
        ValueError
            If records is None.
        """

        if records is None:
            raise ValueError(
                "Coordinate records cannot be None."
            )

        return tuple(
            {
                "source_identifier": record.source_identifier,
                "x_pixel": record.x_pixel,
                "y_pixel": record.y_pixel,
                "right_ascension": record.right_ascension,
                "declination": record.declination,
            }
            for record in records
        )