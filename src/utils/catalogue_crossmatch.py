"""
Astronomical catalogue cross-match utility.

This module cross-matches WCS-derived astronomical source coordinates
against a survey-appropriate external catalogue.

Supported catalogues
--------------------
SDSS
    Sloan Digital Sky Survey DR17 via Astroquery SDSS.

TESS
    TESS Input Catalog (TIC) via Astroquery MAST.

The utility is intentionally diagnostic/discovery-oriented. It is used
to identify astronomical catalogue objects around detected sources and
to provide information that can subsequently be used when constructing
controlled experiments.

It is NOT part of the production photometry measurement path.

The public API remains survey-independent:

    CatalogueCrossmatch.search(...)
    CatalogueCrossmatch.search_many(...)

Existing callers which instantiate ``CatalogueCrossmatch()`` without
specifying a survey continue to use SDSS.
"""

from __future__ import annotations

import logging

from collections.abc import Callable
from dataclasses import dataclass
from math import asin
from math import cos
from math import degrees
from math import isfinite
from math import radians
from math import sin
from math import sqrt
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.sdss import SDSS

if TYPE_CHECKING:
    from utils.crowded_source_coordinates import (
        CrowdedSourceCoordinate,
    )


# ============================================================================
# Exceptions
# ============================================================================


class CatalogueCrossmatchError(RuntimeError):
    """Raised when a catalogue cross-match cannot be completed."""


# ============================================================================
# Data models
# ============================================================================


@dataclass(frozen=True, slots=True)
class CatalogueObject:
    """
    Representation of one astronomical catalogue object.

    The original SDSS-oriented fields are retained for backwards
    compatibility. TESS/TIC-specific fields are optional.

    All coordinates are expressed in decimal degrees.

    Parameters
    ----------
    catalogue
        Catalogue/provider name, e.g. ``SDSS`` or ``TIC``.

    object_identifier
        Catalogue object identifier.

    right_ascension
        Catalogue object right ascension in decimal degrees.

    declination
        Catalogue object declination in decimal degrees.

    separation_arcsec
        Angular separation from the queried source.

    object_type
        Human-readable object type when available.

    object_type_code
        Numeric catalogue object type when available.

    run, rerun, camcol, field
        SDSS imaging metadata. ``None`` for TIC objects.

    u_magnitude, g_magnitude, r_magnitude,
    i_magnitude, z_magnitude
        SDSS photometric magnitudes. ``None`` for TIC objects.

    raw_fields
        Original returned catalogue fields retained for diagnostics.

    tic_id
        TIC identifier when the source is matched against TIC.

    tic_version
        TIC version when available.

    t_magnitude
        TESS magnitude when available.

    gaia_magnitude
        Gaia magnitude associated with the TIC object when available.

    """

    catalogue: str
    object_identifier: str

    right_ascension: float
    declination: float

    separation_arcsec: float

    object_type: str | None
    object_type_code: int | None

    run: int | None
    rerun: int | None
    camcol: int | None
    field: int | None

    u_magnitude: float | None
    g_magnitude: float | None
    r_magnitude: float | None
    i_magnitude: float | None
    z_magnitude: float | None

    raw_fields: tuple[tuple[str, str], ...]

    # TESS / TIC-specific metadata.
    tic_id: int | None = None
    tic_version: int | None = None
    t_magnitude: float | None = None
    gaia_magnitude: float | None = None


@dataclass(frozen=True, slots=True)
class CatalogueCrossmatchResult:
    """
    Cross-match result for one source.

    Candidates are ordered by increasing angular separation.
    """

    source_identifier: str

    source_right_ascension: float
    source_declination: float

    search_radius_arcsec: float

    candidates: tuple[CatalogueObject, ...]

    @property
    def nearest(self) -> CatalogueObject | None:
        """Return the nearest catalogue candidate."""

        if not self.candidates:
            return None

        return self.candidates[0]


# ============================================================================
# Main utility
# ============================================================================


class CatalogueCrossmatch:
    """
    Cross-match WCS-derived sources against SDSS or TIC.

    The class deliberately remains independent from the production
    photometry workflow.

    It does not:

    * perform source detection;
    * perform WCS solving;
    * perform aperture photometry;
    * alter image data;
    * perform AI preprocessing;
    * create experiments.

    Its purpose is to identify catalogue counterparts and provide
    diagnostic information for experiment development.

    Parameters
    ----------
    logger
        Optional logger.

    survey
        Catalogue to query.

        Accepted values are:

        * ``"SDSS"``
        * ``"TESS"``
        * ``"TIC"``

        ``None`` defaults to ``SDSS`` for backwards compatibility.

    search_radius_arcsec
        Maximum angular search radius.

    max_results
        Maximum number of candidates returned for each source.

    Notes
    -----
    The TESS catalogue is the TESS Input Catalog (TIC). TESS itself is
    the observing mission; TIC is the astronomical source catalogue
    used for source identification.
    """

    _DEFAULT_SURVEY = "SDSS"

    _DEFAULT_SEARCH_RADIUS_ARCSEC = 5.0
    _DEFAULT_MAX_RESULTS = 100

    _SDSS_CATALOGUE_NAME = "SDSS"
    _SDSS_DATA_RELEASE = 17

    _TESS_CATALOGUE_NAME = "TIC"
    _TESS_CATALOGUE_NAME_MAST = "Tic"

    _OBJECT_TYPE_NAMES: dict[int, str] = {
        0: "UNKNOWN",
        1: "COSMIC_RAY",
        2: "DEFECT",
        3: "GALAXY",
        4: "GHOST",
        5: "KNOWN_OBJECT",
        6: "STAR",
        7: "STAR_TRAIL",
        8: "SKY",
    }

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        survey: str | Any = _DEFAULT_SURVEY,
        search_radius_arcsec: float = (
            _DEFAULT_SEARCH_RADIUS_ARCSEC
        ),
        max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> None:
        """Initialise the catalogue cross-match utility."""

        self._logger = (
            logger
            if logger is not None
            else logging.getLogger(__name__)
        )

        self._survey = self._normalise_survey(
            survey,
        )

        self._validate_search_radius(
            search_radius_arcsec,
        )

        self._validate_max_results(
            max_results,
        )

        self._search_radius_arcsec = float(
            search_radius_arcsec,
        )

        self._max_results = max_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        *,
        coordinate: CrowdedSourceCoordinate,
    ) -> CatalogueCrossmatchResult:
        """
        Cross-match one WCS-derived source against the selected catalogue.
        """

        self._validate_coordinate(
            coordinate,
        )

        source_identifier = str(
            coordinate.source_identifier,
        )

        source_ra = float(
            coordinate.right_ascension,
        )

        source_dec = float(
            coordinate.declination,
        )

        self._logger.debug(
            (
                "Querying %s around source '%s': "
                "RA=%.8f deg, Dec=%.8f deg, "
                "radius=%.3f arcsec."
            ),
            self._survey,
            source_identifier,
            source_ra,
            source_dec,
            self._search_radius_arcsec,
        )

        sky_coordinate = self._build_sky_coordinate(
            right_ascension=source_ra,
            declination=source_dec,
        )

        if self._survey == "SDSS":
            table = self._query_sdss(
                sky_coordinate,
            )
        elif self._survey == "TIC":
            table = self._query_tic(
                sky_coordinate,
            )
        else:
            raise CatalogueCrossmatchError(
                f"Unsupported catalogue survey '{self._survey}'.",
            )

        candidates = self._build_candidates(
            coordinate=coordinate,
            table=table,
        )

        self._logger.debug(
            (
                "%s cross-match for source '%s' "
                "returned %d candidate(s)."
            ),
            self._survey,
            source_identifier,
            len(candidates),
        )

        return CatalogueCrossmatchResult(
            source_identifier=source_identifier,
            source_right_ascension=source_ra,
            source_declination=source_dec,
            search_radius_arcsec=(
                self._search_radius_arcsec
            ),
            candidates=tuple(candidates),
        )

    def search_many(
        self,
        *,
        coordinates: tuple[CrowdedSourceCoordinate, ...],
    ) -> tuple[CatalogueCrossmatchResult, ...]:
        """
        Cross-match multiple source coordinates independently.
        """

        if coordinates is None:
            raise ValueError(
                "Coordinates cannot be None.",
            )

        results: list[
            CatalogueCrossmatchResult
        ] = []

        for coordinate in coordinates:
            results.append(
                self.search(
                    coordinate=coordinate,
                ),
            )

        return tuple(results)

    # ------------------------------------------------------------------
    # Survey selection
    # ------------------------------------------------------------------

    @classmethod
    def _normalise_survey(
        cls,
        survey: str | Any,
    ) -> str:
        """
        Normalise a survey name.

        Supports both strings and enum-like objects exposing ``name``
        or ``value``. This keeps the utility independent from the
        project's domain enum implementation.
        """

        if survey is None:
            return cls._DEFAULT_SURVEY

        if isinstance(
            survey,
            str,
        ):
            value = survey

        else:
            value = getattr(
                survey,
                "name",
                None,
            )

            if value is None:
                value = getattr(
                    survey,
                    "value",
                    None,
                )

            if value is None:
                value = str(
                    survey,
                )

        normalised = str(
            value,
        ).strip().upper()

        if normalised == "TESS":
            return "TIC"

        if normalised in {
            "TIC",
            "SDSS",
        }:
            return normalised

        raise ValueError(
            (
                "Unsupported catalogue survey. "
                "Expected 'SDSS', 'TESS', or 'TIC'; "
                f"received {survey!r}."
            ),
        )

    # ------------------------------------------------------------------
    # Coordinate handling
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sky_coordinate(
        *,
        right_ascension: float,
        declination: float,
    ) -> SkyCoord:
        """Construct an ICRS SkyCoord."""

        return SkyCoord(
            ra=right_ascension,
            dec=declination,
            unit=(u.deg, u.deg),
            frame="icrs",
        )

    # ------------------------------------------------------------------
    # SDSS access
    # ------------------------------------------------------------------

    def _query_sdss(
        self,
        coordinate: SkyCoord,
    ) -> Any:
        """
        Query SDSS DR17 using Astroquery.
        """

        radius = (
            self._search_radius_arcsec
            * u.arcsec
        )

        query_region_attribute = getattr(
            SDSS,
            "query_region",
            None,
        )

        if not callable(
            query_region_attribute,
        ):
            raise CatalogueCrossmatchError(
                (
                    "The installed Astroquery SDSS client "
                    "does not provide query_region()."
                ),
            )

        query_region = cast(
            Callable[..., Any],
            query_region_attribute,
        )

        try:
            table = query_region(
                coordinate,
                radius=radius,
                data_release=self._SDSS_DATA_RELEASE,
            )

        except Exception as exc:
            self._logger.debug(
                "SDSS query_region() raised an exception.",
                exc_info=True,
            )

            raise CatalogueCrossmatchError(
                (
                    "Unable to query the SDSS DR17 "
                    "catalogue service."
                ),
            ) from exc

        return table

    # ------------------------------------------------------------------
    # TIC / TESS access
    # ------------------------------------------------------------------

    def _query_tic(
        self,
        coordinate: SkyCoord,
    ) -> Any:
        """
        Query the TESS Input Catalog through Astroquery MAST.

        The TIC is accessed using ``Catalogs.query_region()`` rather
        than a direct HTTP request.
        """

        radius = (
            self._search_radius_arcsec
            * u.arcsec
        )

        query_region_attribute = getattr(
            Catalogs,
            "query_region",
            None,
        )

        if not callable(
            query_region_attribute,
        ):
            raise CatalogueCrossmatchError(
                (
                    "The installed Astroquery MAST client "
                    "does not provide Catalogs.query_region()."
                ),
            )

        query_region = cast(
            Callable[..., Any],
            query_region_attribute,
        )

        try:
            table = query_region(
                coordinate,
                radius=radius,
                catalog=self._TESS_CATALOGUE_NAME_MAST,
            )

        except Exception as exc:
            self._logger.debug(
                "TIC Catalogs.query_region() raised an exception.",
                exc_info=True,
            )

            raise CatalogueCrossmatchError(
                (
                    "Unable to query the TESS Input "
                    "Catalog through MAST."
                ),
            ) from exc

        return table

    # ------------------------------------------------------------------
    # Candidate conversion
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        *,
        coordinate: CrowdedSourceCoordinate,
        table: Any,
    ) -> list[CatalogueObject]:
        """
        Convert an Astroquery result table into CatalogueObject values.
        """

        if table is None:
            return []

        try:
            row_count = len(table)

        except TypeError:
            return []

        if row_count == 0:
            return []

        source_ra = float(
            coordinate.right_ascension,
        )

        source_dec = float(
            coordinate.declination,
        )

        candidates: list[
            CatalogueObject
        ] = []

        for row in table:
            catalogue_ra = self._extract_float(
                row,
                (
                    "ra",
                    "RA",
                    "raj2000",
                ),
            )

            catalogue_dec = self._extract_float(
                row,
                (
                    "dec",
                    "Dec",
                    "DEC",
                    "dej2000",
                ),
            )

            if (
                catalogue_ra is None
                or catalogue_dec is None
            ):
                continue

            if not (
                0.0
                <= catalogue_ra
                < 360.0
            ):
                continue

            if not (
                -90.0
                <= catalogue_dec
                <= 90.0
            ):
                continue

            separation = (
                self._angular_separation_arcsec(
                    source_right_ascension=source_ra,
                    source_declination=source_dec,
                    catalogue_right_ascension=(
                        catalogue_ra
                    ),
                    catalogue_declination=(
                        catalogue_dec
                    ),
                )
            )

            if (
                separation
                > self._search_radius_arcsec
            ):
                continue

            if self._survey == "SDSS":
                candidate = (
                    self._build_sdss_candidate(
                        row=row,
                        catalogue_ra=catalogue_ra,
                        catalogue_dec=catalogue_dec,
                        separation=separation,
                    )
                )

            else:
                candidate = (
                    self._build_tic_candidate(
                        row=row,
                        catalogue_ra=catalogue_ra,
                        catalogue_dec=catalogue_dec,
                        separation=separation,
                    )
                )

            candidates.append(
                candidate,
            )

        candidates.sort(
            key=lambda item: (
                item.separation_arcsec,
                item.object_identifier,
            ),
        )

        return candidates[
            : self._max_results
        ]

    # ------------------------------------------------------------------
    # SDSS candidate conversion
    # ------------------------------------------------------------------

    def _build_sdss_candidate(
        self,
        *,
        row: Any,
        catalogue_ra: float,
        catalogue_dec: float,
        separation: float,
    ) -> CatalogueObject:
        """Build a CatalogueObject from an SDSS result row."""

        type_code = self._extract_int(
            row,
            (
                "type",
                "objtype",
                "objType",
                "objc_type",
            ),
        )

        return CatalogueObject(
            catalogue=self._SDSS_CATALOGUE_NAME,

            object_identifier=(
                self._extract_object_identifier(
                    row,
                )
            ),

            right_ascension=catalogue_ra,
            declination=catalogue_dec,

            separation_arcsec=separation,

            object_type=(
                self._extract_object_type(
                    row,
                    type_code,
                )
            ),

            object_type_code=type_code,

            run=self._extract_int(
                row,
                ("run",),
            ),

            rerun=self._extract_int(
                row,
                ("rerun",),
            ),

            camcol=self._extract_int(
                row,
                ("camcol",),
            ),

            field=self._extract_int(
                row,
                ("field",),
            ),

            u_magnitude=self._extract_magnitude(
                row,
                (
                    "psfMag_u",
                    "psfmag_u",
                    "psfMagU",
                    "u",
                    "petroMag_u",
                    "modelMag_u",
                ),
            ),

            g_magnitude=self._extract_magnitude(
                row,
                (
                    "psfMag_g",
                    "psfmag_g",
                    "psfMagG",
                    "g",
                    "petroMag_g",
                    "modelMag_g",
                ),
            ),

            r_magnitude=self._extract_magnitude(
                row,
                (
                    "psfMag_r",
                    "psfmag_r",
                    "psfMagR",
                    "r",
                    "petroMag_r",
                    "modelMag_r",
                ),
            ),

            i_magnitude=self._extract_magnitude(
                row,
                (
                    "psfMag_i",
                    "psfmag_i",
                    "psfMagI",
                    "i",
                    "petroMag_i",
                    "modelMag_i",
                ),
            ),

            z_magnitude=self._extract_magnitude(
                row,
                (
                    "psfMag_z",
                    "psfmag_z",
                    "psfMagZ",
                    "z",
                    "petroMag_z",
                    "modelMag_z",
                ),
            ),

            raw_fields=(
                self._extract_raw_fields(
                    row,
                )
            ),
        )

    # ------------------------------------------------------------------
    # TIC candidate conversion
    # ------------------------------------------------------------------

    def _build_tic_candidate(
        self,
        *,
        row: Any,
        catalogue_ra: float,
        catalogue_dec: float,
        separation: float,
    ) -> CatalogueObject:
        """Build a CatalogueObject from a TIC result row."""

        tic_id = self._extract_int(
            row,
            (
                "ID",
                "id",
                "ticid",
                "TICID",
                "tic_id",
            ),
        )

        tic_version = self._extract_int(
            row,
            (
                "version",
                "tic_version",
            ),
        )

        t_magnitude = self._extract_magnitude(
            row,
            (
                "Tmag",
                "tmag",
                "T_MAG",
                "tessmag",
            ),
        )

        gaia_magnitude = self._extract_magnitude(
            row,
            (
                "GAIAmag",
                "Gaiamag",
                "gaiamag",
                "GAIA",
            ),
        )

        object_identifier = (
            f"TIC {tic_id}"
            if tic_id is not None
            else "UNKNOWN"
        )

        return CatalogueObject(
            catalogue=self._TESS_CATALOGUE_NAME,

            object_identifier=object_identifier,

            right_ascension=catalogue_ra,
            declination=catalogue_dec,

            separation_arcsec=separation,

            object_type=self._extract_tic_object_type(
                row,
            ),

            object_type_code=None,

            run=None,
            rerun=None,
            camcol=None,
            field=None,

            u_magnitude=None,
            g_magnitude=None,
            r_magnitude=None,
            i_magnitude=None,
            z_magnitude=None,

            raw_fields=(
                self._extract_raw_fields(
                    row,
                )
            ),

            tic_id=tic_id,
            tic_version=tic_version,
            t_magnitude=t_magnitude,
            gaia_magnitude=gaia_magnitude,
        )

    @classmethod
    def _extract_tic_object_type(
        cls,
        row: Any,
    ) -> str | None:
        """
        Extract a useful TIC object classification.

        TIC releases can expose classification-related fields under
        slightly different names. The raw fields are retained even
        when no standardised classification is available.
        """

        value = cls._extract_value(
            row,
            (
                "objType",
                "objtype",
                "type",
                "ObjectType",
            ),
        )

        return cls._clean_text(
            value,
        )

    # ------------------------------------------------------------------
    # SDSS metadata
    # ------------------------------------------------------------------

    @classmethod
    def _extract_object_type(
        cls,
        row: Any,
        type_code: int | None,
    ) -> str | None:
        """
        Extract a human-readable SDSS object type.

        Some SDSS table variants expose the type as an integer while
        others can expose a textual representation.
        """

        raw_type = cls._extract_value(
            row,
            (
                "type",
                "objtype",
                "objType",
                "objc_type",
            ),
        )

        if raw_type is not None:
            text = cls._clean_text(
                raw_type,
            )

            if text is not None:
                try:
                    numeric_type = int(
                        float(text),
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    return text

                return cls._OBJECT_TYPE_NAMES.get(
                    numeric_type,
                    f"TYPE_{numeric_type}",
                )

        if type_code is None:
            return None

        return cls._OBJECT_TYPE_NAMES.get(
            type_code,
            f"TYPE_{type_code}",
        )

    @classmethod
    def _extract_object_identifier(
        cls,
        row: Any,
    ) -> str:
        """Extract an SDSS object identifier."""

        value = cls._extract_value(
            row,
            (
                "objid",
                "objID",
                "objId",
                "objectid",
                "objectID",
                "id",
            ),
        )

        text = cls._clean_text(
            value,
        )

        if text is None:
            return "UNKNOWN"

        return text

    @classmethod
    def _extract_magnitude(
        cls,
        row: Any,
        fields: tuple[str, ...],
    ) -> float | None:
        """
        Extract a magnitude while rejecting invalid sentinel values.
        """

        value = cls._extract_float(
            row,
            fields,
        )

        if value is None:
            return None

        if not isfinite(value):
            return None

        if value <= -90.0:
            return None

        if value >= 90.0:
            return None

        return value

    # ------------------------------------------------------------------
    # Coordinate mathematics
    # ------------------------------------------------------------------

    @staticmethod
    def _angular_separation_arcsec(
        *,
        source_right_ascension: float,
        source_declination: float,
        catalogue_right_ascension: float,
        catalogue_declination: float,
    ) -> float:
        """
        Calculate great-circle separation in arcseconds.
        """

        source_ra = radians(
            source_right_ascension,
        )

        source_dec = radians(
            source_declination,
        )

        catalogue_ra = radians(
            catalogue_right_ascension,
        )

        catalogue_dec = radians(
            catalogue_declination,
        )

        delta_ra = (
            catalogue_ra
            - source_ra
        )

        delta_dec = (
            catalogue_dec
            - source_dec
        )

        haversine = (
            sin(delta_dec / 2.0) ** 2
            + cos(source_dec)
            * cos(catalogue_dec)
            * sin(delta_ra / 2.0) ** 2
        )

        haversine = max(
            0.0,
            min(
                1.0,
                haversine,
            ),
        )

        angular_distance = (
            2.0
            * asin(
                sqrt(haversine),
            )
        )

        return (
            degrees(
                angular_distance,
            )
            * 3600.0
        )

    # ------------------------------------------------------------------
    # Generic Astropy table extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _column_names(
        row: Any,
    ) -> tuple[Any, ...]:
        """
        Return available column names from an Astropy row.
        """

        try:
            names = row.colnames

        except AttributeError:
            try:
                names = row.keys()

            except AttributeError:
                return ()

        try:
            return tuple(
                names,
            )

        except TypeError:
            return ()

    @classmethod
    def _extract_value(
        cls,
        row: Any,
        fields: str | tuple[str, ...],
    ) -> Any:
        """
        Extract a field using case-insensitive matching.

        Handles common differences in SDSS/TIC/Astroquery column
        naming.
        """

        requested_fields = (
            (fields,)
            if isinstance(
                fields,
                str,
            )
            else fields
        )

        column_names = cls._column_names(
            row,
        )

        if not column_names:
            return None

        normalised_names: dict[
            str,
            Any,
        ] = {}

        for column_name in column_names:
            normalised_names[
                cls._normalise_field_name(
                    column_name,
                )
            ] = column_name

        for requested in requested_fields:
            normalised_requested = (
                cls._normalise_field_name(
                    requested,
                )
            )

            actual_name = (
                normalised_names.get(
                    normalised_requested,
                )
            )

            if actual_name is None:
                continue

            try:
                value = row[
                    actual_name
                ]

            except (
                KeyError,
                IndexError,
                TypeError,
            ):
                continue

            if cls._is_missing(
                value,
            ):
                continue

            return value

        return None

    @staticmethod
    def _normalise_field_name(
        value: Any,
    ) -> str:
        """Normalise a catalogue field name."""

        text = str(
            value,
        ).strip().lower()

        return "".join(
            character
            for character in text
            if character.isalnum()
        )

    @classmethod
    def _extract_int(
        cls,
        row: Any,
        fields: tuple[str, ...],
    ) -> int | None:
        """Extract an integer catalogue field."""

        value = cls._extract_value(
            row,
            fields,
        )

        if value is None:
            return None

        text = cls._clean_text(
            value,
        )

        if text is None:
            return None

        try:
            return int(
                float(text),
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _extract_float(
        cls,
        row: Any,
        fields: tuple[str, ...],
    ) -> float | None:
        """Extract a finite floating-point catalogue field."""

        value = cls._extract_value(
            row,
            fields,
        )

        if value is None:
            return None

        text = cls._clean_text(
            value,
        )

        if text is None:
            return None

        try:
            parsed = float(
                text,
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

        if not isfinite(
            parsed,
        ):
            return None

        return parsed

    @classmethod
    def _clean_text(
        cls,
        value: Any,
    ) -> str | None:
        """
        Convert an Astropy/numpy/string-like value to clean text.
        """

        if value is None:
            return None

        if cls._is_missing(
            value,
        ):
            return None

        try:
            if hasattr(
                value,
                "item",
            ):
                value = value.item()

        except (
            AttributeError,
            ValueError,
        ):
            pass

        if isinstance(
            value,
            bytes,
        ):
            try:
                value = value.decode(
                    "utf-8",
                    errors="replace",
                )

            except UnicodeDecodeError:
                return None

        text = str(
            value,
        ).strip()

        if not text:
            return None

        if text.lower() in {
            "none",
            "nan",
            "null",
            "--",
            "masked",
            "mask",
        }:
            return None

        return text

    @staticmethod
    def _is_missing(
        value: Any,
    ) -> bool:
        """
        Detect Astropy masked or otherwise missing values.
        """

        if value is None:
            return True

        mask = getattr(
            value,
            "mask",
            False,
        )

        if isinstance(
            mask,
            bool,
        ):
            return mask

        try:
            if bool(mask):
                return True

        except (
            TypeError,
            ValueError,
        ):
            pass

        try:
            if hasattr(
                value,
                "item",
            ):
                scalar = value.item()

            else:
                scalar = value

        except (
            AttributeError,
            ValueError,
        ):
            scalar = value

        if isinstance(
            scalar,
            float,
        ):
            return not isfinite(
                scalar,
            )

        return False

    @classmethod
    def _extract_raw_fields(
        cls,
        row: Any,
    ) -> tuple[tuple[str, str], ...]:
        """
        Preserve returned catalogue fields for diagnostics.
        """

        column_names = cls._column_names(
            row,
        )

        if not column_names:
            return ()

        fields: list[
            tuple[str, str]
        ] = []

        for column_name in column_names:
            try:
                value = row[
                    column_name
                ]

            except (
                KeyError,
                IndexError,
                TypeError,
            ):
                continue

            text = cls._clean_text(
                value,
            )

            if text is None:
                text = "N/A"

            fields.append(
                (
                    str(column_name),
                    text,
                ),
            )

        fields.sort(
            key=lambda item: (
                item[0].lower(),
            ),
        )

        return tuple(
            fields,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_coordinate(
        coordinate: CrowdedSourceCoordinate,
    ) -> None:
        """Validate a WCS-derived source coordinate."""

        if coordinate is None:
            raise ValueError(
                "Catalogue coordinate cannot be None.",
            )

        identifier = str(
            coordinate.source_identifier,
        )

        if not identifier.strip():
            raise ValueError(
                (
                    "Catalogue coordinate has an "
                    "empty source identifier."
                ),
            )

        try:
            right_ascension = float(
                coordinate.right_ascension,
            )

            declination = float(
                coordinate.declination,
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                (
                    f"Source '{identifier}' has "
                    "non-numeric sky coordinates."
                ),
            ) from exc

        if not isfinite(
            right_ascension,
        ):
            raise ValueError(
                (
                    f"Source '{identifier}' has a "
                    "non-finite right ascension."
                ),
            )

        if not isfinite(
            declination,
        ):
            raise ValueError(
                (
                    f"Source '{identifier}' has a "
                    "non-finite declination."
                ),
            )

        if not (
            0.0
            <= right_ascension
            < 360.0
        ):
            raise ValueError(
                (
                    f"Source '{identifier}' has an "
                    f"invalid right ascension: "
                    f"{right_ascension}."
                ),
            )

        if not (
            -90.0
            <= declination
            <= 90.0
        ):
            raise ValueError(
                (
                    f"Source '{identifier}' has an "
                    f"invalid declination: "
                    f"{declination}."
                ),
            )

    @staticmethod
    def _validate_search_radius(
        value: float,
    ) -> None:
        """Validate the catalogue search radius."""

        if (
            isinstance(
                value,
                bool,
            )
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise ValueError(
                "Search radius must be numeric.",
            )

        radius = float(
            value,
        )

        if not isfinite(
            radius,
        ):
            raise ValueError(
                "Search radius must be finite.",
            )

        if radius <= 0.0:
            raise ValueError(
                (
                    "Search radius must be greater "
                    "than zero."
                ),
            )

        if radius > 180.0:
            raise ValueError(
                (
                    "Search radius must not exceed "
                    "180 arcsec (3 arcmin)."
                ),
            )

    @staticmethod
    def _validate_max_results(
        value: int,
    ) -> None:
        """Validate the maximum candidate count."""

        if (
            isinstance(
                value,
                bool,
            )
            or not isinstance(
                value,
                int,
            )
        ):
            raise ValueError(
                "Maximum results must be an integer.",
            )

        if value <= 0:
            raise ValueError(
                (
                    "Maximum results must be greater "
                    "than zero."
                ),
            )