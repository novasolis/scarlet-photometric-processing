"""TESS target setup utility.

Acquire a small TESSCut field, run the existing production preprocessing and
source-detection pipeline, resolve the configured target, cross-match the
resolved sources against TIC, and report candidate reference stars suitable
for differential photometry.

This utility is intended to assist experiment configuration. It does not
perform photometry and does not modify production experiment configuration.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, replace
from typing import Any

from acquisition.tess import TESSAcquirer
from config.experiment_config import (
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_PREPROCESSING_CONFIG,
    TESS_DETECTION_CONFIG,
    TESS_FIELD_ACQUISITION_CONFIG,
    TESS_PHOTOMETRY_CONFIG,
)
from core.models import (
    ExperimentConfig,
    RuntimeConfig,
    TargetConfig,
    TimeSeriesConfig,
)
from detection.dao_star_finder import DAOStarFinderDetector
from domain.enums import (
    ExperimentType,
    ProcessingPipeline,
    SourceScope,
    Survey,
)
from domain.tess_builder import TESSObservationBuilder
from preprocessing.background import BackgroundSubtractor
from target.resolver import TargetSourceResolver
from utils.catalogue_crossmatch import CatalogueCrossmatch
from utils.crowded_source_coordinates import (
    CrowdedSourceCoordinate,
    CrowdedSourceCoordinates,
)
from utils.logger import get_logger


# ============================================================================
# Defaults
# ============================================================================

DEFAULT_CANDIDATE_COUNT = 3
DEFAULT_CUTOUT_SIZE = 25
DEFAULT_CATALOGUE_RADIUS_ARCSEC = 5.0
DEFAULT_TARGET_CATALOGUE_RADIUS_ARCSEC = 30.0
DEFAULT_CATALOGUE_MAX_RESULTS = 10

DEFAULT_MATCHING_TOLERANCE = float(
    DEFAULT_EVALUATION_CONFIG.matching_tolerance
)


# ============================================================================
# Result models
# ============================================================================


@dataclass(frozen=True, slots=True)
class TESSSetupCandidate:
    """Detected source proposed as a differential-photometry reference."""

    source_identifier: str

    x_pixel: float
    y_pixel: float

    right_ascension: float
    declination: float

    target_separation_pixels: float

    tic_identifier: str
    tic_right_ascension: float
    tic_declination: float
    tic_separation_arcsec: float

    object_type: str | None
    t_magnitude: float | None


@dataclass(frozen=True, slots=True)
class TESSSetupResult:
    """Complete TESS target-setup result."""

    target_name: str

    target_source_identifier: str

    target_x_pixel: float
    target_y_pixel: float

    target_right_ascension: float
    target_declination: float

    target_tic_identifier: str | None
    target_tic_right_ascension: float | None
    target_tic_declination: float | None
    target_tic_separation_arcsec: float | None

    target_object_type: str | None
    target_t_magnitude: float | None

    sector: int | None
    image_shape: tuple[int, ...]

    candidates: tuple[TESSSetupCandidate, ...]


# ============================================================================
# Main utility
# ============================================================================


class TESSTargetSetup:
    """Discover a TESS target and candidate reference stars."""

    def __init__(
        self,
        *,
        target: TargetConfig,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT,
        cutout_size: int = DEFAULT_CUTOUT_SIZE,
        matching_tolerance: float = DEFAULT_MATCHING_TOLERANCE,
        catalogue_radius_arcsec: float = DEFAULT_CATALOGUE_RADIUS_ARCSEC,
        target_catalogue_radius_arcsec: float = (
            DEFAULT_TARGET_CATALOGUE_RADIUS_ARCSEC
        ),
        catalogue_max_results: int = DEFAULT_CATALOGUE_MAX_RESULTS,
        logger: logging.Logger | None = None,
    ) -> None:
        self._validate_target(target)

        self._validate_positive_int(
            candidate_count,
            "candidate_count",
        )

        self._validate_positive_int(
            cutout_size,
            "cutout_size",
        )

        self._validate_positive_float(
            matching_tolerance,
            "matching_tolerance",
        )

        self._validate_positive_float(
            catalogue_radius_arcsec,
            "catalogue_radius_arcsec",
        )

        self._validate_positive_float(
            target_catalogue_radius_arcsec,
            "target_catalogue_radius_arcsec",
        )

        self._validate_positive_int(
            catalogue_max_results,
            "catalogue_max_results",
        )

        self._target = target
        self._candidate_count = candidate_count
        self._cutout_size = cutout_size
        self._matching_tolerance = float(matching_tolerance)
        self._catalogue_radius_arcsec = float(catalogue_radius_arcsec)
        self._target_catalogue_radius_arcsec = float(
            target_catalogue_radius_arcsec
        )
        self._catalogue_max_results = catalogue_max_results

        self._logger = logger or get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> TESSSetupResult:
        """Acquire, process and identify the target and candidates."""

        config = self._build_experiment_config()

        target_name = str(
            self._target.target_name
        ).strip()

        target_ra = self._required_float(
            self._target.right_ascension,
            "Target right ascension",
        )

        target_dec = self._required_float(
            self._target.declination,
            "Target declination",
        )

        self._logger.info("=" * 80)
        self._logger.info("TESS TARGET SETUP")
        self._logger.info("=" * 80)

        self._logger.info(
            "Target: %s | RA=%.8f deg | Dec=%.8f deg",
            target_name,
            target_ra,
            target_dec,
        )

        # ==============================================================
        # Acquisition
        # ==============================================================

        self._logger.info(
            "TESS acquisition configuration:"
        )

        self._logger.info(
            "  Target       : %s",
            target_name,
        )

        self._logger.info(
            "  Sector       : any"
        )

        self._logger.info(
            "  Author       : any"
        )

        self._logger.info(
            "  Cadence      : any s"
        )

        self._logger.info(
            "  Cutout size  : %d",
            self._cutout_size,
        )

        self._logger.info(
            "  Acquisition  : TESSCut field"
        )

        self._logger.info(
            "Acquiring TESS data for '%s'.",
            target_name,
        )

        acquirer = TESSAcquirer(
            config=config,
            logger=self._logger,
        )

        dataset: Any = acquirer.acquire()

        # ==============================================================
        # Observation construction
        # ==============================================================

        self._logger.info(
            "Building single Observation from TESS Target Pixel File."
        )

        builder = TESSObservationBuilder(
            target=self._target,
            logger=self._logger,
        )

        observation: Any = builder.build(dataset)

        # ==============================================================
        # Background preprocessing
        # ==============================================================

        self._logger.info(
            "Starting astronomical background subtraction."
        )

        preprocessor = BackgroundSubtractor(
            config.preprocessing,
            logger=self._logger,
        )

        observation = preprocessor.process(
            observation
        )

        # ==============================================================
        # Source detection
        # ==============================================================

        self._logger.info(
            "Starting DAOStarFinder source detection."
        )

        detector = DAOStarFinderDetector(
            config=config.detection,
            logger=self._logger,
        )

        observation = detector.detect(
            observation
        )

        detections: Any = getattr(
            observation,
            "detections",
            None,
        )

        if detections is None:
            raise RuntimeError(
                "Source detection completed without a DetectionResult."
            )

        detected_sources: Any = getattr(
            detections,
            "sources",
            None,
        )

        if detected_sources is None:
            raise RuntimeError(
                "DetectionResult does not contain detected sources."
            )

        sources: tuple[Any, ...] = tuple(
            detected_sources
        )

        if not sources:
            raise RuntimeError(
                "No sources were detected in the TESS field."
            )

        self._logger.info(
            "Detected sources: %d.",
            len(sources),
        )

        # ==============================================================
        # Target resolution
        # ==============================================================

        self._logger.info(
            "Resolving target '%s' against detected sources.",
            target_name,
        )

        resolver = TargetSourceResolver(
            matching_tolerance=self._matching_tolerance,
            logger=self._logger,
        )

        target_match: Any = resolver.resolve(
            observation
        )

        target_source_identifier = str(
            target_match.source_identifier
        )

        self._logger.info(
            "Resolved target source: %s.",
            target_source_identifier,
        )

        target_source = self._find_source(
            sources,
            target_source_identifier,
        )

        if target_source is None:
            raise RuntimeError(
                "Resolved target source "
                f"'{target_source_identifier}' "
                "is missing from the detection result."
            )

        target_x = self._required_float(
            getattr(
                target_source,
                "x_centroid",
                None,
            ),
            "Target X centroid",
        )

        target_y = self._required_float(
            getattr(
                target_source,
                "y_centroid",
                None,
            ),
            "Target Y centroid",
        )

        # ==============================================================
        # WCS conversion
        # ==============================================================

        self._logger.info(
            "Converting detected source positions to sky coordinates."
        )

        coordinate_by_id = (
            self._convert_source_coordinates(
                observation
            )
        )

        target_coordinate = coordinate_by_id.get(
            target_source_identifier
        )

        if target_coordinate is None:
            raise RuntimeError(
                "No WCS-derived coordinates exist for resolved "
                f"target source '{target_source_identifier}'."
            )

        # ==============================================================
        # TIC cross-match
        # ==============================================================

        catalogue = CatalogueCrossmatch(
            logger=self._logger,
            survey="TESS",
            search_radius_arcsec=self._catalogue_radius_arcsec,
            max_results=self._catalogue_max_results,
        )

        target_catalogue = (
            self._search_target_catalogue(
                target_ra=target_ra,
                target_dec=target_dec,
                fallback_coordinate=target_coordinate,
                catalogue=catalogue,
            )
        )

        target_object: Any = getattr(
            target_catalogue,
            "nearest",
            None,
        )

        # ==============================================================
        # Candidate reference stars
        # ==============================================================

        candidates = self._select_candidates(
            observation=observation,
            target_source_identifier=target_source_identifier,
            coordinate_by_id=coordinate_by_id,
            catalogue=catalogue,
        )

        # ==============================================================
        # Result metadata
        # ==============================================================

        raw_image = self._get_raw_image_data(
            observation
        )

        image_shape = tuple(
            int(value)
            for value in raw_image.shape
        )

        result = TESSSetupResult(
            target_name=target_name,

            target_source_identifier=target_source_identifier,

            target_x_pixel=target_x,
            target_y_pixel=target_y,

            target_right_ascension=float(
                target_coordinate.right_ascension
            ),

            target_declination=float(
                target_coordinate.declination
            ),

            target_tic_identifier=(
                self._catalogue_identifier(
                    target_object
                )
            ),

            target_tic_right_ascension=(
                self._catalogue_float(
                    target_object,
                    "right_ascension",
                )
            ),

            target_tic_declination=(
                self._catalogue_float(
                    target_object,
                    "declination",
                )
            ),

            target_tic_separation_arcsec=(
                self._catalogue_float(
                    target_object,
                    "separation_arcsec",
                )
            ),

            target_object_type=(
                self._catalogue_string(
                    target_object,
                    "object_type",
                )
            ),

            target_t_magnitude=(
                self._catalogue_float(
                    target_object,
                    "t_magnitude",
                )
            ),

            sector=self._optional_int(
                getattr(
                    dataset,
                    "sector",
                    None,
                )
            ),

            image_shape=image_shape,

            candidates=tuple(
                candidates
            ),
        )

        self._print_result(
            result
        )

        return result

    # ------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------

    def _build_experiment_config(
        self,
    ) -> ExperimentConfig:
        """Build the minimal configuration required by TESSAcquirer."""

        acquisition = replace(
            TESS_FIELD_ACQUISITION_CONFIG,
            cutout_size=self._cutout_size,
        )

        return ExperimentConfig(
            experiment_type=ExperimentType.CROWDED_FIELD,
            source_scope=SourceScope.FIELD,
            survey=Survey.TESS,
            target=self._target,

            acquisition=acquisition,

            preprocessing=DEFAULT_PREPROCESSING_CONFIG,

            detection=TESS_DETECTION_CONFIG,

            photometry=TESS_PHOTOMETRY_CONFIG,

            evaluation=DEFAULT_EVALUATION_CONFIG,

            runtime=RuntimeConfig(
                save_results=False,
                generate_figures=False,
                show_figures=False,
                overwrite_existing=True,
                logging_level="INFO",
            ),

            time_series=TimeSeriesConfig(),

            differential_photometry=None,

            pipeline=ProcessingPipeline.BASELINE,
        )

    # ------------------------------------------------------------------
    # WCS conversion
    # ------------------------------------------------------------------

    def _convert_source_coordinates(
        self,
        observation: Any,
    ) -> dict[str, CrowdedSourceCoordinate]:
        """Convert all valid detected centroids to sky coordinates."""

        detections: Any = getattr(
            observation,
            "detections",
            None,
        )

        if detections is None:
            raise RuntimeError(
                "Source detection results are unavailable."
            )

        detected_sources: Any = getattr(
            detections,
            "sources",
            None,
        )

        if detected_sources is None:
            raise RuntimeError(
                "DetectionResult does not contain detected sources."
            )

        source_inputs: list[
            tuple[str, float, float]
        ] = []

        for source in tuple(
            detected_sources
        ):
            try:
                x_pixel = float(
                    getattr(
                        source,
                        "x_centroid",
                    )
                )

                y_pixel = float(
                    getattr(
                        source,
                        "y_centroid",
                    )
                )

            except (TypeError, ValueError):
                continue

            if not (
                math.isfinite(x_pixel)
                and math.isfinite(y_pixel)
            ):
                continue

            source_inputs.append(
                (
                    str(
                        getattr(
                            source,
                            "identifier",
                        )
                    ),
                    x_pixel,
                    y_pixel,
                )
            )

        if not source_inputs:
            raise RuntimeError(
                "No detected sources have valid centroid coordinates."
            )

        coordinates = (
            CrowdedSourceCoordinates().convert(
                observation=observation,
                sources=tuple(source_inputs),
            )
        )

        return {
            coordinate.source_identifier: coordinate
            for coordinate in coordinates
        }

    # ------------------------------------------------------------------
    # Target TIC lookup
    # ------------------------------------------------------------------

    def _search_target_catalogue(
        self,
        *,
        target_ra: float,
        target_dec: float,
        fallback_coordinate: CrowdedSourceCoordinate,
        catalogue: CatalogueCrossmatch,
    ) -> Any:
        """Cross-match configured target coordinates against TIC.

        The configured astronomical coordinates are authoritative for target
        identity. The detected WCS coordinate is used as a fallback because
        TESS centroid positions may differ from catalogue coordinates by
        several arcseconds.
        """

        target_coordinate = (
            self._make_coordinate(
                source_identifier="configured-target",

                x_pixel=float(
                    fallback_coordinate.x_pixel
                ),

                y_pixel=float(
                    fallback_coordinate.y_pixel
                ),

                right_ascension=float(
                    target_ra
                ),

                declination=float(
                    target_dec
                ),
            )
        )

        target_catalogue = (
            self._safe_catalogue_search(
                catalogue=catalogue,
                coordinate=target_coordinate,
            )
        )

        if getattr(
            target_catalogue,
            "nearest",
            None,
        ) is not None:
            self._logger.info(
                "Target TIC match found using configured "
                "target coordinates."
            )

            return target_catalogue

        self._logger.warning(
            "No TIC match found at configured target coordinates; "
            "retrying using detected WCS position."
        )

        # The target search deliberately uses a wider radius than reference
        # stars. This is important for coarse TESS centroids and blended
        # targets.
        target_catalogue_search = catalogue

        if not math.isclose(
            self._catalogue_radius_arcsec,
            self._target_catalogue_radius_arcsec,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            target_catalogue_search = (
                CatalogueCrossmatch(
                    logger=self._logger,
                    survey="TESS",
                    search_radius_arcsec=(
                        self._target_catalogue_radius_arcsec
                    ),
                    max_results=self._catalogue_max_results,
                )
            )

        return self._safe_catalogue_search(
            catalogue=target_catalogue_search,
            coordinate=fallback_coordinate,
        )

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _select_candidates(
        self,
        *,
        observation: Any,
        target_source_identifier: str,
        coordinate_by_id: dict[
            str,
            CrowdedSourceCoordinate,
        ],
        catalogue: CatalogueCrossmatch,
    ) -> list[TESSSetupCandidate]:
        """Select nearby detected sources with valid TIC matches."""

        detections: Any = getattr(
            observation,
            "detections",
            None,
        )

        if detections is None:
            raise RuntimeError(
                "Source detection results are unavailable."
            )

        detected_sources: Any = getattr(
            detections,
            "sources",
            None,
        )

        if detected_sources is None:
            raise RuntimeError(
                "DetectionResult does not contain detected sources."
            )

        sources: tuple[Any, ...] = tuple(
            detected_sources
        )

        target_source = self._find_source(
            sources,
            target_source_identifier,
        )

        if target_source is None:
            raise RuntimeError(
                f"Target source '{target_source_identifier}' "
                "was not found."
            )

        target_x = self._required_float(
            getattr(
                target_source,
                "x_centroid",
                None,
            ),
            "Target X centroid",
        )

        target_y = self._required_float(
            getattr(
                target_source,
                "y_centroid",
                None,
            ),
            "Target Y centroid",
        )

        ranked_sources = sorted(
            (
                source
                for source in sources
                if str(
                    getattr(
                        source,
                        "identifier",
                    )
                )
                != target_source_identifier
                and self._finite_centroid(
                    source
                )
            ),
            key=lambda source: self._pixel_distance(
                source,
                target_x,
                target_y,
            ),
        )

        candidates: list[
            TESSSetupCandidate
        ] = []

        for source in ranked_sources:
            if len(candidates) >= self._candidate_count:
                break

            source_identifier = str(
                getattr(
                    source,
                    "identifier",
                )
            )

            coordinate = coordinate_by_id.get(
                source_identifier
            )

            if coordinate is None:
                self._logger.debug(
                    "Skipping detected source '%s': "
                    "no WCS coordinate was produced.",
                    source_identifier,
                )
                continue

            crossmatch = (
                self._safe_catalogue_search(
                    catalogue=catalogue,
                    coordinate=coordinate,
                )
            )

            catalogue_object: Any = getattr(
                crossmatch,
                "nearest",
                None,
            )

            if catalogue_object is None:
                self._logger.debug(
                    "Skipping detected source '%s': "
                    "no TIC counterpart within %.2f arcsec.",
                    source_identifier,
                    self._catalogue_radius_arcsec,
                )
                continue

            object_type = (
                self._catalogue_string(
                    catalogue_object,
                    "object_type",
                )
            )

            if (
                object_type is not None
                and object_type.upper()
                not in {
                    "STAR",
                    "KNOWN_OBJECT",
                }
            ):
                self._logger.debug(
                    "Skipping source '%s': TIC object "
                    "type is '%s'.",
                    source_identifier,
                    object_type,
                )
                continue

            tic_identifier = (
                self._catalogue_identifier(
                    catalogue_object
                )
            )

            tic_ra = (
                self._catalogue_float(
                    catalogue_object,
                    "right_ascension",
                )
            )

            tic_dec = (
                self._catalogue_float(
                    catalogue_object,
                    "declination",
                )
            )

            tic_separation = (
                self._catalogue_float(
                    catalogue_object,
                    "separation_arcsec",
                )
            )

            if (
                tic_identifier is None
                or tic_ra is None
                or tic_dec is None
                or tic_separation is None
            ):
                self._logger.debug(
                    "Skipping source '%s': "
                    "incomplete TIC identification.",
                    source_identifier,
                )
                continue

            x_pixel = self._required_float(
                getattr(
                    source,
                    "x_centroid",
                    None,
                ),
                f"Source {source_identifier} X centroid",
            )

            y_pixel = self._required_float(
                getattr(
                    source,
                    "y_centroid",
                    None,
                ),
                f"Source {source_identifier} Y centroid",
            )

            candidate = TESSSetupCandidate(
                source_identifier=source_identifier,

                x_pixel=x_pixel,
                y_pixel=y_pixel,

                right_ascension=float(
                    coordinate.right_ascension
                ),

                declination=float(
                    coordinate.declination
                ),

                target_separation_pixels=math.hypot(
                    x_pixel - target_x,
                    y_pixel - target_y,
                ),

                tic_identifier=tic_identifier,

                tic_right_ascension=tic_ra,

                tic_declination=tic_dec,

                tic_separation_arcsec=tic_separation,

                object_type=object_type,

                t_magnitude=(
                    self._catalogue_float(
                        catalogue_object,
                        "t_magnitude",
                    )
                ),
            )

            candidates.append(
                candidate
            )

        if len(candidates) < self._candidate_count:
            self._logger.warning(
                "Only %d TIC-identified candidate(s) "
                "were found; %d requested.",
                len(candidates),
                self._candidate_count,
            )

        return candidates

    # ------------------------------------------------------------------
    # Catalogue access
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_catalogue_search(
        *,
        catalogue: CatalogueCrossmatch,
        coordinate: CrowdedSourceCoordinate,
    ) -> Any:
        """Run a catalogue search with diagnostic error reporting."""

        try:
            return catalogue.search(
                coordinate=coordinate
            )

        except Exception as exc:
            raise RuntimeError(
                "TIC catalogue cross-match failed for "
                f"coordinate RA="
                f"{coordinate.right_ascension:.8f}, "
                f"Dec="
                f"{coordinate.declination:.8f}."
            ) from exc

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_coordinate(
        *,
        source_identifier: str,
        x_pixel: float,
        y_pixel: float,
        right_ascension: float,
        declination: float,
    ) -> CrowdedSourceCoordinate:
        """Create a catalogue-compatible source coordinate."""

        return CrowdedSourceCoordinate(
            source_identifier=source_identifier,
            x_pixel=x_pixel,
            y_pixel=y_pixel,
            right_ascension=right_ascension,
            declination=declination,
        )

    @staticmethod
    def _find_source(
        sources: tuple[Any, ...],
        source_identifier: str,
    ) -> Any | None:
        """Find a detected source by identifier."""

        for source in sources:
            if (
                str(
                    getattr(
                        source,
                        "identifier",
                    )
                )
                == source_identifier
            ):
                return source

        return None

    @staticmethod
    def _finite_centroid(
        source: Any,
    ) -> bool:
        """Return whether a detected source has finite centroids."""

        try:
            x_pixel = float(
                getattr(
                    source,
                    "x_centroid",
                )
            )

            y_pixel = float(
                getattr(
                    source,
                    "y_centroid",
                )
            )

        except (TypeError, ValueError):
            return False

        return (
            math.isfinite(x_pixel)
            and math.isfinite(y_pixel)
        )

    @staticmethod
    def _pixel_distance(
        source: Any,
        target_x: float,
        target_y: float,
    ) -> float:
        """Calculate detector-pixel distance from the target."""

        x_pixel = float(
            getattr(
                source,
                "x_centroid",
            )
        )

        y_pixel = float(
            getattr(
                source,
                "y_centroid",
            )
        )

        return math.hypot(
            x_pixel - target_x,
            y_pixel - target_y,
        )

    # ------------------------------------------------------------------
    # Catalogue value extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _catalogue_identifier(
        value: Any,
    ) -> str | None:
        """Extract a human-readable catalogue identifier."""

        if value is None:
            return None

        tic_id = getattr(
            value,
            "tic_id",
            None,
        )

        if tic_id is not None:
            return f"TIC {tic_id}"

        identifier = getattr(
            value,
            "object_identifier",
            None,
        )

        if identifier is None:
            return None

        text = str(
            identifier
        ).strip()

        return text or None

    @staticmethod
    def _catalogue_float(
        value: Any,
        attribute: str,
    ) -> float | None:
        """Extract a finite floating-point catalogue value."""

        if value is None:
            return None

        raw = getattr(
            value,
            attribute,
            None,
        )

        if raw is None:
            return None

        try:
            number = float(raw)

        except (TypeError, ValueError):
            return None

        return (
            number
            if math.isfinite(number)
            else None
        )

    @staticmethod
    def _catalogue_string(
        value: Any,
        attribute: str,
    ) -> str | None:
        """Extract a catalogue string value."""

        if value is None:
            return None

        raw = getattr(
            value,
            attribute,
            None,
        )

        if raw is None:
            return None

        text = str(
            raw
        ).strip()

        return text or None

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_raw_image_data(
        observation: Any,
    ) -> Any:
        """Return the raw image data from the Observation."""

        try:
            images = getattr(
                observation,
                "images",
            )

            raw_image = images[
                "raw"
            ]

            image = getattr(
                raw_image,
                "image",
            )

            data = getattr(
                image,
                "data",
            )

        except (
            AttributeError,
            KeyError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "Observation does not contain the expected "
                "raw image data."
            ) from exc

        if data is None:
            raise RuntimeError(
                "Observation raw image contains no data."
            )

        if not hasattr(
            data,
            "shape",
        ):
            raise RuntimeError(
                "Observation raw image data does not expose "
                "a valid shape."
            )

        return data

    # ------------------------------------------------------------------
    # Generic value helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> int | None:
        """Convert an optional value to int."""

        if value is None:
            return None

        try:
            return int(value)

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _required_float(
        value: Any,
        name: str,
    ) -> float:
        """Convert a required numeric value to a finite float."""

        if value is None:
            raise ValueError(
                f"{name} is required."
            )

        try:
            number = float(value)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be numeric."
            ) from exc

        if not math.isfinite(number):
            raise ValueError(
                f"{name} must be finite."
            )

        return number

    # ------------------------------------------------------------------
    # Result presentation
    # ------------------------------------------------------------------

    def _print_result(
        self,
        result: TESSSetupResult,
    ) -> None:
        """Print a human-readable setup result."""

        self._logger.info("")
        self._logger.info("=" * 80)
        self._logger.info("TESS TARGET SETUP RESULT")
        self._logger.info("=" * 80)

        self._logger.info(
            "TESS Sector: %s | Image shape: %s",
            result.sector,
            result.image_shape,
        )

        self._logger.info(
            "TARGET | source=%s | X=%.3f | Y=%.3f | "
            "RA=%.8f | Dec=%.8f",
            result.target_source_identifier,
            result.target_x_pixel,
            result.target_y_pixel,
            result.target_right_ascension,
            result.target_declination,
        )

        self._logger.info(
            "       | TIC=%s | separation=%s arcsec | "
            "Tmag=%s | type=%s",
            result.target_tic_identifier or "NOT FOUND",
            self._format_float(
                result.target_tic_separation_arcsec
            ),
            self._format_float(
                result.target_t_magnitude
            ),
            result.target_object_type or "UNKNOWN",
        )

        self._logger.info("-" * 80)

        self._logger.info(
            "REFERENCE CANDIDATES (%d requested)",
            self._candidate_count,
        )

        self._logger.info(
            "%-8s %-9s %-9s %-18s %-10s %-10s %-8s",
            "Source",
            "X",
            "Y",
            "TIC",
            "Targ px",
            "TIC arcsec",
            "Tmag",
        )

        self._logger.info("-" * 80)

        for candidate in result.candidates:
            self._logger.info(
                "%-8s %-9.3f %-9.3f %-18s %-10.3f %-10s %-8s",
                candidate.source_identifier,
                candidate.x_pixel,
                candidate.y_pixel,
                candidate.tic_identifier,
                candidate.target_separation_pixels,
                self._format_float(
                    candidate.tic_separation_arcsec
                ),
                self._format_float(
                    candidate.t_magnitude
                ),
            )

        self._logger.info("-" * 80)

        self._logger.info(
            "COPY-PASTE REFERENCE CONFIGURATION:"
        )

        for candidate in result.candidates:
            self._logger.info(
                "ReferenceStarConfig("
                'identifier="%s", '
                "right_ascension=%.8f, "
                "declination=%.8f),",
                candidate.tic_identifier,
                candidate.tic_right_ascension,
                candidate.tic_declination,
            )

        self._logger.info("=" * 80)

    @staticmethod
    def _format_float(
        value: float | None,
    ) -> str:
        """Format an optional floating-point value."""

        if value is None:
            return "N/A"

        return f"{value:.3f}"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_positive_int(
        value: int,
        name: str,
    ) -> None:
        """Validate a positive integer parameter."""

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

        if value < 1:
            raise ValueError(
                f"{name} must be a positive integer."
            )

    @staticmethod
    def _validate_positive_float(
        value: float,
        name: str,
    ) -> None:
        """Validate a finite positive floating-point parameter."""

        try:
            numeric_value = float(value)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a finite value greater than zero."
            ) from exc

        if (
            not math.isfinite(numeric_value)
            or numeric_value <= 0
        ):
            raise ValueError(
                f"{name} must be a finite value greater than zero."
            )

    @staticmethod
    def _validate_target(
        target: TargetConfig,
    ) -> None:
        """Validate target identity and sky coordinates."""

        if target is None:
            raise ValueError(
                "Target cannot be None."
            )

        target_name = str(
            target.target_name
        ).strip()

        if not target_name:
            raise ValueError(
                "Target name must not be empty."
            )

        if (
            target.right_ascension is None
            or target.declination is None
        ):
            raise ValueError(
                "TESS target setup requires both right ascension "
                "and declination."
            )

        try:
            right_ascension = float(
                target.right_ascension
            )

            declination = float(
                target.declination
            )

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Target coordinates must be numeric."
            ) from exc

        if not math.isfinite(
            right_ascension
        ):
            raise ValueError(
                "Target right ascension must be finite."
            )

        if not math.isfinite(
            declination
        ):
            raise ValueError(
                "Target declination must be finite."
            )

        if not (
            0.0
            <= right_ascension
            <= 360.0
        ):
            raise ValueError(
                "Target right ascension must be between "
                "0 and 360 degrees."
            )

        if not (
            -90.0
            <= declination
            <= 90.0
        ):
            raise ValueError(
                "Target declination must be between "
                "-90 and 90 degrees."
            )


# ============================================================================
# CLI
# ============================================================================


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Acquire a TESSCut field, identify the target, "
            "and report TIC-matched candidate reference stars."
        )
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Astronomical target name, e.g. 'WR 21a'.",
    )

    parser.add_argument(
        "--ra",
        type=float,
        required=True,
        help=(
            "Target right ascension in decimal degrees (ICRS)."
        ),
    )

    parser.add_argument(
        "--dec",
        type=float,
        required=True,
        help=(
            "Target declination in decimal degrees (ICRS)."
        ),
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATE_COUNT,
        help=(
            "Number of candidate reference stars to report "
            f"(default: {DEFAULT_CANDIDATE_COUNT})."
        ),
    )

    parser.add_argument(
        "--cutout-size",
        type=int,
        default=DEFAULT_CUTOUT_SIZE,
        help=(
            "TESSCut width/height in pixels "
            f"(default: {DEFAULT_CUTOUT_SIZE})."
        ),
    )

    parser.add_argument(
        "--catalogue-radius",
        type=float,
        default=DEFAULT_CATALOGUE_RADIUS_ARCSEC,
        help=(
            "TIC cross-match radius for reference stars in arcsec "
            f"(default: {DEFAULT_CATALOGUE_RADIUS_ARCSEC})."
        ),
    )

    parser.add_argument(
        "--target-catalogue-radius",
        type=float,
        default=DEFAULT_TARGET_CATALOGUE_RADIUS_ARCSEC,
        help=(
            "TIC cross-match radius for the configured target in arcsec "
            f"(default: {DEFAULT_TARGET_CATALOGUE_RADIUS_ARCSEC})."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )

    return parser


def main() -> int:
    """CLI entry point."""

    parser = _build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format=(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        ),
    )

    try:
        target = TargetConfig(
            target_name=args.name.strip(),
            right_ascension=float(args.ra),
            declination=float(args.dec),
        )

        TESSTargetSetup(
            target=target,
            candidate_count=args.candidates,
            cutout_size=args.cutout_size,
            catalogue_radius_arcsec=args.catalogue_radius,
            target_catalogue_radius_arcsec=(
                args.target_catalogue_radius
            ),
        ).run()

    except Exception as exc:
        logging.getLogger(__name__).error(
            "TESS target setup failed: %s",
            exc,
            exc_info=True,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())