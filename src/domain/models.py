"""
Domain models for the Photometric Analysis Pipeline.

These dataclasses represent the scientific entities that flow through the
pipeline. Unlike the configuration models in ``core.models``, these models
capture the runtime state of astronomical observations.

The models are intentionally independent of survey-specific acquisition
and processing libraries. External datasets are converted into these
domain objects by survey-specific ObservationBuilder implementations.

Astropy types are used only where they form part of the scientific domain
contract, such as image containers and celestial WCS information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from datetime import datetime
from typing import Any

import numpy as np
from astropy.nddata import NDData
from astropy.wcs import WCS

from domain.enums import ObservationSource, Survey


@dataclass(slots=True)
class ImageData:
    """
    Represents a single image state.

    This is a lightweight wrapper around Astropy's NDData container,
    providing a survey-independent representation of image data
    throughout the photometric analysis pipeline.
    """

    image: NDData


@dataclass(frozen=True, slots=True)
class PSFData:
    """
    Survey-independent representation of an acquired or reconstructed PSF.

    Survey-specific acquisition code (for example, SDSS psField
    reconstruction) converts the native survey PSF into this canonical
    representation before passing it to downstream processing.

    The domain layer deliberately has no dependency on SDSS, TESS, SCARLET,
    or any survey/library-specific PSF class.
    """

    data: np.ndarray
    model_type: str

    normalised: bool = True

    # Detector/image position at which the PSF applies. These remain None
    # for a spatially invariant PSF.
    x: float | None = None
    y: float | None = None

    # Optional angular pixel scale associated with the PSF.
    pixel_scale_arcsec: float | None = None

    # Provenance of the PSF calibration/reconstruction.
    source_path: str | None = None

    # Opaque survey-specific metadata. Keeping this here prevents SDSS/TESS
    # implementation details from leaking into the common domain contract.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the canonical PSF representation."""

        array = np.asarray(self.data)

        if array.ndim != 2:
            raise ValueError(
                f"PSF data must be two-dimensional; received shape {array.shape}."
            )

        if array.size == 0:
            raise ValueError("PSF data must not be empty.")

        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(
                f"PSF data must be numeric; received dtype {array.dtype}."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError("PSF data contains non-finite values.")

        if not np.any(array != 0):
            raise ValueError("PSF data must contain non-zero values.")

        if self.normalised:
            total = float(np.sum(array))

            if not np.isfinite(total) or total <= 0:
                raise ValueError(
                    "A normalised PSF must have a finite, positive total flux."
                )

        if self.x is not None and not np.isfinite(self.x):
            raise ValueError(
                "PSF x coordinate must be finite when provided."
            )

        if self.y is not None and not np.isfinite(self.y):
            raise ValueError(
                "PSF y coordinate must be finite when provided."
            )

        if self.pixel_scale_arcsec is not None and (
            not np.isfinite(self.pixel_scale_arcsec)
            or self.pixel_scale_arcsec <= 0
        ):
            raise ValueError(
                "PSF pixel_scale_arcsec must be positive and finite "
                "when provided."
            )

        if not self.model_type.strip():
            raise ValueError("PSF model_type must not be empty.")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the two-dimensional PSF image shape."""

        return tuple(np.asarray(self.data).shape)  # type: ignore[return-value]

    @property
    def total_flux(self) -> float:
        """Return the total PSF flux."""

        return float(np.sum(self.data))

    @property
    def peak(self) -> float:
        """Return the maximum PSF pixel value."""

        return float(np.max(self.data))


@dataclass(frozen=True, slots=True)
class ObservationMetadata:
    """
    Scientific metadata describing an astronomical observation.

    These values describe the observation itself and must not be modified
    during image preprocessing, source detection, photometry or evaluation.

    The observation-time representation is intentionally flexible because
    astronomical archives do not all expose time in the same form.

    ``observation_time`` may therefore contain either:

    * a ``datetime`` when the observation is represented directly as a
      calendar timestamp; or
    * a ``float`` when the source archive uses a numerical astronomical
      time coordinate.

    When the value is numerical, ``observation_time_scale`` and
    ``observation_time_format`` identify its scientific meaning.

    Examples
    --------
    TESS::

        observation_time=2825.262629476596
        observation_time_scale="tdb"
        observation_time_format="btjd"

    Calendar timestamp::

        observation_time=datetime(...)
        observation_time_scale="utc"
        observation_time_format="datetime"
    """

    target_name: str

    observation_time: datetime | float | None

    observation_time_scale: str | None = None

    observation_time_format: str | None = None

    right_ascension: float | None = None

    declination: float | None = None

    instrument: str | None = None

    filter_name: str | None = None

    exposure_time: float | None = None

    wcs: WCS | None = None


@dataclass(frozen=True, slots=True)
class ObservationProvenance:
    """
    Provenance information describing the origin of an observation.
    """

    source: ObservationSource

    survey: Survey

    dataset_identifier: str

    filename: str | None

    download_timestamp: datetime


@dataclass(slots=True)
class ProcessingState:
    """
    Tracks the current processing status of an observation.

    Processing flags describe completed pipeline stages and are updated
    by the corresponding processing components.
    """

    background_removed: bool = False

    source_detection_completed: bool = False

    scarlet_applied: bool = False

    photometry_completed: bool = False


@dataclass(slots=True)
class Observation:
    """
    Complete scientific representation of an astronomical observation.

    The Observation is the primary domain object exchanged between
    pipeline components.

    Processing stages create derived image products while preserving
    previous image states for scientific reproducibility. The active
    image indicates the image currently used by downstream processing.

    Parameters
    ----------
    metadata
        Immutable scientific metadata describing the observation.

    provenance
        Immutable information describing the observation's origin.

    original
        Original acquired image.

    active
        Image currently used by downstream processing.

    psf
        Optional observation-level PSF calibration product. Survey-specific
        acquisition code may populate this when a native PSF is available.
        Baseline workflows remain valid when it is ``None``.

    images
        Named image products generated during processing.

    detections
        Source-detection result, when source detection has completed.

    photometry
        Photometric result, when photometry has completed.

    processing_state
        Mutable state describing completed processing stages.
    """

    metadata: ObservationMetadata

    provenance: ObservationProvenance

    original: ImageData

    active: ImageData

    # Observation-level calibration product.
    #
    # Optional so existing baseline workflows remain unchanged.
    # SCARLET-capable workflows can explicitly require it at their
    # preprocessing boundary.
    psf: PSFData | None = None

    images: dict[str, ImageData] = field(
        default_factory=dict,
    )

    detections: DetectionResult | None = None

    photometry: PhotometryResult | None = None

    processing_state: ProcessingState = field(
        default_factory=ProcessingState,
    )

    def add_image(
        self,
        name: str,
        image: ImageData,
        *,
        make_active: bool = False,
    ) -> None:
        """
        Add an image to the observation.

        Parameters
        ----------
        name
            Unique image identifier.

        image
            Image to add.

        make_active
            Whether this image becomes the active image.

        Raises
        ------
        ValueError
            If the image name is empty or an image with the same name
            already exists.
        """

        if not name:
            raise ValueError(
                "Image name must not be empty."
            )

        if name in self.images:
            raise ValueError(
                f"Image '{name}' already exists."
            )

        self.images[name] = image

        if make_active:
            self.active = image

    def get_image(
        self,
        name: str,
    ) -> ImageData:
        """
        Retrieve a named image.

        Parameters
        ----------
        name
            Name of the stored image.

        Returns
        -------
        ImageData
            Requested image.

        Raises
        ------
        KeyError
            If the image does not exist.
        """

        return self.images[name]


@dataclass(frozen=True, slots=True)
class DetectedSource:
    """
    Represents a single astronomical source identified during detection.

    A detected source identifies the location and peak intensity of a
    candidate astronomical object within an image.

    Parameters
    ----------
    identifier
        Unique identifier assigned within a detection result.

    x_centroid
        X-coordinate of the detected source centroid in pixel
        coordinates.

    y_centroid
        Y-coordinate of the detected source centroid in pixel
        coordinates.

    peak_value
        Peak pixel value associated with the detection after
        preprocessing.

    Notes
    -----
    This model intentionally contains only the properties required by
    the current source-detection and photometry stages.

    Additional source properties such as FWHM, sharpness, roundness,
    ellipticity, signal-to-noise ratio, world coordinates and catalogue
    cross-matching metadata can be introduced when required by the
    scientific methodology.
    """

    identifier: str

    x_centroid: float

    y_centroid: float

    peak_value: float


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """
    Represents the result of a source-detection operation.

    A DetectionResult captures the complete output of a single execution
    of a source-detection algorithm together with the essential
    provenance required to reproduce and interpret the detection process.

    The source collection is immutable after construction.
    """

    sources: tuple[DetectedSource, ...]

    algorithm: str

    detection_threshold: float

    detection_timestamp: datetime


@dataclass(frozen=True, slots=True)
class PhotometryMeasurement:
    """
    Represents an attempted photometric measurement of a detected source.

    Every detected source should have one corresponding measurement
    record, regardless of whether the measurement is ultimately
    considered valid.

    This preserves measurement completeness and allows downstream
    evaluation to distinguish successful measurements from
    quality-control failures.
    """

    source_identifier: str

    x_centroid: float

    y_centroid: float

    aperture_flux: float | None

    background_level: float | None

    background_flux: float | None

    net_flux: float | None

    instrumental_magnitude: float | None

    aperture_area: float | None

    background_area: float | None

    aperture_coverage: float | None

    annulus_coverage: float | None

    background_fraction: float | None

    valid: bool

    quality_flag: str


@dataclass(frozen=True, slots=True)
class PhotometryResult:
    """
    Represents the complete result of a photometry operation.

    A PhotometryResult contains one measurement record for each
    attempted detected source together with the algorithm and aperture
    configuration required to interpret the result.

    The measurement collection is immutable after construction.
    """

    measurements: tuple[PhotometryMeasurement, ...]

    algorithm: str

    aperture_radius: float

    annulus_inner_radius: float

    annulus_outer_radius: float

    photometry_timestamp: datetime


@dataclass(frozen=True, slots=True)
class TargetSourceMatch:
    """
    Represents the association between the configured astronomical
    target and a detected source.

    The association is established by transforming the target's
    celestial coordinates into image pixel coordinates using the
    observation WCS and matching the resulting position against
    detected source centroids.
    """

    target_name: str

    source_identifier: str

    target_x: float

    target_y: float

    source_x: float

    source_y: float

    separation_pixels: float


@dataclass(frozen=True, slots=True)
class LocalField:
    """
    Immutable target-centred experimental source population.

    A LocalField is a runtime scientific domain object derived from an
    observation's detected source catalogue and an already-resolved target.
    It contains the target plus the nearest selected neighbouring sources.

    Parameters
    ----------
    target_source
        Detected source corresponding to the configured astronomical target.

    neighbour_sources
        Selected neighbouring detected sources, ordered by increasing
        distance from the target.

    neighbour_distances_pixels
        Pixel distances corresponding to ``neighbour_sources``.

    available_source_count
        Number of valid detected sources available before local-field
        selection.

    max_sources
        Maximum number of sources permitted in the local field, including
        the target.

    selection_method
        Name of the deterministic selection strategy used.
    """

    target_source: DetectedSource

    neighbour_sources: tuple[DetectedSource, ...]

    neighbour_distances_pixels: tuple[float, ...]

    available_source_count: int

    max_sources: int

    selection_method: str = "nearest_pixel_distance"

    def __post_init__(self) -> None:
        """Validate the local-field result."""

        if self.max_sources <= 0:
            raise ValueError(
                "Local-field max_sources must be greater than zero."
            )

        if len(self.neighbour_sources) != len(
            self.neighbour_distances_pixels,
        ):
            raise ValueError(
                "Neighbour-source and neighbour-distance collections "
                "must have the same length."
            )

        if self.available_source_count < 1:
            raise ValueError(
                "Local field must contain at least one available source."
            )

        if len(self.sources) > self.max_sources:
            raise ValueError(
                "Local field contains more sources than max_sources."
            )

        if not math.isfinite(
            self.target_source.x_centroid,
        ) or not math.isfinite(
            self.target_source.y_centroid,
        ):
            raise ValueError(
                "Local-field target source must have finite coordinates."
            )

        for distance in self.neighbour_distances_pixels:
            if not math.isfinite(distance):
                raise ValueError(
                    "Local-field neighbour distances must be finite."
                )

            if distance < 0.0:
                raise ValueError(
                    "Local-field neighbour distances cannot be negative."
                )

        for first, second in zip(
            self.neighbour_distances_pixels,
            self.neighbour_distances_pixels[1:],
        ):
            if second < first:
                raise ValueError(
                    "Local-field neighbours must be ordered by "
                    "increasing target distance."
                )

    @property
    def sources(self) -> tuple[DetectedSource, ...]:
        """
        Return all selected sources.

        The target is always the first source, followed by neighbours in
        increasing distance order.
        """

        return (
            self.target_source,
            *self.neighbour_sources,
        )

    @property
    def selected_source_count(self) -> int:
        """Return the total number of selected sources."""

        return len(self.sources)

    @property
    def selected_neighbour_count(self) -> int:
        """Return the number of selected neighbouring sources."""

        return len(self.neighbour_sources)

    @property
    def selection_radius_pixels(self) -> float:
        """
        Return the distance to the furthest selected neighbour.

        Returns zero when the local field contains only the target.
        """

        if not self.neighbour_distances_pixels:
            return 0.0

        return self.neighbour_distances_pixels[-1]

    @property
    def source_centers(self) -> tuple[tuple[float, float], ...]:
        """
        Return selected source centres in canonical ``(y, x)`` order.

        This is the format expected by the existing SCARLET input
        boundary.
        """

        return tuple(
            (
                float(source.y_centroid),
                float(source.x_centroid),
            )
            for source in self.sources
        )


@dataclass(frozen=True, slots=True)
class ReferenceSourceMatch:
    """
    Represents the association between a configured differential-
    photometry reference star and a detected source.

    The association is established by transforming the reference
    star's celestial coordinates into image pixel coordinates using
    the observation WCS and matching the resulting position against
    detected source centroids.

    This model records the resolved association and positional
    agreement. It does not contain photometric measurements.
    """

    # Persistent astronomical identifier of the configured
    # reference star, e.g. a TIC identifier.
    reference_identifier: str

    # Identifier assigned to the corresponding detected source
    # within the observation's DetectionResult.
    source_identifier: str

    # Expected reference position derived from the reference star's
    # celestial coordinates and the observation WCS.
    reference_x: float
    reference_y: float

    # Actual detected-source centroid position.
    source_x: float
    source_y: float

    # Euclidean separation between expected and detected positions,
    # expressed in image pixels.
    separation_pixels: float