"""
Core configuration models for the Photometric Analysis Pipeline.

These dataclasses define the immutable configuration shared between
pipeline components. They contain no implementation logic and remain
independent of astronomy-specific libraries.

Time-series execution is represented explicitly through
``TimeSeriesConfig`` so that acquisition configuration, scientific
processing configuration and execution-mode configuration remain
separate concerns.

SCARLET execution parameters are represented explicitly through
``ScarletConfig`` so that SCARLET-specific processing behaviour is
reproducible and remains part of the experiment configuration rather
than being embedded in the preprocessing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.enums import (
    ExperimentType,
    ProcessingPipeline,
    Survey,
)


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """
    Configuration describing the astronomical target.
    """

    # Human-readable target identifier.
    target_name: str

    # Right Ascension in decimal degrees (ICRS).
    right_ascension: float | None = None

    # Declination in decimal degrees (ICRS).
    declination: float | None = None


@dataclass(frozen=True, slots=True)
class ReferenceStarConfig:
    """
    Configuration describing a differential-photometry reference star.

    A reference star is identified by a persistent astronomical
    identifier and celestial coordinates. The coordinates are used
    to resolve the reference against detected sources independently
    for each observation or time-series cadence.
    """

    # Persistent astronomical identifier, e.g. a TIC identifier.
    identifier: str

    # Right Ascension in decimal degrees (ICRS).
    right_ascension: float

    # Declination in decimal degrees (ICRS).
    declination: float

    def __post_init__(self) -> None:
        """
        Validate the reference-star configuration.
        """

        if not self.identifier.strip():
            raise ValueError(
                "Reference star identifier must not be empty."
            )

        if not 0.0 <= self.right_ascension <= 360.0:
            raise ValueError(
                "Reference star right ascension must be "
                "between 0 and 360 degrees."
            )

        if not -90.0 <= self.declination <= 90.0:
            raise ValueError(
                "Reference star declination must be "
                "between -90 and 90 degrees."
            )


@dataclass(frozen=True, slots=True)
class DifferentialPhotometryConfig:
    """
    Configuration controlling differential photometry.

    Reference stars may initially be empty while an experiment is being
    configured. A DIFFERENTIAL_PHOTOMETRY experiment cannot execute
    successfully until reference stars have been explicitly selected.
    """

    reference_stars: tuple[ReferenceStarConfig, ...] = ()

    def __post_init__(self) -> None:
        identifiers = tuple(
            reference.identifier.strip()
            for reference in self.reference_stars
        )

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                "Differential photometry reference stars "
                "must have unique identifiers."
            )


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    """
    Configuration controlling astronomical data acquisition.

    The configuration stores preferred acquisition parameters.
    Individual acquisition backends may interpret these as
    preferences rather than strict requirements.
    """

    # Preferred TESS observing sector.
    sector: int | None = None

    # Preferred processing pipeline (e.g. "SPOC").
    author: str | None = "SPOC"

    # Preferred observation cadence in seconds.
    cadence_seconds: int | None = 120

    # None means that the acquisition backend should preserve the
    # native dataset rather than creating a spatial cutout.
    #
    # TESS cutout width/height in pixels.
    # 25x25 provides sufficient spatial context for source detection,
    # local-field construction, crowding assessment, reference-star
    # selection and future differential photometry.
    cutout_size: int | None = 25


@dataclass(frozen=True, slots=True)
class ScarletConfig:
    """
    Configuration controlling SCARLET preprocessing.

    The parameters in this configuration correspond to the
    currently exposed SCARLET fitting and channel-selection
    controls used by the production preprocessor.

    Target-specific model initialisation parameters remain internal
    to the SCARLET implementation until the target-centred local-field
    source initialisation strategy is established.
    """

    # Maximum number of optimisation iterations used by SCARLET.
    max_iter: int = 100

    # Relative convergence tolerance used by SCARLET optimisation.
    e_rel: float = 1.0e-4

    # Explicit SCARLET channel names.
    #
    # None means that the preprocessor determines the available
    # channels from the Observation.
    channel_names: tuple[str, ...] | None = None

    # Channel used for photometric evaluation.
    #
    # None means that the preprocessor determines the appropriate
    # channel according to the existing implementation.
    photometry_channel: str | None = None


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """
    Configuration controlling image preprocessing.

    Background-estimation parameters and SCARLET-specific
    preprocessing parameters are kept together because both
    constitute the configured preprocessing stage of an experiment.
    """

    # Background2D mesh box size (pixels).
    background_box_size: int = 64

    # Background2D median filter size.
    background_filter_size: int = 5

    # Sigma-clipping threshold used during background estimation.
    sigma_clip: float = 3.0

    # Percentage of masked pixels permitted in each mesh.
    exclude_percentile: float = 10.0

    # SCARLET preprocessing configuration.
    #
    # ``default_factory`` avoids sharing a mutable default object if
    # this configuration is ever extended in the future, while
    # preserving the immutable semantics of the containing dataclass.
    scarlet: ScarletConfig = field(
        default_factory=ScarletConfig,
    )


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """
    Configuration controlling source detection.
    """

    # Full Width at Half Maximum (FWHM) of stellar sources (pixels).
    fwhm: float

    # Minimum detection threshold above the background (sigma).
    detection_threshold_sigma: float


@dataclass(frozen=True, slots=True)
class PhotometryConfig:
    """
    Configuration controlling aperture photometry.

    The configuration defines the aperture and local-background
    geometry together with quality thresholds used to determine
    whether a photometric measurement has sufficient valid image
    coverage.

    The geometry is configured independently for each survey so that
    the same photometric methodology can be applied using parameters
    appropriate to the spatial characteristics of each dataset.

    Baseline and SCARLET-enhanced workflows must use the same
    PhotometryConfig for a given dataset to ensure that differences
    in measured photometry are attributable to preprocessing rather
    than changes in photometric geometry.
    """

    # Circular source aperture radius (pixels).
    aperture_radius: float

    # Inner radius of the local-background annulus (pixels).
    annulus_inner_radius: float

    # Outer radius of the local-background annulus (pixels).
    annulus_outer_radius: float

    # Minimum fraction of the theoretical aperture area that must
    # fall within the image for a measurement to be considered valid.
    minimum_aperture_coverage: float = 0.90

    # Minimum fraction of the theoretical annulus area that must
    # fall within the image for a measurement to be considered valid.
    minimum_annulus_coverage: float = 0.70

    # Minimum number of usable annulus pixels required to estimate
    # the local residual background reliably.
    minimum_annulus_pixels: int = 20

    # Number of reference stars used for differential photometry.
    reference_star_count: int = 10

    # Maximum number of sources retained in the target-centred LocalField.
    local_field_max_sources: int = 30


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """
    Configuration controlling evaluation of photometric results.
    """

    # Maximum positional difference (pixels) when matching detections.
    matching_tolerance: float

    # Minimum separation (pixels) when assessing crowding.
    crowding_radius: float


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """
    Configuration controlling pipeline execution.
    """

    # Persist generated outputs.
    save_results: bool

    # Generate figures during execution.
    generate_figures: bool

    # Display figures interactively.
    show_figures: bool = True

    # Replace existing output files.
    overwrite_existing: bool = True

    # Logging level (e.g. INFO, DEBUG).
    logging_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class TimeSeriesConfig:
    """
    Configuration controlling cadence-level time-series execution.

    Parameters
    ----------
    enabled
        Whether the experiment should execute through the workflow's
        time-series path rather than the normal single-observation path.

    cadence_indices
        Explicit zero-based cadence indices to process.

        ``None`` means that the complete available time series should
        be processed by the workflow.

        An explicit tuple is preferred when a reproducible subset of
        cadences is required.

    Notes
    -----
    This configuration controls workflow execution, not acquisition.

    TESS acquisition preferences such as sector, author, cadence and
    cutout size remain part of ``AcquisitionConfig``.

    An empty tuple is not permitted because it would request a
    time-series experiment while selecting no observations.
    """

    enabled: bool = False

    cadence_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """
        Validate immutable time-series configuration.
        """

        if self.cadence_indices is None:
            return

        if not self.cadence_indices:
            raise ValueError(
                (
                    "Time-series cadence_indices must be None or "
                    "contain at least one cadence index."
                ),
            )

        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            for index in self.cadence_indices
        ):
            raise ValueError(
                (
                    "Time-series cadence_indices must contain only "
                    "integers."
                ),
            )

        if any(
            index < 0
            for index in self.cadence_indices
        ):
            raise ValueError(
                (
                    "Time-series cadence indices must be "
                    "non-negative."
                ),
            )

        if len(
            self.cadence_indices
        ) != len(
            set(
                self.cadence_indices,
            ),
        ):
            raise ValueError(
                (
                    "Time-series cadence_indices must not contain "
                    "duplicates."
                ),
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """
    Complete configuration for a target-centred photometric
    analysis experiment.

    The configuration separates:

    * scientific experiment definition;
    * target identification;
    * dataset acquisition preferences;
    * image preprocessing;
    * source detection;
    * aperture photometry;
    * evaluation;
    * runtime behaviour;
    * optional cadence-level time-series execution; and
    * optional differential-photometry configuration.

    Every experiment is anchored to an explicit astronomical target.
    The detected source population and any target-centred local field
    are derived from the observation during workflow execution.

    This allows Baseline and SCARLET processing variants to use the
    same scientific experiment, acquisition configuration, target,
    detected source population and photometric geometry while
    selecting different processing pipelines.
    """

    # Scientific analysis being performed.
    experiment_type: ExperimentType

    # Dataset/survey used by the experiment.
    survey: Survey

    # Astronomical target configuration.
    #
    # A target is mandatory for every experiment.
    target: TargetConfig

    # Dataset acquisition preferences.
    acquisition: AcquisitionConfig

    # Image preprocessing configuration.
    preprocessing: PreprocessingConfig

    # Source-detection configuration.
    detection: DetectionConfig

    # Aperture-photometry configuration.
    photometry: PhotometryConfig

    # Evaluation configuration.
    evaluation: EvaluationConfig

    # Runtime/output configuration.
    runtime: RuntimeConfig

    # Time-series execution configuration.
    time_series: TimeSeriesConfig = TimeSeriesConfig()

    # Optional differential-photometry configuration.
    #
    # This is required only when experiment_type is
    # DIFFERENTIAL_PHOTOMETRY. Existing target-photometry experiments
    # do not need to provide it.
    differential_photometry: DifferentialPhotometryConfig | None = None

    # Processing pipeline used by the experiment.
    pipeline: ProcessingPipeline = ProcessingPipeline.BASELINE