"""
Classical aperture photometry.

Provides the baseline aperture-photometry implementation used by the
photometric analysis pipeline.

The implementation is survey-independent and operates exclusively on the
project's Observation domain model. Photutils is used internally for
aperture construction and aperture statistics.

The photometer operates on the active image produced by the preceding
preprocessing stages. The active image is therefore expected to have
already undergone global background subtraction.

A local background annulus is nevertheless used around each detected
source to estimate and remove residual local background that remains
after preprocessing.

For every detected source, the implementation records a photometric
measurement, including measurements that fail configured quality criteria.
This preserves measurement completeness and allows downstream evaluation
to distinguish valid measurements from quality-control failures.

For valid measurements, the implementation calculates:

    aperture flux
        ↓
    local background level
        ↓
    background contribution
        ↓
    net flux
        ↓
    instrumental magnitude

Instrumental magnitude is calculated from positive background-corrected
net flux using the classical logarithmic relationship:

    m_inst = -2.5 log10(F_net)

No photometric zero point is applied at this stage because the pipeline
currently performs instrumental rather than calibrated photometry.

The resulting measurements provide the classical photometry baseline
against which future AI-enhanced or SCARLET-based measurements can be
compared.

SCARLET source-specific measurements use the same aperture geometry and
quality thresholds, but operate on source-isolated observation-space images
constructed as ``I_residual + I_source``. This removes fitted neighbour
contributions while retaining the residual/noise field for local-background
estimation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from logging import Logger

from collections.abc import Mapping
import numpy as np
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
)

from core.exceptions import ObservationError
from core.models import PhotometryConfig
from domain.models import (
    DetectedSource,
    Observation,
    PhotometryMeasurement,
    PhotometryResult,
)
from photometry.photometry import Photometer
from utils.logger import get_logger


class AperturePhotometer(Photometer):
    """
    Perform classical circular-aperture photometry.

    Each detected source is measured using:

    * a circular source aperture
    * a surrounding circular background annulus

    The source aperture provides the total measured source flux.

    The surrounding annulus provides an estimate of the residual local
    background level. This local background is scaled by the effective
    source-aperture area and subtracted from the aperture flux.

    Measurement coverage and background-pixel availability are evaluated
    before a measurement is classified as valid.

    Invalid measurements are retained in the PhotometryResult rather than
    discarded so that downstream evaluation can quantify measurement
    completeness and quality.

    Parameters
    ----------
    config
        Photometry configuration controlling aperture geometry and
        measurement-quality thresholds.

    logger
        Optional logger. If omitted, the project photometry logger
        is used.
    """

    _ALGORITHM = "Classical Aperture Photometry"

    _QUALITY_VALID = "VALID"

    _QUALITY_MISSING_SCARLET_SOURCE_MODEL = (
        "MISSING_SCARLET_SOURCE_MODEL"
    )

    _QUALITY_APERTURE_COVERAGE = (
        "INSUFFICIENT_APERTURE_COVERAGE"
    )

    _QUALITY_ANNULUS_COVERAGE = (
        "INSUFFICIENT_ANNULUS_COVERAGE"
    )

    _QUALITY_BACKGROUND_PIXELS = (
        "INSUFFICIENT_BACKGROUND_PIXELS"
    )

    _QUALITY_APERTURE_FLUX = (
        "NON_FINITE_APERTURE_FLUX"
    )

    _QUALITY_APERTURE_AREA = (
        "INVALID_APERTURE_AREA"
    )

    _QUALITY_BACKGROUND = (
        "NON_FINITE_BACKGROUND"
    )

    _QUALITY_BACKGROUND_AREA = (
        "INVALID_BACKGROUND_AREA"
    )

    _QUALITY_BACKGROUND_FLUX = (
        "NON_FINITE_BACKGROUND_FLUX"
    )

    _QUALITY_NET_FLUX = (
        "NON_FINITE_NET_FLUX"
    )

    _QUALITY_NON_POSITIVE_FLUX = (
        "NON_POSITIVE_NET_FLUX"
    )

    def __init__(
        self,
        config: PhotometryConfig,
        logger: Logger | None = None,
    ) -> None:
        """
        Initialise the aperture photometer.

        Parameters
        ----------
        config
            Aperture-photometry configuration.

        logger
            Optional logger used for diagnostic messages.
        """

        super().__init__(
            logger=logger or get_logger(__name__),
        )

        self._config = config

        self._validate_configuration()

    def measure(
        self,
        observation: Observation,
    ) -> Observation:
        """
        Perform aperture photometry on detected sources.

        The active image is assumed to have already undergone global
        background subtraction during preprocessing.

        A local background annulus is then used for each source to
        estimate residual background and calculate background-corrected
        net flux.

        Every detected source receives a PhotometryMeasurement record.
        Measurements that fail quality criteria are retained with
        ``valid=False`` and an explanatory ``quality_flag``.

        Parameters
        ----------
        observation
            Observation containing the processed image and detected
            sources.

        Returns
        -------
        Observation
            Updated Observation containing the photometry result.

        Raises
        ------
        ObservationError
            If the observation does not contain the prerequisites
            required for photometric measurement.
        """

        self._validate_observation(observation)

        data = self._extract_image_data(observation)

        detections = observation.detections

        if detections is None:
            raise ObservationError(
                "Source detection result is unavailable."
            )

        sources = detections.sources

        self._validate_source_coordinates(sources)

        self._logger.info(
            "Starting classical aperture photometry."
        )

        self._logger.info(
            "Aperture radius: %.2f pixels.",
            self._config.aperture_radius,
        )

        self._logger.info(
            "Background annulus: %.2f - %.2f pixels.",
            self._config.annulus_inner_radius,
            self._config.annulus_outer_radius,
        )

        self._logger.info(
            "Minimum aperture coverage: %.1f%%.",
            100.0 * self._config.minimum_aperture_coverage,
        )

        self._logger.info(
            "Minimum annulus coverage: %.1f%%.",
            100.0 * self._config.minimum_annulus_coverage,
        )

        self._logger.info(
            "Minimum usable annulus pixels: %d.",
            self._config.minimum_annulus_pixels,
        )

        self._logger.info(
            "Measuring %d detected sources.",
            len(sources),
        )

        positions = np.asarray(
            [
                (
                    source.x_centroid,
                    source.y_centroid,
                )
                for source in sources
            ],
            dtype=float,
        )

        #
        # Circular source apertures.
        #
        aperture = CircularAperture(
            positions,
            r=float(
                self._config.aperture_radius,
            ),
        )

        #
        # Circular local-background annuli.
        #
        annulus = CircularAnnulus(
            positions,
            r_in=float(
                self._config.annulus_inner_radius,
            ),
            r_out=float(
                self._config.annulus_outer_radius,
            ),
        )

        #
        # Explicitly mask non-finite image pixels. This prevents NaN
        # and infinite pixels from contaminating aperture statistics.
        #
        finite_mask = ~np.isfinite(data)

        #
        # Measure source apertures.
        #
        aperture_stats = ApertureStats(
            data,
            aperture,
            mask=finite_mask,
            sum_method="exact",
        )

        #
        # Measure local-background annuli.
        #
        background_stats = ApertureStats(
            data,
            annulus,
            mask=finite_mask,
            sum_method="exact",
        )

        measurements = self._build_measurements(
            sources=sources,
            aperture=aperture,
            annulus=annulus,
            aperture_stats=aperture_stats,
            background_stats=background_stats,
            data_shape=data.shape,
            finite_mask=finite_mask,
        )

        #
        # There should be one measurement record per detected source.
        #
        if len(measurements) != len(sources):
            raise ObservationError(
                "Photometry did not produce one measurement record "
                "for every detected source."
            )

        result = PhotometryResult(
            measurements=measurements,
            algorithm=self._ALGORITHM,
            aperture_radius=float(
                self._config.aperture_radius,
            ),
            annulus_inner_radius=float(
                self._config.annulus_inner_radius,
            ),
            annulus_outer_radius=float(
                self._config.annulus_outer_radius,
            ),
            photometry_timestamp=datetime.now(
                timezone.utc,
            ),
        )

        observation.photometry = result

        observation.processing_state.photometry_completed = True

        valid_count = sum(
            measurement.valid
            for measurement in measurements
        )

        invalid_count = (
            len(measurements)
            - valid_count
        )

        non_positive_flux_count = sum(
            measurement.quality_flag
            == self._QUALITY_NON_POSITIVE_FLUX
            for measurement in measurements
        )

        self._logger.info(
            "Classical aperture photometry completed successfully."
        )

        self._logger.info(
            "Recorded %d measurement records.",
            len(measurements),
        )

        self._logger.info(
            "Valid measurements: %d.",
            valid_count,
        )

        self._logger.info(
            "Invalid measurements: %d.",
            invalid_count,
        )

        self._logger.info(
            "Non-positive net flux measurements: %d.",
            non_positive_flux_count,
        )

        return observation

    def measure_source_images(
        self,
        observation: Observation,
        source_images: Mapping[str, np.ndarray],
    ) -> Observation:
        """
        Perform aperture photometry on source-specific deblended images.

        Each detected source is measured against its own two-dimensional
        source-specific image. In the SCARLET workflow that image is:

            I_deblended,i = I_observed - I_field_model + I_source,i

        which is equivalently the residual plus the individual SCARLET
        source model. The construction removes the fitted contribution of
        neighbouring sources while retaining the observation residual and
        therefore the image-domain noise/model residuals.

        The aperture radius, background annulus and all quality thresholds
        are exactly the same as the baseline aperture-photometry method.
        This is intentional: the experiment changes the source image, not
        the photometric measurement geometry.

        Parameters
        ----------
        observation
            Observation containing the completed LocalField detection
            result. The active image is used only to validate the expected
            image shape; source-specific images provide the measurement data.

        source_images
            Mapping from detected-source identifier to its source-specific
            deblended image. Missing entries are retained as invalid
            measurements rather than silently reverting to the blended image.

        Returns
        -------
        Observation
            Updated Observation containing source-level SCARLET photometry.
        """
        self._validate_observation(observation)

        reference_shape = self._extract_image_data(observation).shape
        sources = observation.detections.sources  # type: ignore[union-attr]
        self._validate_source_coordinates(sources)

        self._logger.info(
            "Starting SCARLET source-specific aperture photometry."
        )
        self._logger.info(
            "Measuring %d detected sources using individual deblended images.",
            len(sources),
        )

        measurements: list[PhotometryMeasurement] = []

        for source in sources:
            image = source_images.get(source.identifier)

            if image is None:
                measurements.append(
                    self._invalid_measurement(
                        source=source,
                        aperture_coverage=None,
                        annulus_coverage=None,
                        quality_flag=(
                            self._QUALITY_MISSING_SCARLET_SOURCE_MODEL
                        ),
                    )
                )
                continue

            data = np.asarray(image, dtype=float)

            if data.shape != reference_shape:
                raise ObservationError(
                    "SCARLET source-specific image for source "
                    f"'{source.identifier}' has shape {data.shape}; "
                    f"expected {reference_shape}."
                )

            if data.ndim != 2:
                raise ObservationError(
                    "SCARLET source-specific photometry requires "
                    "two-dimensional images."
                )

            if not np.isfinite(data).any():
                measurements.append(
                    self._invalid_measurement(
                        source=source,
                        aperture_coverage=None,
                        annulus_coverage=None,
                        quality_flag=self._QUALITY_APERTURE_FLUX,
                    )
                )
                continue

            measurements.append(
                self._measure_single_source_image(
                    source=source,
                    data=data,
                )
            )

        if len(measurements) != len(sources):
            raise ObservationError(
                "SCARLET source-specific photometry did not produce one "
                "measurement record for every detected source."
            )

        result = PhotometryResult(
            measurements=tuple(measurements),
            algorithm="SCARLET Source-Specific Aperture Photometry",
            aperture_radius=float(self._config.aperture_radius),
            annulus_inner_radius=float(
                self._config.annulus_inner_radius
            ),
            annulus_outer_radius=float(
                self._config.annulus_outer_radius
            ),
            photometry_timestamp=datetime.now(timezone.utc),
        )

        observation.photometry = result
        observation.processing_state.photometry_completed = True

        valid_count = sum(
            measurement.valid
            for measurement in measurements
        )

        self._logger.info(
            "SCARLET source-specific aperture photometry completed successfully."
        )
        self._logger.info(
            "Recorded %d measurement records; valid=%d, invalid=%d.",
            len(measurements),
            valid_count,
            len(measurements) - valid_count,
        )

        return observation

    def _measure_single_source_image(
        self,
        *,
        source: DetectedSource,
        data: np.ndarray,
    ) -> PhotometryMeasurement:
        """Measure one source on its source-specific deblended image."""
        position = np.asarray(
            [(source.x_centroid, source.y_centroid)],
            dtype=float,
        )

        aperture = CircularAperture(
            position,
            r=float(self._config.aperture_radius),
        )

        annulus = CircularAnnulus(
            position,
            r_in=float(self._config.annulus_inner_radius),
            r_out=float(self._config.annulus_outer_radius),
        )

        finite_mask = ~np.isfinite(data)

        aperture_stats = ApertureStats(
            data,
            aperture,
            mask=finite_mask,
            sum_method="exact",
        )

        background_stats = ApertureStats(
            data,
            annulus,
            mask=finite_mask,
            sum_method="exact",
        )

        measurements = self._build_measurements(
            sources=(source,),
            aperture=aperture,
            annulus=annulus,
            aperture_stats=aperture_stats,
            background_stats=background_stats,
            data_shape=data.shape,
            finite_mask=finite_mask,
        )

        if len(measurements) != 1:
            raise ObservationError(
                "SCARLET source-specific measurement produced an unexpected "
                "number of records."
            )

        return measurements[0]

    def _build_measurements(
        self,
        *,
        sources: Sequence[DetectedSource],
        aperture: CircularAperture,
        annulus: CircularAnnulus,
        aperture_stats: ApertureStats,
        background_stats: ApertureStats,
        data_shape: tuple[int, ...],
        finite_mask: np.ndarray,
    ) -> tuple[PhotometryMeasurement, ...]:
        """
        Construct domain photometry measurements.

        One measurement record is created for every detected source.

        The method evaluates:

        * aperture coverage
        * annulus coverage
        * usable annulus-pixel count
        * aperture flux
        * local background
        * background contribution
        * net flux
        * instrumental magnitude

        Invalid measurements are retained with ``valid=False`` and a
        descriptive quality flag.

        Parameters
        ----------
        sources
            Detected sources being measured.

        aperture
            Photutils source aperture.

        annulus
            Photutils local-background annulus.

        aperture_stats
            Photutils statistics for source apertures.

        background_stats
            Photutils statistics for local background annuli.

        data_shape
            Shape of the image being measured.

        finite_mask
            Boolean mask identifying non-finite image pixels.

        Returns
        -------
        tuple[PhotometryMeasurement, ...]
            Measurement records corresponding to all detected sources.
        """

        measurements: list[PhotometryMeasurement] = []

        aperture_sums = np.atleast_1d(
            np.asarray(
                aperture_stats.sum,
                dtype=float,
            )
        )

        aperture_areas = np.atleast_1d(
            np.asarray(
                aperture_stats.sum_aper_area,
                dtype=float,
            )
        )

        background_medians = np.atleast_1d(
            np.asarray(
                background_stats.median,
                dtype=float,
            )
        )

        background_areas = np.atleast_1d(
            np.asarray(
                background_stats.sum_aper_area,
                dtype=float,
            )
        )

        aperture_coverage = self._calculate_coverages(
            aperture,
            data_shape,
            finite_mask,
            theoretical_area=(
                np.pi
                * self._config.aperture_radius ** 2
            ),
        )

        annulus_coverage = self._calculate_coverages(
            annulus,
            data_shape,
            finite_mask,
            theoretical_area=(
                np.pi
                * (
                    self._config.annulus_outer_radius ** 2
                    - self._config.annulus_inner_radius ** 2
                )
            ),
        )

        usable_annulus_pixels = (
            self._calculate_usable_pixel_counts(
                annulus,
                data_shape,
                finite_mask,
            )
        )

        for index, source in enumerate(sources):

            measurement = self._create_measurement(
                source=source,
                index=index,
                aperture_sums=aperture_sums,
                aperture_areas=aperture_areas,
                background_medians=background_medians,
                background_areas=background_areas,
                aperture_coverages=aperture_coverage,
                annulus_coverages=annulus_coverage,
                usable_annulus_pixels=usable_annulus_pixels,
            )

            measurements.append(
                measurement,
            )

        return tuple(measurements)

    def _create_measurement(
        self,
        *,
        source: DetectedSource,
        index: int,
        aperture_sums: np.ndarray,
        aperture_areas: np.ndarray,
        background_medians: np.ndarray,
        background_areas: np.ndarray,
        aperture_coverages: np.ndarray,
        annulus_coverages: np.ndarray,
        usable_annulus_pixels: np.ndarray,
    ) -> PhotometryMeasurement:
        """
        Create one photometric measurement record.

        The method performs quality checks in a deliberate order so that
        the resulting quality flag identifies the first meaningful
        measurement failure.
        """

        aperture_coverage = self._get_array_value(
            aperture_coverages,
            index,
        )

        annulus_coverage = self._get_array_value(
            annulus_coverages,
            index,
        )

        #
        # Coverage must be available before the photometric quantities
        # are interpreted.
        #
        if (
            aperture_coverage is None
            or aperture_coverage
            < self._config.minimum_aperture_coverage
        ):
            return self._invalid_measurement(
                source=source,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                quality_flag=self._QUALITY_APERTURE_COVERAGE,
            )

        if (
            annulus_coverage is None
            or annulus_coverage
            < self._config.minimum_annulus_coverage
        ):
            return self._invalid_measurement(
                source=source,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                quality_flag=self._QUALITY_ANNULUS_COVERAGE,
            )

        usable_pixels = self._get_array_value(
            usable_annulus_pixels,
            index,
        )

        if (
            usable_pixels is None
            or usable_pixels
            < self._config.minimum_annulus_pixels
        ):
            return self._invalid_measurement(
                source=source,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                quality_flag=self._QUALITY_BACKGROUND_PIXELS,
            )

        #
        # Photutils must return all expected source statistics.
        #
        aperture_flux = self._get_array_value(
            aperture_sums,
            index,
        )

        if (
            aperture_flux is None
            or not np.isfinite(aperture_flux)
        ):
            return self._invalid_measurement(
                source=source,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                quality_flag=self._QUALITY_APERTURE_FLUX,
            )

        aperture_area = self._get_array_value(
            aperture_areas,
            index,
        )

        if (
            aperture_area is None
            or not np.isfinite(aperture_area)
            or aperture_area <= 0.0
        ):
            return self._invalid_measurement(
                source=source,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                quality_flag=self._QUALITY_APERTURE_AREA,
            )

        background_level = self._get_array_value(
            background_medians,
            index,
        )

        if (
            background_level is None
            or not np.isfinite(background_level)
        ):
            return self._invalid_measurement(
                source=source,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                quality_flag=self._QUALITY_BACKGROUND,
            )

        background_area = self._get_array_value(
            background_areas,
            index,
        )

        if (
            background_area is None
            or not np.isfinite(background_area)
            or background_area <= 0.0
        ):
            return self._invalid_measurement(
                source=source,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                quality_flag=self._QUALITY_BACKGROUND_AREA,
            )

        #
        # Estimate residual local background contribution within the
        # source aperture.
        #
        background_flux = (
            background_level
            * aperture_area
        )

        if not np.isfinite(background_flux):
            return self._invalid_measurement(
                source=source,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                quality_flag=self._QUALITY_BACKGROUND_FLUX,
            )

        #
        # Calculate background-corrected source flux.
        #
        net_flux = (
            aperture_flux
            - background_flux
        )

        if not np.isfinite(net_flux):
            return self._invalid_measurement(
                source=source,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                aperture_flux=aperture_flux,
                background_level=background_level,
                background_flux=background_flux,
                aperture_area=aperture_area,
                background_area=background_area,
                quality_flag=self._QUALITY_NET_FLUX,
            )

        #
        # A non-positive net flux is retained as an attempted
        # measurement. It cannot produce a logarithmic magnitude,
        # but retaining it is important for evaluation.
        #
        if net_flux <= 0.0:

            self._logger.debug(
                "Source '%s' produced non-positive net flux: %.6f.",
                source.identifier,
                net_flux,
            )

            return PhotometryMeasurement(
                source_identifier=source.identifier,
                x_centroid=float(source.x_centroid),
                y_centroid=float(source.y_centroid),
                aperture_flux=aperture_flux,
                background_level=background_level,
                background_flux=background_flux,
                net_flux=net_flux,
                instrumental_magnitude=None,
                aperture_area=aperture_area,
                background_area=background_area,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                background_fraction=self._calculate_background_fraction(
                    aperture_flux=aperture_flux,
                    background_flux=background_flux,
                ),
                valid=False,
                quality_flag=self._QUALITY_NON_POSITIVE_FLUX,
            )

        instrumental_magnitude = (
            self._calculate_instrumental_magnitude(
                net_flux,
            )
        )

        #
        # Positive finite net flux should always produce a finite
        # instrumental magnitude. Retain defensive validation here
        # because this is a scientific boundary.
        #
        if instrumental_magnitude is None:
            return self._invalid_measurement(
                source=source,
                aperture_coverage=aperture_coverage,
                annulus_coverage=annulus_coverage,
                aperture_flux=aperture_flux,
                background_level=background_level,
                background_flux=background_flux,
                net_flux=net_flux,
                aperture_area=aperture_area,
                background_area=background_area,
                quality_flag=self._QUALITY_NET_FLUX,
            )

        return PhotometryMeasurement(
            source_identifier=source.identifier,
            x_centroid=float(source.x_centroid),
            y_centroid=float(source.y_centroid),
            aperture_flux=aperture_flux,
            background_level=background_level,
            background_flux=background_flux,
            net_flux=net_flux,
            instrumental_magnitude=instrumental_magnitude,
            aperture_area=aperture_area,
            background_area=background_area,
            aperture_coverage=aperture_coverage,
            annulus_coverage=annulus_coverage,
            background_fraction=self._calculate_background_fraction(
                aperture_flux=aperture_flux,
                background_flux=background_flux,
            ),
            valid=True,
            quality_flag=self._QUALITY_VALID,
        )

    def _invalid_measurement(
        self,
        *,
        source: DetectedSource,
        aperture_coverage: float | None,
        annulus_coverage: float | None,
        quality_flag: str,
        aperture_flux: float | None = None,
        background_level: float | None = None,
        background_flux: float | None = None,
        net_flux: float | None = None,
        aperture_area: float | None = None,
        background_area: float | None = None,
    ) -> PhotometryMeasurement:
        """
        Construct an invalid photometry measurement.

        Invalid records retain all quantities that were successfully
        calculated before the failure occurred.
        """

        return PhotometryMeasurement(
            source_identifier=source.identifier,
            x_centroid=float(source.x_centroid),
            y_centroid=float(source.y_centroid),
            aperture_flux=aperture_flux,
            background_level=background_level,
            background_flux=background_flux,
            net_flux=net_flux,
            instrumental_magnitude=(
                self._calculate_instrumental_magnitude(
                    net_flux,
                )
                if net_flux is not None
                else None
            ),
            aperture_area=aperture_area,
            background_area=background_area,
            aperture_coverage=aperture_coverage,
            annulus_coverage=annulus_coverage,
            background_fraction=self._calculate_background_fraction(
                aperture_flux=aperture_flux,
                background_flux=background_flux,
            ),
            valid=False,
            quality_flag=quality_flag,
        )

    @staticmethod
    def _calculate_instrumental_magnitude(
        net_flux: float | None,
    ) -> float | None:
        """
        Calculate instrumental magnitude from net flux.

        The relationship used is:

            m_inst = -2.5 log10(F_net)

        Parameters
        ----------
        net_flux
            Background-corrected source flux.

        Returns
        -------
        float | None
            Instrumental magnitude for positive finite net flux.
            ``None`` for zero, negative or non-finite flux.
        """

        if net_flux is None:
            return None

        if not np.isfinite(net_flux):
            return None

        if net_flux <= 0.0:
            return None

        magnitude = -2.5 * np.log10(
            net_flux,
        )

        if not np.isfinite(magnitude):
            return None

        return float(magnitude)

    @staticmethod
    def _calculate_background_fraction(
        *,
        aperture_flux: float | None,
        background_flux: float | None,
    ) -> float | None:
        """
        Calculate the fraction of aperture flux attributable to
        the estimated local background.

        The quantity is defined as:

            background_fraction =
                background_flux / aperture_flux

        ``None`` is returned when the required quantities are unavailable
        or the aperture flux is zero or non-finite.
        """

        if (
            aperture_flux is None
            or background_flux is None
        ):
            return None

        if (
            not np.isfinite(aperture_flux)
            or not np.isfinite(background_flux)
        ):
            return None

        if aperture_flux == 0.0:
            return None

        fraction = (
            background_flux
            / aperture_flux
        )

        if not np.isfinite(fraction):
            return None

        return float(fraction)

    def _calculate_coverages(
        self,
        aperture: CircularAperture | CircularAnnulus,
        data_shape: tuple[int, ...],
        finite_mask: np.ndarray,
        *,
        theoretical_area: float,
    ) -> np.ndarray:
        """
        Calculate the fraction of each aperture's theoretical area
        covered by finite image pixels.

        Fractional pixel weights produced by Photutils' exact aperture
        masks are respected.

        Parameters
        ----------
        aperture
            Circular aperture or annulus.

        data_shape
            Shape of the image.

        finite_mask
            Boolean mask identifying non-finite pixels.

        theoretical_area
            Geometric area of the aperture or annulus in pixel units.

        Returns
        -------
        numpy.ndarray
            Coverage fraction for each source.
        """

        if (
            not np.isfinite(theoretical_area)
            or theoretical_area <= 0.0
        ):
            raise ValueError(
                "Theoretical aperture area must be a finite "
                "value greater than zero."
            )

        coverages: list[float] = []

        masks = aperture.to_mask(
            method="exact",
        )

        if not isinstance(masks, list):
            masks = [masks]

        for mask in masks:

            if mask is None:
                coverages.append(0.0)
                continue

            weighted_data = mask.get_values(
                np.ones(
                    data_shape,
                    dtype=float,
                ),
                mask=finite_mask,
            )

            if weighted_data.size == 0:
                coverages.append(0.0)
                continue

            weights = np.asarray(
                weighted_data,
                dtype=float,
            )

            valid_area = float(
                np.sum(weights),
            )

            coverage = (
                valid_area
                / theoretical_area
            )

            coverages.append(
                float(
                    np.clip(
                        coverage,
                        0.0,
                        1.0,
                    )
                )
            )

        return np.asarray(
            coverages,
            dtype=float,
        )

    def _calculate_usable_pixel_counts(
        self,
        annulus: CircularAnnulus,
        data_shape: tuple[int, ...],
        finite_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the number of finite image pixels contributing to
        each background annulus.

        Pixels are counted when they overlap the annulus and are not
        masked as non-finite.

        This metric is used as a conservative minimum-data-availability
        check in addition to annulus coverage.
        """

        counts: list[int] = []

        masks = annulus.to_mask(
            method="exact",
        )

        if not isinstance(masks, list):
            masks = [masks]

        for mask in masks:

            if mask is None:
                counts.append(0)
                continue

            values = mask.get_values(
                np.ones(
                    data_shape,
                    dtype=float,
                ),
                mask=finite_mask,
            )

            if values.size == 0:
                counts.append(0)
                continue

            counts.append(
                int(values.size),
            )

        return np.asarray(
            counts,
            dtype=int,
        )

    @staticmethod
    def _get_array_value(
        values: np.ndarray,
        index: int,
    ) -> float | None:
        """
        Safely retrieve a scalar value from a Photutils result array.
        """

        if index >= len(values):
            return None

        value = float(
            values[index],
        )

        if not np.isfinite(value):
            return value

        return value

    def _validate_configuration(
        self,
    ) -> None:
        """
        Validate aperture-photometry configuration.

        Raises
        ------
        ValueError
            If any configured photometry parameter is invalid.
        """

        config = self._config

        if (
            not np.isfinite(config.aperture_radius)
            or config.aperture_radius <= 0.0
        ):
            raise ValueError(
                "Aperture radius must be a finite value "
                "greater than zero."
            )

        if (
            not np.isfinite(config.annulus_inner_radius)
            or config.annulus_inner_radius <= 0.0
        ):
            raise ValueError(
                "Annulus inner radius must be a finite value "
                "greater than zero."
            )

        if (
            not np.isfinite(config.annulus_outer_radius)
            or config.annulus_outer_radius <= 0.0
        ):
            raise ValueError(
                "Annulus outer radius must be a finite value "
                "greater than zero."
            )

        #
        # The annulus must begin outside the source aperture so that
        # source flux does not contaminate the local-background estimate.
        #
        if (
            config.annulus_inner_radius
            <= config.aperture_radius
        ):
            raise ValueError(
                "Annulus inner radius must be greater than "
                "the aperture radius."
            )

        if (
            config.annulus_outer_radius
            <= config.annulus_inner_radius
        ):
            raise ValueError(
                "Annulus outer radius must be greater than "
                "the annulus inner radius."
            )

        if (
            not np.isfinite(config.minimum_aperture_coverage)
            or not 0.0
            < config.minimum_aperture_coverage
            <= 1.0
        ):
            raise ValueError(
                "Minimum aperture coverage must be a finite "
                "value greater than zero and less than or "
                "equal to one."
            )

        if (
            not np.isfinite(config.minimum_annulus_coverage)
            or not 0.0
            < config.minimum_annulus_coverage
            <= 1.0
        ):
            raise ValueError(
                "Minimum annulus coverage must be a finite "
                "value greater than zero and less than or "
                "equal to one."
            )

        if config.minimum_annulus_pixels <= 0:
            raise ValueError(
                "Minimum annulus pixels must be greater than zero."
            )

        if config.reference_star_count <= 0:
            raise ValueError(
                "Reference star count must be greater than zero."
            )

    @staticmethod
    def _validate_source_coordinates(
        sources: Sequence[DetectedSource],
    ) -> None:
        """
        Validate detected-source centroid coordinates.

        Raises
        ------
        ObservationError
            If any detected source contains non-finite centroid
            coordinates.
        """

        for source in sources:
            if (
                not np.isfinite(source.x_centroid)
                or not np.isfinite(source.y_centroid)
            ):
                raise ObservationError(
                    "Source detection contains non-finite centroid "
                    f"coordinates for source '{source.identifier}'."
                )

    def _validate_observation(
        self,
        observation: Observation,
    ) -> None:
        """
        Validate the Observation before photometric measurement.

        Raises
        ------
        ObservationError
            If required processing products are missing.
        """

        if observation is None:
            raise ObservationError(
                "Observation cannot be None."
            )

        if observation.detections is None:
            raise ObservationError(
                "Source detection must be completed before "
                "aperture photometry."
            )

        if not observation.processing_state.source_detection_completed:
            raise ObservationError(
                "Source detection is not marked as completed."
            )

        if len(observation.detections.sources) == 0:
            raise ObservationError(
                "No detected sources are available for photometry."
            )

        if observation.active is None:
            raise ObservationError(
                "Observation has no active image."
            )

        #
        # Prevent accidental replacement of an existing result.
        #
        if observation.processing_state.photometry_completed:
            raise ObservationError(
                "Photometry has already been completed for this "
                "observation."
            )

    @staticmethod
    def _extract_image_data(
        observation: Observation,
    ) -> np.ndarray:
        """
        Extract and validate the active image array.

        Returns
        -------
        numpy.ndarray
            Two-dimensional image array.

        Raises
        ------
        ObservationError
            If the active image is unsuitable for aperture photometry.
        """

        data = np.asarray(
            observation.active.image.data,
            dtype=float,
        )

        if data.ndim != 2:
            raise ObservationError(
                "Aperture photometry requires a two-dimensional image."
            )

        if data.size == 0:
            raise ObservationError(
                "Active image contains no pixels."
            )

        if not np.isfinite(data).any():
            raise ObservationError(
                "Active image contains no finite pixel values."
            )

        return data