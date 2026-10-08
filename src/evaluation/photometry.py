"""
Photometry evaluation.

Provides survey-independent evaluation of photometric measurements
contained within an Observation.

The evaluator calculates source-level and observation-level quality
metrics for the completed photometry stage, including:

    * measurement validity
    * measurement completeness
    * valid measurement fraction
    * invalid measurement count
    * missing measurement count
    * non-positive flux count
    * local neighbour count
    * crowded-source classification
    * crowding fraction
    * net flux statistics
    * local background statistics
    * background contribution statistics

The evaluator operates exclusively on the project's domain models.
It does not depend on Photutils, Lightkurve, Astroquery or any
survey-specific implementation.

No scientific processing or modification of the Observation is
performed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from logging import Logger
from math import hypot, isfinite
from statistics import median
from typing import Iterable

from core.models import EvaluationConfig
from domain.models import (
    DetectedSource,
    Observation,
    PhotometryMeasurement,
)
from evaluation.base import Evaluator
from evaluation.models import (
    PhotometryEvaluation,
    SourceEvaluation,
)
from utils.logger import get_logger


class PhotometryEvaluator(Evaluator):
    """
    Evaluate the photometric result contained in an Observation.

    The evaluator produces:

        1. one SourceEvaluation for each detected source; and
        2. one PhotometryEvaluation containing observation-level
           summary statistics.

    Measurement validity is determined by the ``valid`` flag stored
    in each PhotometryMeasurement.

    Crowding is evaluated independently from photometric validity by
    examining the positions of all detected sources.

    A measurement record and a valid measurement are deliberately
    treated as different concepts:

        * ``measured_source_count`` counts detected sources with a
          corresponding photometric measurement record.
        * ``valid_measurement_count`` counts measurement records that
          passed photometric quality control.
        * ``invalid_measurement_count`` counts produced measurement
          records that failed quality control.
        * ``missing_measurement_count`` counts detected sources for
          which no measurement record was produced.

    Parameters
    ----------
    config
        Evaluation configuration controlling the crowding radius.

    logger
        Optional logger used for diagnostic and evaluation messages.
        If omitted, the project evaluation logger is used.
    """

    _MISSING_MEASUREMENT_FLAG = "MISSING_MEASUREMENT"

    def __init__(
        self,
        config: EvaluationConfig,
        logger: Logger | None = None,
    ) -> None:
        """
        Initialise the photometry evaluator.

        Parameters
        ----------
        config
            Evaluation configuration.

        logger
            Optional logger used for diagnostic messages.
        """

        super().__init__(
            logger=logger or get_logger(__name__),
        )

        self._config = config

        self._validate_configuration()

    def evaluate(
        self,
        observation: Observation,
    ) -> PhotometryEvaluation:
        """
        Evaluate the photometric result contained in an Observation.

        Parameters
        ----------
        observation
            Observation containing completed source detection and
            photometric measurements.

        Returns
        -------
        PhotometryEvaluation
            Immutable source-level and observation-level evaluation
            results.

        Raises
        ------
        ValueError
            If the Observation does not contain the prerequisites
            required for evaluation.
        """

        self._validate_observation(
            observation,
        )

        #
        # Validation guarantees that these objects exist.
        #
        assert observation.detections is not None
        assert observation.photometry is not None

        detected_sources = observation.detections.sources
        measurements = observation.photometry.measurements

        self._validate_detected_sources(
            detected_sources,
        )

        detected_source_count = len(
            detected_sources,
        )

        self._logger.info(
            "Starting photometry evaluation."
        )

        self._logger.info(
            "Detected sources: %d.",
            detected_source_count,
        )

        self._logger.info(
            "Photometry measurement records: %d.",
            len(measurements),
        )

        #
        # Build a lookup of measurements by source identifier.
        #
        # The photometry stage is expected to produce one measurement
        # for each detected source. The evaluator nevertheless handles
        # missing records defensively so incomplete processing is
        # explicitly represented in the evaluation.
        #
        measurement_by_source = self._index_measurements(
            measurements,
        )

        detected_source_identifiers = {
            source.identifier
            for source in detected_sources
        }

        #
        # Detect measurement records that do not correspond to a
        # detected source. Such records are retained in the photometry
        # result but are excluded from source-level evaluation metrics.
        #
        orphan_measurement_count = sum(
            1
            for identifier in measurement_by_source
            if identifier not in detected_source_identifiers
        )

        if orphan_measurement_count > 0:
            self._logger.warning(
                "%d photometric measurement records do not "
                "correspond to a detected source.",
                orphan_measurement_count,
            )

        #
        # Count detected sources for which a measurement record exists.
        #
        measured_source_count = sum(
            1
            for source in detected_sources
            if source.identifier in measurement_by_source
        )

        missing_measurement_count = (
            detected_source_count
            - measured_source_count
        )

        if missing_measurement_count > 0:
            self._logger.warning(
                "%d detected sources have no photometric "
                "measurement record.",
                missing_measurement_count,
            )

        #
        # Determine crowding independently from measurement validity.
        #
        neighbour_counts = self._calculate_neighbour_counts(
            detected_sources,
        )

        #
        # Build one immutable source-level evaluation record for every
        # detected source.
        #
        source_evaluations = tuple(
            self._evaluate_source(
                source=source,
                measurement=measurement_by_source.get(
                    source.identifier,
                ),
                neighbour_count=neighbour_counts.get(
                    source.identifier,
                    0,
                ),
            )
            for source in detected_sources
        )

        #
        # Separate produced measurements into valid and invalid
        # measurement records.
        #
        measured_source_evaluations = tuple(
            evaluation
            for evaluation in source_evaluations
            if evaluation.quality_flag
            != self._MISSING_MEASUREMENT_FLAG
        )

        valid_source_evaluations = tuple(
            evaluation
            for evaluation in measured_source_evaluations
            if evaluation.valid_measurement
        )

        invalid_source_evaluations = tuple(
            evaluation
            for evaluation in measured_source_evaluations
            if not evaluation.valid_measurement
        )

        valid_measurement_count = len(
            valid_source_evaluations,
        )

        invalid_measurement_count = len(
            invalid_source_evaluations,
        )

        #
        # Measurement completeness describes whether a measurement
        # record was produced for each detected source.
        #
        measurement_completeness = self._calculate_fraction(
            measured_source_count,
            detected_source_count,
        )

        #
        # Valid measurement fraction describes the proportion of
        # produced measurement records that passed photometric QC.
        #
        valid_measurement_fraction = self._calculate_fraction(
            valid_measurement_count,
            measured_source_count,
        )

        #
        # Non-positive flux is treated separately from general
        # measurement validity because zero and negative flux cannot
        # produce a finite instrumental magnitude.
        #
        non_positive_flux_count = self._count_non_positive_flux(
            measurements,
        )

        #
        # Crowding is based on all detected sources rather than only
        # successfully measured sources.
        #
        crowded_source_count = sum(
            1
            for evaluation in source_evaluations
            if evaluation.crowded
        )

        crowding_fraction = self._calculate_fraction(
            crowded_source_count,
            detected_source_count,
        )

        #
        # Summary photometric statistics are calculated from valid
        # source evaluations only.
        #
        median_net_flux = self._calculate_median(
            evaluation.net_flux
            for evaluation in valid_source_evaluations
        )

        median_background_level = self._calculate_median(
            evaluation.background_level
            for evaluation in valid_source_evaluations
        )

        median_background_fraction = self._calculate_median(
            evaluation.background_fraction
            for evaluation in valid_source_evaluations
        )

        evaluation = PhotometryEvaluation(
            detected_source_count=detected_source_count,
            measured_source_count=measured_source_count,
            valid_measurement_count=valid_measurement_count,
            measurement_completeness=measurement_completeness,
            valid_measurement_fraction=valid_measurement_fraction,
            invalid_measurement_count=invalid_measurement_count,
            missing_measurement_count=missing_measurement_count,
            non_positive_flux_count=non_positive_flux_count,
            crowded_source_count=crowded_source_count,
            crowding_fraction=crowding_fraction,
            crowding_radius=float(
                self._config.crowding_radius,
            ),
            median_net_flux=median_net_flux,
            median_background_level=median_background_level,
            median_background_fraction=(
                median_background_fraction
            ),
            algorithm=observation.photometry.algorithm,
            evaluation_timestamp=datetime.now(
                timezone.utc,
            ),
            source_evaluations=source_evaluations,
        )

        self._logger.info(
            "Photometry evaluation completed successfully."
        )

        self._logger.info(
            "Detected sources: %d.",
            detected_source_count,
        )

        self._logger.info(
            "Measurement records: %d.",
            measured_source_count,
        )

        self._logger.info(
            "Valid measurements: %d.",
            valid_measurement_count,
        )

        self._logger.info(
            "Invalid measurements: %d.",
            invalid_measurement_count,
        )

        self._logger.info(
            "Missing measurements: %d.",
            missing_measurement_count,
        )

        self._logger.info(
            "Measurement completeness: %.2f%%.",
            measurement_completeness * 100.0,
        )

        self._logger.info(
            "Valid measurement fraction: %.2f%%.",
            valid_measurement_fraction * 100.0,
        )

        self._logger.info(
            "Non-positive net flux measurements: %d.",
            non_positive_flux_count,
        )

        self._logger.info(
            "Crowded sources: %d of %d (%.2f%%).",
            crowded_source_count,
            detected_source_count,
            crowding_fraction * 100.0,
        )

        return evaluation

    def _evaluate_source(
        self,
        *,
        source: DetectedSource,
        measurement: PhotometryMeasurement | None,
        neighbour_count: int,
    ) -> SourceEvaluation:
        """
        Create a source-level evaluation record.

        Parameters
        ----------
        source
            Detected source being evaluated.

        measurement
            Photometric measurement corresponding to the source, or
            ``None`` when no measurement record exists.

        neighbour_count
            Number of neighbouring detected sources within the
            configured crowding radius.

        Returns
        -------
        SourceEvaluation
            Immutable evaluation record for the source.
        """

        crowded = neighbour_count > 0

        if measurement is None:
            return SourceEvaluation(
                source_identifier=source.identifier,
                valid_measurement=False,
                quality_flag=self._MISSING_MEASUREMENT_FLAG,
                net_flux=None,
                instrumental_magnitude=None,
                background_level=None,
                background_fraction=None,
                aperture_coverage=None,
                annulus_coverage=None,
                neighbour_count=neighbour_count,
                crowded=crowded,
            )

        #
        # Only expose finite numerical measurement values to the
        # evaluation model. This prevents NaN and infinity from
        # silently entering aggregate statistics.
        #
        net_flux = self._finite_or_none(
            measurement.net_flux,
        )

        instrumental_magnitude = self._finite_or_none(
            measurement.instrumental_magnitude,
        )

        background_level = self._finite_or_none(
            measurement.background_level,
        )

        background_fraction = self._finite_or_none(
            measurement.background_fraction,
        )

        aperture_coverage = self._finite_or_none(
            measurement.aperture_coverage,
        )

        annulus_coverage = self._finite_or_none(
            measurement.annulus_coverage,
        )

        return SourceEvaluation(
            source_identifier=source.identifier,
            valid_measurement=measurement.valid,
            quality_flag=measurement.quality_flag,
            net_flux=net_flux,
            instrumental_magnitude=instrumental_magnitude,
            background_level=background_level,
            background_fraction=background_fraction,
            aperture_coverage=aperture_coverage,
            annulus_coverage=annulus_coverage,
            neighbour_count=neighbour_count,
            crowded=crowded,
        )

    @staticmethod
    def _index_measurements(
        measurements: tuple[PhotometryMeasurement, ...],
    ) -> dict[str, PhotometryMeasurement]:
        """
        Index photometric measurements by source identifier.

        Parameters
        ----------
        measurements
            Immutable collection of photometric measurements.

        Returns
        -------
        dict[str, PhotometryMeasurement]
            Measurements indexed by source identifier.

        Raises
        ------
        ValueError
            If a measurement contains an empty identifier or duplicate
            source identifiers are present.
        """

        indexed: dict[str, PhotometryMeasurement] = {}

        for measurement in measurements:
            identifier = measurement.source_identifier

            if not identifier:
                raise ValueError(
                    "Photometric measurement contains an empty "
                    "source identifier."
                )

            if identifier in indexed:
                raise ValueError(
                    "Duplicate photometric measurement found for "
                    f"source '{identifier}'."
                )

            indexed[identifier] = measurement

        return indexed

    @staticmethod
    def _validate_detected_sources(
        sources: tuple[DetectedSource, ...],
    ) -> None:
        """
        Validate detected-source identifiers and centroid coordinates.

        Parameters
        ----------
        sources
            Detected sources.

        Raises
        ------
        ValueError
            If duplicate identifiers, empty identifiers or
            non-finite centroid coordinates are present.
        """

        identifiers: set[str] = set()

        for source in sources:
            if not source.identifier:
                raise ValueError(
                    "Detected source contains an empty identifier."
                )

            if source.identifier in identifiers:
                raise ValueError(
                    "Duplicate detected source identifier found: "
                    f"'{source.identifier}'."
                )

            identifiers.add(
                source.identifier,
            )

            if (
                not isfinite(
                    float(source.x_centroid),
                )
                or not isfinite(
                    float(source.y_centroid),
                )
            ):
                raise ValueError(
                    "Detected source contains non-finite centroid "
                    f"coordinates: '{source.identifier}'."
                )

    def _validate_configuration(
        self,
    ) -> None:
        """
        Validate evaluation configuration.

        Raises
        ------
        ValueError
            If the crowding radius is invalid.
        """

        crowding_radius = float(
            self._config.crowding_radius,
        )

        if (
            not isfinite(crowding_radius)
            or crowding_radius <= 0.0
        ):
            raise ValueError(
                "Crowding radius must be a finite value "
                "greater than zero."
            )

    @staticmethod
    def _validate_observation(
        observation: Observation,
    ) -> None:
        """
        Validate that the Observation contains the prerequisites
        required for evaluation.

        Raises
        ------
        ValueError
            If the Observation does not contain source detections,
            photometric measurements, or completed processing state.
        """

        if observation is None:
            raise ValueError(
                "Observation cannot be None."
            )

        if observation.detections is None:
            raise ValueError(
                "Observation contains no source detections."
            )

        if observation.photometry is None:
            raise ValueError(
                "Observation contains no photometric result."
            )

        if not observation.processing_state.source_detection_completed:
            raise ValueError(
                "Source detection is not marked as completed."
            )

        if not observation.processing_state.photometry_completed:
            raise ValueError(
                "Photometry is not marked as completed."
            )

    @staticmethod
    def _count_non_positive_flux(
        measurements: tuple[PhotometryMeasurement, ...],
    ) -> int:
        """
        Count measurements with non-positive net flux.

        Zero and negative net flux are both included because neither
        can produce a finite instrumental magnitude.

        Non-finite values are excluded from this metric because they
        represent a different quality failure.
        """

        count = 0

        for measurement in measurements:
            net_flux = measurement.net_flux

            if net_flux is None:
                continue

            numeric_value = float(
                net_flux,
            )

            if not isfinite(numeric_value):
                continue

            if numeric_value <= 0.0:
                count += 1

        return count

    def calculate_neighbour_counts(
        self,
        sources: tuple[DetectedSource, ...],
    ) -> dict[str, int]:
        """
        Calculate the number of neighbouring detected sources for each source.

        This exposes the existing crowding calculation for field-level
        analysis without requiring target resolution or photometric evaluation.

        The underlying neighbour-count algorithm is shared with the normal
        photometric evaluation path and is intentionally unchanged.
        """
        return self._calculate_neighbour_counts(sources)

    def _calculate_neighbour_counts(
        self,
        sources: tuple[DetectedSource, ...],
    ) -> dict[str, int]:
        """
        Calculate the number of nearby neighbours for every source.

        A source is considered to have a neighbour when another
        detected source lies within the configured crowding radius.

        The source itself is excluded from the comparison.

        Parameters
        ----------
        sources
            Detected sources in the Observation.

        Returns
        -------
        dict[str, int]
            Number of neighbours for each source identifier.
        """

        neighbour_counts = {
            source.identifier: 0
            for source in sources
        }

        if len(sources) < 2:
            return neighbour_counts

        radius = float(
            self._config.crowding_radius,
        )

        #
        # Evaluate each unique source pair once and update both
        # source neighbour counts.
        #
        for index, source in enumerate(sources):
            for neighbour in sources[index + 1:]:
                distance = hypot(
                    float(source.x_centroid)
                    - float(neighbour.x_centroid),
                    float(source.y_centroid)
                    - float(neighbour.y_centroid),
                )

                if distance <= radius:
                    neighbour_counts[
                        source.identifier
                    ] += 1

                    neighbour_counts[
                        neighbour.identifier
                    ] += 1

        return neighbour_counts

    @staticmethod
    def _calculate_fraction(
        numerator: int,
        denominator: int,
    ) -> float:
        """
        Calculate a fraction safely.

        Parameters
        ----------
        numerator
            Numerator of the fraction.

        denominator
            Denominator of the fraction.

        Returns
        -------
        float
            Fraction between 0.0 and 1.0.

        Notes
        -----
        A zero denominator returns 0.0 because there is no population
        from which to calculate the corresponding fraction.
        """

        if denominator <= 0:
            return 0.0

        fraction = numerator / denominator

        return float(
            max(
                0.0,
                min(
                    1.0,
                    fraction,
                ),
            )
        )

    @staticmethod
    def _calculate_median(
        values: Iterable[float | None],
    ) -> float | None:
        """
        Calculate the median of finite values.

        Parameters
        ----------
        values
            Iterable of numerical values.

        Returns
        -------
        float | None
            Median value, or ``None`` when no finite values exist.
        """

        finite_values: list[float] = []

        for value in values:
            if value is None:
                continue

            numeric_value = float(
                value,
            )

            if not isfinite(numeric_value):
                continue

            finite_values.append(
                numeric_value,
            )

        if not finite_values:
            return None

        return float(
            median(
                finite_values,
            )
        )

    @staticmethod
    def _finite_or_none(
        value: float | None,
    ) -> float | None:
        """
        Return a finite numerical value or ``None``.

        This prevents non-finite floating-point values from entering
        evaluation records.
        """

        if value is None:
            return None

        numeric_value = float(
            value,
        )

        if not isfinite(numeric_value):
            return None

        return numeric_value