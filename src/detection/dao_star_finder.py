"""
DAOStarFinder source detector.

This module provides a concrete implementation of the survey-independent
SourceDetector interface using Photutils' DAOStarFinder algorithm.

The detector operates exclusively on the Observation domain model and
identifies point-like astronomical sources within the active image.

An independent local-peak diagnostic is available at DEBUG logging level.
This diagnostic does not modify the production detection result. It is used
to distinguish genuine local image peaks from sources rejected by the
DAOStarFinder point-source morphology criteria.

This implementation is intentionally independent of any specific survey.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, cast

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from numpy.typing import NDArray
from photutils.detection import DAOStarFinder, find_peaks

from core.models import DetectionConfig
from detection.detection import SourceDetector
from domain.models import (
    DetectedSource,
    DetectionResult,
    Observation,
)


class DAOStarFinderDetector(SourceDetector):
    """
    Detect astronomical sources using Photutils' DAOStarFinder algorithm.

    Parameters
    ----------
    config
        Immutable source detection configuration.

    logger
        Optional logger used for diagnostic and processing messages.

    Notes
    -----
    The production source list is generated exclusively by
    ``DAOStarFinder``. The local-peak analysis performed at DEBUG level is
    diagnostic only and never contributes sources to the returned
    ``DetectionResult``.
    """

    # ------------------------------------------------------------------
    # Diagnostic configuration.
    # ------------------------------------------------------------------

    # Small local neighbourhood used to identify compact local maxima.
    #
    # This is deliberately independent of DAOStarFinder so that the
    # diagnostic can distinguish:
    #
    #     bright local peak
    #
    # from:
    #
    #     DAOStarFinder-accepted point source.
    #
    _DIAGNOSTIC_PEAK_BOX_SIZE = 3

    # Prevent DEBUG logging from becoming excessive for large,
    # crowded-field images.
    _MAX_DIAGNOSTIC_PEAKS = 20

    def __init__(
        self,
        config: DetectionConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the DAOStarFinder detector.

        Parameters
        ----------
        config
            Source detection configuration.

        logger
            Optional logger.
        """

        super().__init__(
            logger,
        )

        self._config = config

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def detect(
        self,
        observation: Observation,
    ) -> Observation:
        """
        Detect astronomical sources within an observation.

        Parameters
        ----------
        observation
            Observation whose active image will be analysed.

        Returns
        -------
        Observation
            Observation updated with source detection results.

        Raises
        ------
        ValueError
            If the observation is not suitable for source detection.
        """

        self._logger.info(
            "Starting DAOStarFinder source detection."
        )

        self._validate_observation(
            observation,
        )

        image_data = cast(
            NDArray[np.floating],
            observation.active.image.data,
        )

        #
        # Basic image diagnostics are DEBUG-only so that normal production
        # execution remains concise.
        #
        self._log_image_diagnostics(
            image_data,
        )

        threshold = self._compute_detection_threshold(
            image_data,
        )

        self._logger.info(
            "DAOStarFinder configuration: "
            "FWHM=%.3f pixels, threshold_sigma=%.3f.",
            float(self._config.fwhm),
            float(
                self._config.detection_threshold_sigma,
            ),
        )

        self._logger.info(
            "Detection threshold: %.6f.",
            threshold,
        )

        #
        # Compare the image's strongest pixels with the detection threshold.
        #
        self._log_threshold_diagnostics(
            image_data=image_data,
            threshold=threshold,
        )

        #
        # Independent diagnostic only.
        #
        # This does NOT contribute any sources to the production result.
        #
        self._run_local_peak_diagnostic(
            image_data=image_data,
            threshold=threshold,
            mask=observation.active.image.mask,
        )

        #
        # DAOStarFinder remains the sole source of production detections.
        #
        detections = self._run_dao_star_finder(
            image_data=image_data,
            threshold=threshold,
        )

        self._log_dao_detections(
            detections,
        )

        sources = self._map_detected_sources(
            detections,
        )

        observation.detections = DetectionResult(
            sources=tuple(
                sources,
            ),
            algorithm="DAOStarFinder",
            detection_threshold=threshold,
            detection_timestamp=datetime.now(
                timezone.utc,
            ),
        )

        observation.processing_state.source_detection_completed = True

        self._logger.info(
            "Source detection completed successfully "
            "with %d detected sources.",
            len(sources),
        )

        return observation

    # ------------------------------------------------------------------
    # Detection diagnostics.
    # ------------------------------------------------------------------

    def _log_image_diagnostics(
        self,
        image_data: NDArray[np.floating],
    ) -> None:
        """
        Log basic image statistics at DEBUG level.

        Parameters
        ----------
        image_data
            Two-dimensional active image.
        """

        if not self._logger.isEnabledFor(
            logging.DEBUG,
        ):
            return

        finite_mask = np.isfinite(
            image_data,
        )

        finite_count = int(
            np.count_nonzero(
                finite_mask,
            )
        )

        total_count = int(
            image_data.size,
        )

        self._logger.debug(
            "Image shape: %s.",
            image_data.shape,
        )

        self._logger.debug(
            "Image dtype: %s.",
            image_data.dtype,
        )

        self._logger.debug(
            "Finite image pixels: %d of %d.",
            finite_count,
            total_count,
        )

        if finite_count == 0:
            self._logger.debug(
                "Image contains no finite pixels."
            )
            return

        finite_values = image_data[
            finite_mask
        ]

        self._logger.debug(
            "Image minimum: %.6f.",
            float(
                np.min(
                    finite_values,
                )
            ),
        )

        self._logger.debug(
            "Image maximum: %.6f.",
            float(
                np.max(
                    finite_values,
                )
            ),
        )

        #
        # Report the brightest pixel when all image pixels are finite.
        # This avoids reporting an incorrect coordinate when the finite
        # subset is smaller than the complete image.
        #
        if finite_count == total_count:

            maximum_index = int(
                np.argmax(
                    finite_values,
                )
            )

            maximum_position = np.unravel_index(
                maximum_index,
                image_data.shape,
            )

            y_position = int(
                maximum_position[0],
            )

            x_position = int(
                maximum_position[1],
            )

            self._logger.debug(
                "Brightest pixel: x=%d y=%d value=%.6f.",
                x_position,
                y_position,
                float(
                    finite_values[
                        maximum_index
                    ]
                ),
            )

        _, median, standard_deviation = sigma_clipped_stats(
            image_data,
        )

        self._logger.debug(
            "Sigma-clipped image statistics: "
            "median=%.6f std=%.6f.",
            float(median),
            float(standard_deviation),
        )

    def _log_threshold_diagnostics(
        self,
        *,
        image_data: NDArray[np.floating],
        threshold: float,
    ) -> None:
        """
        Log diagnostics relating image brightness to the detection
        threshold.

        Parameters
        ----------
        image_data
            Two-dimensional active image.

        threshold
            Absolute DAOStarFinder detection threshold.
        """

        if not self._logger.isEnabledFor(
            logging.DEBUG,
        ):
            return

        finite_values = image_data[
            np.isfinite(image_data)
        ]

        if finite_values.size == 0:
            return

        maximum_value = float(
            np.max(
                finite_values,
            )
        )

        if threshold <= 0.0:
            self._logger.debug(
                "Detection threshold is non-positive; "
                "threshold ratio is unavailable."
            )
            return

        self._logger.debug(
            "Brightest pixel / detection threshold ratio: %.3f.",
            maximum_value / threshold,
        )

    def _run_local_peak_diagnostic(
        self,
        *,
        image_data: NDArray[np.floating],
        threshold: float,
        mask: np.ndarray | None,
    ) -> None:
        """
        Independently identify strong local image peaks for diagnostics.

        This method is diagnostic only. Its results are never added to the
        ``DetectionResult``.

        Parameters
        ----------
        image_data
            Two-dimensional background-subtracted image.

        threshold
            Absolute detection threshold.

        mask
            Optional image mask. ``True`` values are ignored.

        Notes
        -----
        ``find_peaks`` identifies local maxima above the supplied threshold,
        whereas DAOStarFinder additionally evaluates point-source
        morphology. Comparing the two results helps determine whether a
        bright local maximum is being rejected by DAOStarFinder rather than
        being absent from the image.
        """

        if not self._logger.isEnabledFor(
            logging.DEBUG,
        ):
            return

        self._logger.debug(
            "Running independent local-peak diagnostic "
            "with box_size=%d and threshold=%.6f.",
            self._DIAGNOSTIC_PEAK_BOX_SIZE,
            threshold,
        )

        try:

            peaks = find_peaks(
                image_data,
                threshold=threshold,
                box_size=self._DIAGNOSTIC_PEAK_BOX_SIZE,
                mask=mask,
            )

        except Exception:
            self._logger.exception(
                "Local-peak diagnostic failed."
            )
            return

        if peaks is None or len(peaks) == 0:

            self._logger.debug(
                "Local-peak diagnostic found no peaks above "
                "the detection threshold."
            )

            return

        self._logger.debug(
            "Local-peak diagnostic found %d peaks above "
            "the detection threshold.",
            len(peaks),
        )

        #
        # Astropy Table iteration is not sufficiently typed for Pylance.
        # Explicitly cast each row to Any for diagnostic access.
        #
        peak_rows: list[Any] = []

        for index in range(
            len(peaks),
        ):

            row = cast(
                Any,
                peaks[index],
            )

            peak_rows.append(
                row,
            )

        #
        # Strongest local maxima are reported first.
        #
        peak_rows.sort(
            key=lambda row: float(
                row["peak_value"],
            ),
            reverse=True,
        )

        diagnostic_rows = peak_rows[
            : self._MAX_DIAGNOSTIC_PEAKS
        ]

        for index, row in enumerate(
            diagnostic_rows,
            start=1,
        ):

            self._logger.debug(
                "Local peak %d: "
                "x=%s y=%s peak=%.6f.",
                index,
                row["x_peak"],
                row["y_peak"],
                float(
                    row["peak_value"],
                ),
            )

        if len(peaks) > self._MAX_DIAGNOSTIC_PEAKS:

            self._logger.debug(
                "Local-peak diagnostic logging limited to "
                "the %d strongest peaks.",
                self._MAX_DIAGNOSTIC_PEAKS,
            )

    def _log_dao_detections(
        self,
        detections: Table | None,
    ) -> None:
        """
        Log DAOStarFinder output and source morphology diagnostics.

        Parameters
        ----------
        detections
            DAOStarFinder detection table.
        """

        if detections is None:

            self._logger.info(
                "DAOStarFinder returned no detections."
            )

            return

        self._logger.info(
            "DAOStarFinder returned %d detections.",
            len(detections),
        )

        if not self._logger.isEnabledFor(
            logging.DEBUG,
        ):
            return

        self._logger.debug(
            "DAOStarFinder columns: %s.",
            detections.colnames,
        )

        for index in range(
            len(detections),
        ):

            row = cast(
                Any,
                detections[index],
            )

            self._logger.debug(
                "Candidate %d: "
                "x=%.3f y=%.3f "
                "peak=%.3f flux=%.3f "
                "sharpness=%.3f "
                "roundness1=%.3f "
                "roundness2=%.3f.",
                index + 1,
                float(
                    row["x_centroid"],
                ),
                float(
                    row["y_centroid"],
                ),
                float(
                    row["peak"],
                ),
                float(
                    row["flux"],
                ),
                float(
                    row["sharpness"],
                ),
                float(
                    row["roundness1"],
                ),
                float(
                    row["roundness2"],
                ),
            )

    # ------------------------------------------------------------------
    # Domain mapping.
    # ------------------------------------------------------------------

    @staticmethod
    def _create_detected_source(
        *,
        identifier: int,
        x_centroid: float,
        y_centroid: float,
        peak_value: float,
    ) -> DetectedSource:
        """
        Create a survey-independent detected source.

        Parameters
        ----------
        identifier
            Sequential identifier within the detection result.

        x_centroid
            X-coordinate of the source centroid.

        y_centroid
            Y-coordinate of the source centroid.

        peak_value
            Peak pixel value of the detected source.

        Returns
        -------
        DetectedSource
            Immutable detected source domain model.
        """

        return DetectedSource(
            identifier=str(
                identifier,
            ),
            x_centroid=x_centroid,
            y_centroid=y_centroid,
            peak_value=peak_value,
        )

    @staticmethod
    def _map_detected_sources(
        detections: Table | None,
    ) -> list[DetectedSource]:
        """
        Convert DAOStarFinder detections into survey-independent
        domain models.

        Parameters
        ----------
        detections
            DAOStarFinder detection table.

        Returns
        -------
        list[DetectedSource]
            Detected sources extracted from the detection table.
        """

        if detections is None:
            return []

        sources: list[DetectedSource] = []

        for index in range(
            len(detections),
        ):

            row = cast(
                Any,
                detections[index],
            )

            sources.append(
                DAOStarFinderDetector._create_detected_source(
                    identifier=index + 1,
                    x_centroid=float(
                        row["x_centroid"],
                    ),
                    y_centroid=float(
                        row["y_centroid"],
                    ),
                    peak_value=float(
                        row["peak"],
                    ),
                )
            )

        return sources

    # ------------------------------------------------------------------
    # Validation.
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_observation(
        observation: Observation,
    ) -> None:
        """
        Validate that an observation is suitable for source detection.

        Parameters
        ----------
        observation
            Observation to validate.

        Raises
        ------
        ValueError
            If the observation is invalid or does not contain a usable
            active image.
        """

        if observation is None:

            raise ValueError(
                "Observation cannot be None."
            )

        if observation.active is None:

            raise ValueError(
                "Observation does not contain an active image."
            )

        active_image = observation.active.image

        if active_image is None:

            raise ValueError(
                "Active image is missing."
            )

        image_data = active_image.data

        if image_data is None:

            raise ValueError(
                "Active image does not contain image data."
            )

        if not isinstance(
            image_data,
            np.ndarray,
        ):

            raise ValueError(
                "Active image must contain a NumPy ndarray."
            )

        if image_data.ndim != 2:

            raise ValueError(
                "Source detection requires a two-dimensional image."
            )

        if image_data.size == 0:

            raise ValueError(
                "Source detection requires a non-empty image."
            )

        if not np.issubdtype(
            image_data.dtype,
            np.number,
        ):

            raise ValueError(
                "Active image must contain numeric data."
            )

        if not np.any(
            np.isfinite(
                image_data,
            )
        ):

            raise ValueError(
                "Active image contains no finite pixels."
            )

    # ------------------------------------------------------------------
    # Threshold and DAOStarFinder execution.
    # ------------------------------------------------------------------

    def _compute_detection_threshold(
        self,
        image_data: NDArray[np.floating],
    ) -> float:
        """
        Compute the absolute detection threshold for the active image.

        The threshold is calculated from the sigma-clipped standard
        deviation of the image and the configured detection threshold
        multiplier.

        Parameters
        ----------
        image_data
            Two-dimensional image array.

        Returns
        -------
        float
            Absolute detection threshold.
        """

        _, _, standard_deviation = sigma_clipped_stats(
            image_data,
        )

        if not np.isfinite(
            standard_deviation,
        ):

            raise ValueError(
                "Unable to compute a finite image standard deviation."
            )

        threshold = (
            float(
                self._config.detection_threshold_sigma,
            )
            * float(
                standard_deviation,
            )
        )

        if not np.isfinite(
            threshold,
        ):

            raise ValueError(
                "Computed detection threshold is not finite."
            )

        if threshold <= 0.0:

            raise ValueError(
                "Computed detection threshold must be greater than zero."
            )

        return threshold

    def _run_dao_star_finder(
        self,
        *,
        image_data: NDArray[np.floating],
        threshold: float,
    ) -> Table | None:
        """
        Execute the DAOStarFinder algorithm.

        Parameters
        ----------
        image_data
            Two-dimensional image array.

        threshold
            Absolute detection threshold.

        Returns
        -------
        Table | None
            Astropy table containing detected sources, or ``None`` if no
            sources are detected.
        """

        self._logger.debug(
            "Executing DAOStarFinder with "
            "threshold=%.6f and fwhm=%.6f.",
            threshold,
            float(
                self._config.fwhm,
            ),
        )

        dao_star_finder = DAOStarFinder(
            threshold=threshold,
            fwhm=float(
                self._config.fwhm,
            ),
        )

        return dao_star_finder(
            image_data,
        )