"""
Utilities for inspecting Observation objects.

This module provides diagnostic utilities for validating observations
during development, experimentation and demonstration workflows.

The inspector performs no scientific processing and never modifies the
supplied Observation. It reports:

    * scientific target metadata;
    * observation provenance;
    * available image products;
    * active-image statistics;
    * source-detection state;
    * photometry configuration and state;
    * target-source resolution, when available;
    * already-computed photometric evaluation metrics;
    * overall processing state.

Scientific calculations remain the responsibility of the production
processing and evaluation components. The inspector only reports values
that are already present in the supplied domain objects.

The inspector is therefore suitable for:

    * development;
    * debugging;
    * experiment validation;
    * interactive demonstrations;
    * thesis workflow inspection.

It does not:

    * modify an Observation;
    * perform source detection;
    * perform photometry;
    * calculate evaluation metrics;
    * resolve astronomical targets;
    * generate figures;
    * persist results.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray

from domain.models import (
    Observation,
    TargetSourceMatch,
)
from evaluation.models import PhotometryEvaluation
from utils.logger import get_logger


class ObservationInspector:
    """
    Inspect an Observation and log useful diagnostic information.

    The inspector reports the state and products already produced by
    the photometric workflow. It does not perform scientific processing
    or mutate the supplied objects.

    Parameters
    ----------
    logger
        Optional logger. If omitted, the project logger is used.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or get_logger(
            __name__,
        )

    def inspect(
        self,
        observation: Observation,
        evaluation: PhotometryEvaluation | None = None,
        target_match: TargetSourceMatch | None = None,
    ) -> None:
        """
        Log a diagnostic summary of an Observation.

        Parameters
        ----------
        observation
            Observation to inspect.

        evaluation
            Optional completed photometric evaluation. When supplied,
            the inspector reports the evaluation metrics already
            calculated by PhotometryEvaluator.

        target_match
            Optional target-source association already produced by the
            target-resolution stage of the workflow.

        Raises
        ------
        ValueError
            If the supplied Observation is invalid.
        """

        if observation is None:
            raise ValueError(
                "Observation cannot be None.",
            )

        image = observation.active.image

        image_data = cast(
            NDArray[np.number],
            np.asarray(
                image.data,
            ),
        )

        self._logger.info("")
        self._logger.info(
            "=" * 72,
        )
        self._logger.info(
            "OBSERVATION SUMMARY",
        )
        self._logger.info(
            "=" * 72,
        )

        self._log_target_information(
            observation,
        )

        self._log_provenance(
            observation,
        )

        self._log_image_products(
            observation,
        )

        self._log_active_image(
            observation,
            image_data,
        )

        self._log_detection(
            observation,
        )

        self._log_photometry(
            observation,
        )

        self._log_target_resolution(
            target_match,
        )

        self._log_evaluation(
            evaluation,
        )

        self._log_processing_state(
            observation,
        )

        self._logger.info(
            "=" * 72,
        )

    def _log_target_information(
        self,
        observation: Observation,
    ) -> None:
        """
        Log scientific target and observation metadata.
        """

        metadata = observation.metadata

        self._logger.info("")
        self._logger.info(
            "Target Information",
        )
        self._logger.info(
            "------------------",
        )

        self._logger.info(
            "Target            : %s",
            metadata.target_name,
        )

        self._logger.info(
            "Right Ascension   : %s",
            (
                f"{metadata.right_ascension:.6f}°"
                if metadata.right_ascension is not None
                else "Unknown"
            ),
        )

        self._logger.info(
            "Declination       : %s",
            (
                f"{metadata.declination:.6f}°"
                if metadata.declination is not None
                else "Unknown"
            ),
        )

        self._logger.info(
            "Instrument        : %s",
            metadata.instrument,
        )

        self._logger.info(
            "Filter            : %s",
            metadata.filter_name,
        )

        self._logger.info(
            "Observation Time  : %s",
            metadata.observation_time,
        )

        self._logger.info(
            "Exposure Time     : %s",
            metadata.exposure_time,
        )

    def _log_provenance(
        self,
        observation: Observation,
    ) -> None:
        """
        Log observation provenance.
        """

        provenance = observation.provenance

        self._logger.info("")
        self._logger.info(
            "Provenance",
        )
        self._logger.info(
            "----------",
        )

        self._logger.info(
            "Survey            : %s",
            provenance.survey.value,
        )

        self._logger.info(
            "Source            : %s",
            provenance.source.value,
        )

        self._logger.info(
            "Dataset           : %s",
            provenance.dataset_identifier,
        )

        self._logger.info(
            "Filename          : %s",
            provenance.filename,
        )

        self._logger.info(
            "Downloaded        : %s",
            provenance.download_timestamp.isoformat(),
        )

    def _log_image_products(
        self,
        observation: Observation,
    ) -> None:
        """
        Log image products currently stored in the Observation.
        """

        image_names = list(
            observation.images.keys(),
        )

        self._logger.info("")
        self._logger.info(
            "Image Products",
        )
        self._logger.info(
            "--------------",
        )

        if not image_names:
            self._logger.info(
                "Stored Images    : None",
            )
            return

        self._logger.info(
            "Stored Images    : %d",
            len(image_names),
        )

        self._logger.info(
            "Image Names      : %s",
            ", ".join(image_names),
        )

        self._logger.info(
            "Active Image     : %s",
            self._find_active_image_name(
                observation,
            ),
        )

    def _find_active_image_name(
        self,
        observation: Observation,
    ) -> str:
        """
        Determine the name of the active image.

        Returns
        -------
        str
            Stored image name if the active image is present in the
            image collection, otherwise ``"unknown"``.
        """

        for name, image_data in observation.images.items():

            if image_data is observation.active:
                return name

        return "unknown"

    def _log_active_image(
        self,
        observation: Observation,
        image_data: NDArray[np.number],
    ) -> None:
        """
        Log statistics describing the active image.

        The statistics are diagnostic pixel statistics only. They do
        not contribute to scientific evaluation.
        """

        image = observation.active.image

        self._logger.info("")
        self._logger.info(
            "Active Image",
        )
        self._logger.info(
            "------------",
        )

        self._logger.info(
            "Shape             : %s",
            image_data.shape,
        )

        self._logger.info(
            "Dimensions        : %d",
            image_data.ndim,
        )

        self._logger.info(
            "Data Type         : %s",
            image_data.dtype,
        )

        self._logger.info(
            "Units             : %s",
            image.unit,
        )

        self._logger.info(
            "WCS Available     : %s",
            image.wcs is not None,
        )

        total_pixels = image_data.size

        if total_pixels == 0:

            self._logger.info(
                "Valid Pixels      : 0 / 0",
            )

            self._logger.info(
                "Invalid Pixels    : 0",
            )

            self._logger.info(
                "Minimum           : Unknown",
            )

            self._logger.info(
                "Maximum           : Unknown",
            )

            self._logger.info(
                "Mean              : Unknown",
            )

            self._logger.info(
                "Median            : Unknown",
            )

            self._logger.info(
                "Std Dev           : Unknown",
            )

            return

        finite_mask = np.isfinite(
            image_data,
        )

        finite_count = int(
            finite_mask.sum(),
        )

        invalid_count = (
            total_pixels
            - finite_count
        )

        self._logger.info(
            "Valid Pixels      : %d / %d",
            finite_count,
            total_pixels,
        )

        self._logger.info(
            "Invalid Pixels    : %d (%.2f%%)",
            invalid_count,
            100.0
            * invalid_count
            / total_pixels,
        )

        if finite_count == 0:

            self._logger.info(
                "Minimum           : Unknown "
                "(no finite pixels)",
            )

            self._logger.info(
                "Maximum           : Unknown "
                "(no finite pixels)",
            )

            self._logger.info(
                "Mean              : Unknown "
                "(no finite pixels)",
            )

            self._logger.info(
                "Median            : Unknown "
                "(no finite pixels)",
            )

            self._logger.info(
                "Std Dev           : Unknown "
                "(no finite pixels)",
            )

            return

        finite_values = image_data[
            finite_mask
        ]

        self._logger.info(
            "Minimum           : %.6f",
            float(
                np.min(
                    finite_values,
                ),
            ),
        )

        self._logger.info(
            "Maximum           : %.6f",
            float(
                np.max(
                    finite_values,
                ),
            ),
        )

        self._logger.info(
            "Mean              : %.6f",
            float(
                np.mean(
                    finite_values,
                ),
            ),
        )

        self._logger.info(
            "Median            : %.6f",
            float(
                np.median(
                    finite_values,
                ),
            ),
        )

        self._logger.info(
            "Std Dev           : %.6f",
            float(
                np.std(
                    finite_values,
                ),
            ),
        )

    def _log_detection(
        self,
        observation: Observation,
    ) -> None:
        """
        Log source-detection results already contained in the
        Observation.
        """

        self._logger.info("")
        self._logger.info(
            "Detection",
        )
        self._logger.info(
            "---------",
        )

        detection_result = observation.detections

        if detection_result is None:

            self._logger.info(
                "Completed         : False",
            )

            self._logger.info(
                "Detected Sources  : 0",
            )

            return

        self._logger.info(
            "Completed         : True",
        )

        self._logger.info(
            "Algorithm         : %s",
            detection_result.algorithm,
        )

        self._logger.info(
            "Threshold         : %.6f",
            detection_result.detection_threshold,
        )

        self._logger.info(
            "Detected Sources  : %d",
            len(
                detection_result.sources,
            ),
        )

        self._logger.info(
            "Detection Time    : %s",
            detection_result.detection_timestamp.isoformat(),
        )

    def _log_photometry(
        self,
        observation: Observation,
    ) -> None:
        """
        Log photometry results already contained in the Observation.

        The inspector reports configuration and processing state only.
        Scientific evaluation metrics are reported separately from the
        supplied PhotometryEvaluation.
        """

        self._logger.info("")
        self._logger.info(
            "Photometry",
        )
        self._logger.info(
            "----------",
        )

        photometry_result = observation.photometry

        if photometry_result is None:

            self._logger.info(
                "Completed         : False",
            )

            self._logger.info(
                "Measured Sources  : 0",
            )

            return

        self._logger.info(
            "Completed         : True",
        )

        self._logger.info(
            "Algorithm         : %s",
            photometry_result.algorithm,
        )

        self._logger.info(
            "Measured Sources  : %d",
            len(
                photometry_result.measurements,
            ),
        )

        self._logger.info(
            "Aperture Radius   : %.2f pixels",
            photometry_result.aperture_radius,
        )

        self._logger.info(
            "Annulus Inner     : %.2f pixels",
            photometry_result.annulus_inner_radius,
        )

        self._logger.info(
            "Annulus Outer     : %.2f pixels",
            photometry_result.annulus_outer_radius,
        )

        self._logger.info(
            "Photometry Time   : %s",
            photometry_result.photometry_timestamp.isoformat(),
        )

    def _log_target_resolution(
        self,
        target_match: TargetSourceMatch | None,
    ) -> None:
        """
        Log an already-computed target-source association.

        No target resolution is performed by the inspector.
        """

        self._logger.info("")
        self._logger.info(
            "Target Resolution",
        )
        self._logger.info(
            "-----------------",
        )

        if target_match is None:

            self._logger.info(
                "Resolved          : False",
            )

            self._logger.info(
                "Target Match      : None",
            )

            return

        self._logger.info(
            "Resolved          : True",
        )

        self._logger.info(
            "Target            : %s",
            target_match.target_name,
        )

        self._logger.info(
            "Matched Source    : %s",
            target_match.source_identifier,
        )

        self._logger.info(
            "Expected Position : (%.3f, %.3f) pixels",
            target_match.target_x,
            target_match.target_y,
        )

        self._logger.info(
            "Detected Position : (%.3f, %.3f) pixels",
            target_match.source_x,
            target_match.source_y,
        )

        self._logger.info(
            "Separation        : %.3f pixels",
            target_match.separation_pixels,
        )

    def _log_evaluation(
        self,
        evaluation: PhotometryEvaluation | None,
    ) -> None:
        """
        Log already-computed photometric evaluation metrics.

        The inspector does not calculate or reinterpret these values.
        They are reported exactly as supplied by PhotometryEvaluator.
        """

        self._logger.info("")
        self._logger.info(
            "Evaluation",
        )
        self._logger.info(
            "----------",
        )

        if evaluation is None:

            self._logger.info(
                "Available         : False",
            )

            return

        self._logger.info(
            "Available         : True",
        )

        self._logger.info(
            "Algorithm         : %s",
            evaluation.algorithm,
        )

        self._logger.info(
            "Detected Sources  : %d",
            evaluation.detected_source_count,
        )

        self._logger.info(
            "Measured Sources  : %d",
            evaluation.measured_source_count,
        )

        self._logger.info(
            "Completeness      : %.2f%%",
            evaluation.measurement_completeness
            * 100.0,
        )

        self._logger.info(
            "Invalid Measure.  : %d",
            evaluation.invalid_measurement_count,
        )

        self._logger.info(
            "Non-positive Flux : %d",
            evaluation.non_positive_flux_count,
        )

        self._logger.info(
            "Crowded Sources   : %d",
            evaluation.crowded_source_count,
        )

        self._logger.info(
            "Crowding Fraction : %.2f%%",
            evaluation.crowding_fraction
            * 100.0,
        )

        self._logger.info(
            "Evaluation Time   : %s",
            evaluation.evaluation_timestamp.isoformat(),
        )

    def _log_processing_state(
        self,
        observation: Observation,
    ) -> None:
        """
        Log the current processing state.
        """

        state = observation.processing_state

        self._logger.info("")
        self._logger.info(
            "Processing State",
        )
        self._logger.info(
            "----------------",
        )

        self._logger.info(
            "Background Removed : %s",
            state.background_removed,
        )

        self._logger.info(
            "Source Detection   : %s",
            state.source_detection_completed,
        )

        self._logger.info(
            "SCARLET Applied    : %s",
            state.scarlet_applied,
        )

        self._logger.info(
            "Photometry Complete: %s",
            state.photometry_completed,
        )