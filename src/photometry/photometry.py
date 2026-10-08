"""
Photometry interface.

Defines the survey-independent interface for photometric measurement
within the photometric analysis pipeline.

Concrete implementations are responsible for performing a specific
photometric technique, such as classical aperture photometry or
SCARLET-based photometry.

The interface operates exclusively on the project's Observation domain
model and therefore remains independent of survey-specific acquisition
libraries.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from collections.abc import Mapping
import numpy as np

from domain.models import Observation
from utils.logger import get_logger


class Photometer(
    ABC,
):
    """
    Abstract interface for photometric measurement.

    A Photometer consumes an Observation containing a processed image
    and detected sources, performs photometric measurements and returns
    the updated Observation containing the resulting PhotometryResult.

    Concrete implementations must remain independent of survey-specific
    acquisition mechanisms. They operate only on the common Observation
    domain model.

    Parameters
    ----------
    logger
        Optional logger used by the photometry implementation. If omitted,
        a module-specific logger is created automatically.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the photometer.

        Parameters
        ----------
        logger
            Optional logger used for diagnostic and processing messages.
            If omitted, the module-specific project logger is used.
        """

        self._logger = logger or get_logger(
            __name__,
        )

    @abstractmethod
    def measure(
        self,
        observation: Observation,
    ) -> Observation:
        """
        Perform photometric measurements on an observation.

        Parameters
        ----------
        observation
            Observation containing the processed image and detected
            sources to be measured.

        Returns
        -------
        Observation
            Updated Observation containing the resulting
            PhotometryResult.

        Raises
        ------
        ValueError
            If the observation does not contain the prerequisites
            required for photometric measurement.
        """

        raise NotImplementedError

    def measure_source_images(
        self,
        observation: Observation,
        source_images: Mapping[str, np.ndarray],
    ) -> Observation:
        """
        Measure sources using a source-specific image for each source.

        This optional interface is used by SCARLET-enhanced workflows.
        ``source_images`` maps detected-source identifiers to two-dimensional
        images constructed for that source. A concrete photometer may use the
        same aperture/background geometry as the baseline measurement while
        replacing the blended observation with the supplied source-specific
        deblended observation.

        The default implementation deliberately fails rather than silently
        falling back to measurement of the blended active image. That keeps
        the baseline and SCARLET measurement populations scientifically
        distinct.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement source-specific "
            "photometric measurement."
        )

