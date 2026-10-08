"""
Abstract interfaces for source detection.

This module defines the survey-independent interface implemented by all
source detection algorithms used within the photometric analysis pipeline.

A source detector identifies astronomical sources within the active image
of an Observation and enriches the Observation with a DetectionResult while
preserving all existing image products and metadata.

Concrete implementations may use algorithms such as DAOStarFinder,
segmentation-based methods, or future AI-assisted detectors without changing
the pipeline architecture.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from domain.models import Observation
from utils.logger import get_logger


class SourceDetector(ABC):
    """
    Abstract base class for source detection algorithms.

    A SourceDetector operates on the active image contained within an
    Observation, identifies candidate astronomical sources, and stores
    the resulting DetectionResult within the same Observation.

    Implementations remain survey-independent and operate exclusively
    on the common Observation domain model.

    Parameters
    ----------
    logger
        Optional logger. If omitted, the project logger is used.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the detector.

        Parameters
        ----------
        logger
            Optional logger used for diagnostic and processing messages.
            If omitted, the module-specific project logger is used.
        """

        self._logger = logger or get_logger(__name__)

    @abstractmethod
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
            Updated observation containing the detection results.

        Raises
        ------
        ValueError
            If the observation is unsuitable for source detection.

        Notes
        -----
        Implementations are responsible for validating the supplied
        Observation before performing source detection.

        The returned Observation must contain a DetectionResult whose
        source collection represents the complete output of the
        detection operation.
        """

        raise NotImplementedError