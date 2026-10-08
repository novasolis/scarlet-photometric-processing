"""
Base interfaces for image preprocessing.

All preprocessing stages operate on the survey-independent Observation
domain model and return the updated Observation after processing.

Implementations may create additional image products, such as background
models or background-subtracted images, while preserving the original
observation for reproducibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import logging
from collections.abc import Sequence

from domain.models import Observation
from utils.logger import get_logger


class Preprocessor(ABC):
    """
    Abstract base class for all preprocessing stages.

    Each preprocessing component accepts an Observation, performs a
    defined scientific processing step and returns the updated
    Observation.

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
        Initialise the preprocessor.

        Parameters
        ----------
        logger
            Optional logger used for diagnostic and processing messages.
            If omitted, the module-specific project logger is used.
        """

        self._logger = logger or get_logger(__name__)

    @abstractmethod
    def process(
        self,
        observation: Observation,
    ) -> Observation:
        """
        Process an observation.

        Parameters
        ----------
        observation
            Observation to process.

        Returns
        -------
        Observation
            Updated observation containing the derived preprocessing
            products.

        Raises
        ------
        ValueError
            If the observation is unsuitable for preprocessing.

        Notes
        -----
        Concrete implementations are responsible for validating the
        supplied Observation and for updating the appropriate
        processing-state flags.
        """

        raise NotImplementedError


class SequentialPreprocessor(Preprocessor):
    """
    Apply multiple preprocessing stages sequentially.

    Each stage receives the Observation returned by the preceding stage.
    This allows preprocessing operations to be composed without requiring
    individual preprocessing implementations to know about one another.

    For example, the SCARLET pipeline can be composed as:

        BackgroundSubtractor
            |
            v
        ScarletPreprocessor

    while the baseline pipeline can continue to use:

        BackgroundSubtractor

    Parameters
    ----------
    preprocessors
        Ordered sequence of preprocessing stages to execute.
    logger
        Optional logger. If omitted, the project logger is used.
    """

    def __init__(
        self,
        preprocessors: Sequence[Preprocessor],
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(logger)

        if not preprocessors:
            raise ValueError(
                "SequentialPreprocessor requires at least one "
                "preprocessing stage.",
            )

        self._preprocessors = tuple(preprocessors)

    def process(
        self,
        observation: Observation,
    ) -> Observation:
        """
        Apply each preprocessing stage in order.

        Parameters
        ----------
        observation
            Observation to process.

        Returns
        -------
        Observation
            Observation returned by the final preprocessing stage.

        Raises
        ------
        ValueError
            If the supplied Observation is invalid or a preprocessing
            stage rejects it.
        """

        current = observation

        for preprocessor in self._preprocessors:
            current = preprocessor.process(current)

        return current