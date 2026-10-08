"""
Observation builders.

Builders convert external astronomy library objects into the project's
internal Observation domain model. They isolate acquisition-specific
libraries, such as Lightkurve and Astropy FITS, from the remainder of
the photometric analysis pipeline.

Concrete builders are responsible for translating external datasets
into the survey-independent Observation domain model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Generic, TypeVar

from domain.models import Observation
from utils.logger import get_logger


T = TypeVar("T")


class ObservationBuilder(
    ABC,
    Generic[T],
):
    """
    Abstract base class for Observation builders.

    An ObservationBuilder converts an externally acquired astronomical
    dataset into the project's survey-independent Observation model.

    The builder is the boundary between acquisition-specific external
    libraries and the remainder of the processing pipeline.

    Parameters
    ----------
    logger
        Optional logger used for diagnostic and construction messages.
        If omitted, the project logger is used.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the Observation builder.

        Parameters
        ----------
        logger
            Optional logger.
        """

        self._logger = logger or get_logger(
            __name__,
        )

    @abstractmethod
    def build(
        self,
        dataset: T,
    ) -> Observation:
        """
        Convert an external dataset into an Observation.

        Parameters
        ----------
        dataset
            External astronomy-library dataset returned by the
            acquisition layer.

        Returns
        -------
        Observation
            Survey-independent Observation domain model.

        Raises
        ------
        ValueError
            If the supplied dataset cannot be converted into a valid
            Observation.

        Notes
        -----
        Concrete implementations are responsible for validating the
        external dataset and preserving the metadata required by
        downstream processing stages.

        The builder must not perform scientific processing such as
        background subtraction, source detection or photometry.
        """

        raise NotImplementedError