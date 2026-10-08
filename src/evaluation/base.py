"""
Evaluation interface.

Defines the survey-independent interface for evaluating photometric
results within the photometric analysis pipeline.

Concrete evaluators operate exclusively on the project's Observation
domain model and return immutable evaluation results.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from domain.models import Observation
from evaluation.models import PhotometryEvaluation
from utils.logger import get_logger


class Evaluator(ABC):
    """
    Abstract interface for photometric evaluation.

    Concrete evaluators evaluate the scientific results contained
    within an Observation without modifying the Observation itself.

    Parameters
    ----------
    logger
        Optional logger used by the evaluator. If omitted, the
        module-specific project logger is created automatically.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the evaluator.

        Parameters
        ----------
        logger
            Optional logger used for diagnostic messages.
            If omitted, the module-specific project logger is used.
        """

        self._logger = logger or get_logger(__name__)

    @abstractmethod
    def evaluate(
        self,
        observation: Observation,
    ) -> PhotometryEvaluation:
        """
        Evaluate the photometric result contained in an Observation.

        Concrete implementations must evaluate the existing
        observation state without modifying the Observation.

        Parameters
        ----------
        observation
            Observation containing source detections and photometric
            measurements.

        Returns
        -------
        PhotometryEvaluation
            Immutable summary of photometric quality.

        Raises
        ------
        ValueError
            If the Observation does not contain the prerequisites
            required for evaluation.
        """

        raise NotImplementedError