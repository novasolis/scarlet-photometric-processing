"""
Abstract base class for astronomical data acquisition.

Concrete implementations are responsible for acquiring astronomical
datasets from specific sources (e.g. TESS or SDSS) while exposing a
consistent interface to the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")
"""
Generic type representing the dataset returned by an acquisition
implementation.
"""


class BaseAcquirer(ABC, Generic[T]):
    """
    Abstract interface for astronomical data acquisition.

    The concrete dataset type is defined by subclasses through the
    generic type parameter.
    """

    @abstractmethod
    def acquire(self) -> T:
        """
        Acquire astronomical data from the underlying source.

        Returns
        -------
        T
            The acquired dataset.
        """
        raise NotImplementedError