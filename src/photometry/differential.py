"""
Differential photometry.

Provides the downstream calculation layer for classical differential
photometry. The implementation consumes already-computed source
photometry and target/reference source associations; it does not perform
source detection, aperture construction, image processing, or source
resolution.

For a target source T and a configured ensemble of reference sources R,
the reference flux is currently defined as the unweighted arithmetic mean
of the valid, positive reference net flux measurements::

    F_ref = mean(F_ref_1, ..., F_ref_N)

The differential flux and differential magnitude are then::

    F_diff = F_target / F_ref

    dm = -2.5 * log10(F_diff)

This intentionally provides a transparent baseline implementation.
Reference weighting, reference-star variability modelling and robust
ensemble estimators can be introduced later as separate analytical
methods without changing the underlying aperture-photometry measurement
layer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from core.exceptions import ObservationError
from domain.models import (
    PhotometryMeasurement,
    PhotometryResult,
    ReferenceSourceMatch,
    TargetSourceMatch,
)
from utils.logger import get_logger


@dataclass(frozen=True, slots=True)
class DifferentialPhotometryResult:
    """Result of a single differential-photometry calculation."""

    target_source_identifier: str
    reference_source_identifiers: tuple[str, ...]
    excluded_reference_source_identifiers: tuple[str, ...]
    target_net_flux: float
    reference_net_fluxes: tuple[float, ...]
    reference_flux: float
    differential_flux: float
    differential_magnitude: float

    @property
    def valid_reference_count(self) -> int:
        """Return the number of reference measurements used."""
        return len(self.reference_source_identifiers)


class DifferentialPhotometer:
    """
    Calculate classical differential photometry from source measurements.

    This class deliberately operates after aperture photometry and source
    resolution. It consumes the ``PhotometryResult`` produced by the
    photometry layer together with the target and reference source matches
    produced by the target-resolution layer.

    The current baseline combines valid reference net fluxes using an
    unweighted arithmetic mean.
    """

    _ALGORITHM = "Classical Differential Photometry"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialise the differential-photometry calculator."""
        self._logger = logger or get_logger(__name__)

    def calculate(
        self,
        photometry: PhotometryResult,
        target_match: TargetSourceMatch,
        reference_matches: Sequence[ReferenceSourceMatch],
    ) -> DifferentialPhotometryResult:
        """
        Calculate differential flux and differential magnitude.

        Invalid reference measurements are excluded from the reference
        ensemble. The target measurement must be valid and have a finite,
        positive net flux.
        """
        self._validate_inputs(
            photometry=photometry,
            target_match=target_match,
            reference_matches=reference_matches,
        )

        measurements = self._index_measurements(photometry)

        target_measurement = self._get_measurement(
            measurements,
            target_match.source_identifier,
            role="target",
        )
        assert target_measurement is not None
        target_net_flux = self._validate_measurement(
            target_measurement,
            role="target",
        )

        reference_source_identifiers: list[str] = []
        excluded_reference_source_identifiers: list[str] = []
        reference_net_fluxes: list[float] = []
        seen_reference_sources: set[str] = set()

        for reference_match in reference_matches:
            source_identifier = reference_match.source_identifier

            if source_identifier in seen_reference_sources:
                raise ObservationError(
                    "Differential photometry cannot use the same detected "
                    f"source '{source_identifier}' more than once as a "
                    "reference source."
                )

            seen_reference_sources.add(source_identifier)

            measurement = self._get_measurement(
                measurements,
                source_identifier,
                role=f"reference '{reference_match.reference_identifier}'",
                required=False,
            )

            if measurement is None:
                excluded_reference_source_identifiers.append(source_identifier)
                self._logger.warning(
                    "Reference source '%s' has no photometry measurement "
                    "and will be excluded.",
                    source_identifier,
                )
                continue

            reference_flux = self._reference_flux_or_none(measurement)
            if reference_flux is None:
                excluded_reference_source_identifiers.append(source_identifier)
                self._logger.warning(
                    "Reference source '%s' has an invalid or non-positive "
                    "net flux and will be excluded.",
                    source_identifier,
                )
                continue

            reference_source_identifiers.append(source_identifier)
            reference_net_fluxes.append(reference_flux)

        if not reference_net_fluxes:
            raise ObservationError(
                "Differential photometry requires at least one valid, "
                "positive reference net-flux measurement."
            )

        reference_flux = sum(reference_net_fluxes) / len(reference_net_fluxes)
        if not math.isfinite(reference_flux) or reference_flux <= 0.0:
            raise ObservationError(
                "Differential photometry produced an invalid combined "
                "reference flux."
            )

        differential_flux = target_net_flux / reference_flux
        if not math.isfinite(differential_flux) or differential_flux <= 0.0:
            raise ObservationError(
                "Differential photometry produced an invalid differential "
                "flux."
            )

        differential_magnitude = -2.5 * math.log10(differential_flux)
        if not math.isfinite(differential_magnitude):
            raise ObservationError(
                "Differential photometry produced an invalid differential "
                "magnitude."
            )

        result = DifferentialPhotometryResult(
            target_source_identifier=target_match.source_identifier,
            reference_source_identifiers=tuple(reference_source_identifiers),
            excluded_reference_source_identifiers=tuple(
                excluded_reference_source_identifiers
            ),
            target_net_flux=target_net_flux,
            reference_net_fluxes=tuple(reference_net_fluxes),
            reference_flux=float(reference_flux),
            differential_flux=float(differential_flux),
            differential_magnitude=float(differential_magnitude),
        )

        self._logger.info(
            "Differential photometry completed for target '%s'. "
            "Used %d/%d reference sources. "
            "Differential flux=%.8f, differential magnitude=%.8f.",
            target_match.target_name,
            result.valid_reference_count,
            len(reference_matches),
            result.differential_flux,
            result.differential_magnitude,
        )

        return result

    @staticmethod
    def _validate_inputs(
        *,
        photometry: PhotometryResult,
        target_match: TargetSourceMatch,
        reference_matches: Sequence[ReferenceSourceMatch],
    ) -> None:
        """Validate the structural inputs required for calculation."""
        if photometry is None:
            raise ObservationError(
                "Differential photometry requires a completed "
                "photometry result."
            )
        if target_match is None:
            raise ObservationError(
                "Differential photometry requires a resolved target source."
            )
        if not reference_matches:
            raise ObservationError(
                "Differential photometry requires at least one resolved "
                "reference source."
            )
        if not photometry.measurements:
            raise ObservationError(
                "Differential photometry cannot proceed because the "
                "photometry result contains no measurements."
            )

    @staticmethod
    def _index_measurements(
        photometry: PhotometryResult,
    ) -> dict[str, PhotometryMeasurement]:
        """Index measurements by detected-source identifier."""
        measurements: dict[str, PhotometryMeasurement] = {}
        for measurement in photometry.measurements:
            source_identifier = measurement.source_identifier
            if source_identifier in measurements:
                raise ObservationError(
                    "Differential photometry cannot unambiguously map "
                    f"source identifier '{source_identifier}' because "
                    "multiple photometry measurements use that identifier."
                )
            measurements[source_identifier] = measurement
        return measurements

    @staticmethod
    def _get_measurement(
        measurements: dict[str, PhotometryMeasurement],
        source_identifier: str,
        *,
        role: str,
        required: bool = True,
    ) -> PhotometryMeasurement | None:
        """Retrieve a source measurement by identifier."""
        measurement = measurements.get(source_identifier)
        if measurement is None and required:
            raise ObservationError(
                f"Differential photometry could not find the {role} "
                f"measurement for detected source '{source_identifier}'."
            )
        return measurement

    @staticmethod
    def _validate_measurement(
        measurement: PhotometryMeasurement,
        *,
        role: str,
    ) -> float:
        """Validate and return a positive finite source net flux."""
        if not measurement.valid:
            raise ObservationError(
                f"Differential photometry cannot use the {role} "
                f"measurement for source '{measurement.source_identifier}' "
                f"because it is marked invalid ({measurement.quality_flag})."
            )

        net_flux = measurement.net_flux
        if net_flux is None:
            raise ObservationError(
                f"Differential photometry cannot use the {role} "
                f"measurement for source '{measurement.source_identifier}' "
                "because its net flux is unavailable."
            )
        if not math.isfinite(net_flux) or net_flux <= 0.0:
            raise ObservationError(
                f"Differential photometry cannot use the {role} "
                f"measurement for source '{measurement.source_identifier}' "
                "because its net flux is not finite and positive."
            )
        return float(net_flux)

    @staticmethod
    def _reference_flux_or_none(
        measurement: PhotometryMeasurement,
    ) -> float | None:
        """Return a usable reference net flux or ``None``."""
        if not measurement.valid:
            return None
        net_flux = measurement.net_flux
        if net_flux is None:
            return None
        if not math.isfinite(net_flux) or net_flux <= 0.0:
            return None
        return float(net_flux)
