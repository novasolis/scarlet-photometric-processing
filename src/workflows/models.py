"""
Workflow result models.

Defines immutable result objects returned by the photometric workflow.

The workflow is target-centred: every completed experiment contains a
resolved target source and its selected local field. Differential
photometry may additionally contain resolved reference sources and a
differential-photometry result.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.models import (
    Observation,
    ReferenceSourceMatch,
    TargetSourceMatch,
)

from target.local_field import LocalField

from evaluation.models import PhotometryEvaluation
from photometry.differential import DifferentialPhotometryResult


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """
    Immutable result produced by a completed photometric workflow.

    Parameters
    ----------
    observation
        Observation containing the processed photometric measurements.

    evaluation
        Evaluation metrics produced from the workflow photometry.

    target_match
        Resolved association between the configured astronomical target
        and its detected source.

    local_field
        Target-centred population of detected sources selected for the
        experiment. The target is always the first source, followed by
        the selected neighbouring sources.

    reference_matches
        Resolved reference-source associations used for differential
        photometry. Empty for non-differential experiments.

    differential_photometry
        Differential-photometry result when the experiment type is
        ``DIFFERENTIAL_PHOTOMETRY``; otherwise ``None``.

    Notes
    -----
    All experiments are target-centred. Consequently, ``target_match``
    and ``local_field`` are required for every completed workflow.

    A local field is analytical experiment context derived from the
    observation's detected sources. It is intentionally kept separate
    from ``Observation`` because it represents a target-specific source
    selection rather than an intrinsic property of the observation.
    """

    observation: Observation

    evaluation: PhotometryEvaluation

    target_match: TargetSourceMatch

    local_field: LocalField

    reference_matches: tuple[ReferenceSourceMatch, ...] = ()

    differential_photometry: DifferentialPhotometryResult | None = None

@dataclass(frozen=True, slots=True)
class FieldAnalysisResult:
    """
    Result produced by a field-level analysis.

    FIELD_ANALYSIS performs the existing observation preparation,
    source detection and neighbour/crowding analysis without
    resolving a target or performing photometry.
    """

    observation: Observation

    neighbour_counts: dict[str, int]