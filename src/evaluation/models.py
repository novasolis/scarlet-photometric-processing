"""
Evaluation models.

Defines survey-independent models used to represent source-level
photometric quality and observation-level evaluation summaries.

These models contain no implementation logic and remain independent
of Photutils, Lightkurve, Astroquery and other external libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SourceEvaluation:
    """
    Evaluation metrics for a single detected source.

    The source evaluation preserves the measurements and contextual
    characteristics required to assess photometric quality at the
    individual-source level.

    Parameters
    ----------
    source_identifier
        Identifier of the corresponding detected source.

    valid_measurement
        Whether the photometric measurement is considered valid for
        scientific evaluation.

    quality_flag
        Quality classification assigned to the photometric measurement.
        A missing measurement is represented by the evaluator using
        the ``MISSING_MEASUREMENT`` classification.

    net_flux
        Background-subtracted source flux. ``None`` when unavailable
        or invalid.

    instrumental_magnitude
        Instrumental magnitude derived from the net flux. ``None``
        when the net flux cannot produce a valid logarithmic
        magnitude.

    background_level
        Estimated local background level per pixel. ``None`` when
        unavailable.

    background_fraction
        Ratio of estimated background contribution within the
        aperture to the total measured aperture flux. ``None`` when
        it cannot be calculated reliably.

    aperture_coverage
        Fraction of the theoretical source-aperture area covered by
        valid image pixels. ``None`` when unavailable.

    annulus_coverage
        Fraction of the theoretical background-annulus area covered
        by valid image pixels. ``None`` when unavailable.

    neighbour_count
        Number of other detected sources within the configured
        crowding radius.

    crowded
        Whether the source has at least one neighbouring detected
        source within the configured crowding radius.

    Notes
    -----
    This model records evaluation results but performs no scientific
    calculations. It is intentionally independent of Photutils and
    other implementation libraries.
    """

    source_identifier: str

    valid_measurement: bool

    quality_flag: str

    net_flux: float | None

    instrumental_magnitude: float | None

    background_level: float | None

    background_fraction: float | None

    aperture_coverage: float | None

    annulus_coverage: float | None

    neighbour_count: int

    crowded: bool


@dataclass(frozen=True, slots=True)
class PhotometryEvaluation:
    """
    Summary evaluation of a photometric result.

    The evaluation describes the quality and basic characteristics
    of the measurements produced for a single Observation while
    preserving the corresponding source-level evaluation records.

    Parameters
    ----------
    detected_source_count
        Number of sources identified by the source-detection stage.

    measured_source_count
        Number of detected sources for which a photometric measurement
        record was produced.

    valid_measurement_count
        Number of photometric measurements classified as valid for
        scientific evaluation.

    measurement_completeness
        Fraction of detected sources that received a photometric
        measurement record. Expressed as a value between 0.0 and 1.0.

    valid_measurement_fraction
        Fraction of produced photometric measurements classified as
        valid. Expressed as a value between 0.0 and 1.0.

    invalid_measurement_count
        Number of produced photometric measurements classified as
        invalid.

    missing_measurement_count
        Number of detected sources for which no photometric
        measurement record was produced.

    non_positive_flux_count
        Number of photometric measurements having a non-positive
        background-subtracted net flux.

    crowded_source_count
        Number of detected sources having at least one neighbouring
        detected source within the configured crowding radius.

    crowding_fraction
        Fraction of detected sources classified as crowded.
        Expressed as a value between 0.0 and 1.0.

    crowding_radius
        Maximum centroid separation in pixels used to classify
        sources as crowded.

    median_net_flux
        Median background-subtracted source flux across valid
        photometric measurements.

    median_background_level
        Median estimated local background level across valid
        photometric measurements.

    median_background_fraction
        Median ratio of estimated background contribution within
        the source aperture to the total measured aperture flux
        across valid photometric measurements.

    algorithm
        Name of the photometry algorithm being evaluated.

    evaluation_timestamp
        UTC timestamp indicating when the evaluation was performed.

    source_evaluations
        Immutable collection containing the evaluation record for
        each detected source.

    Notes
    -----
    The model contains only evaluation results and provenance. It does
    not perform calculations and remains independent of implementation
    libraries such as Photutils.
    """

    detected_source_count: int

    measured_source_count: int

    valid_measurement_count: int

    measurement_completeness: float

    valid_measurement_fraction: float

    invalid_measurement_count: int

    missing_measurement_count: int

    non_positive_flux_count: int

    crowded_source_count: int

    crowding_fraction: float

    crowding_radius: float

    median_net_flux: float | None

    median_background_level: float | None

    median_background_fraction: float | None

    algorithm: str

    evaluation_timestamp: datetime

    source_evaluations: tuple[SourceEvaluation, ...]