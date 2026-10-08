"""
Experimental result export models.

Defines immutable, survey-independent data-transfer objects used to
persist photometric experiment results.

These models contain no scientific calculation logic and no file I/O.
They represent the flattened experimental results produced from the
photometric workflow and are intentionally separated from the runtime
Observation and evaluation domain models.

The export layer therefore provides a stable boundary between:

    WorkflowResult
        |
        v
    Export records
        |
        v
    CSV / JSON / analytical outputs

The models are independent of Photutils, Lightkurve, Astroquery,
SCARLET and other implementation libraries.

Source-selection scope is not stored as a separate export field. The
current workflow uses LocalField as the authoritative source-selection
boundary, and that runtime result is represented by the exported source
records themselves rather than by the removed legacy ``source_scope``
configuration field.

Time-series support
-------------------
``TimeSeriesMeasurementRecord`` represents one source measurement at
one observation/cadence within a time-series experiment.

``DifferentialTimeSeriesMeasurementRecord`` represents one completed
target/reference ensemble calculation at one observation/cadence within
a differential-photometry time-series experiment.

The two records are deliberately kept separate:

* ``TimeSeriesMeasurementRecord`` preserves source-level photometry.
* ``DifferentialTimeSeriesMeasurementRecord`` preserves the derived
  target/reference ensemble result.

This allows the source-level time-series export to remain stable while
providing a dedicated representation of the differential light curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """
    Experiment-level summary of a completed photometric workflow.

    One record represents one execution of a configured experiment.
    The record contains the observation provenance, target association,
    source-detection characteristics, photometric configuration and
    evaluation metrics required to reproduce and compare experimental
    outcomes.
    """

    experiment_id: str
    target_name: str
    survey: str
    experiment_type: str
    pipeline: str

    observation_time: datetime | float | None

    instrument: str | None
    filter_name: str | None
    exposure_time: float | None

    dataset_identifier: str
    detected_source_count: int
    detection_algorithm: str
    detection_threshold: float

    target_source_identifier: str | None
    target_x: float | None
    target_y: float | None
    source_x: float | None
    source_y: float | None
    target_source_separation_pixels: float | None

    photometry_algorithm: str
    aperture_radius: float
    annulus_inner_radius: float
    annulus_outer_radius: float

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

    evaluation_algorithm: str
    evaluation_timestamp: datetime

    observation_time_scale: str | None = None
    observation_time_format: str | None = None


@dataclass(frozen=True, slots=True)
class SourceMeasurementRecord:
    """
    Export record representing one attempted source measurement.

    One record is produced for each detected source represented in the
    photometric result, including measurements that failed quality
    control.

    Retaining invalid measurements is intentional. It allows downstream
    analysis to distinguish measurement failure from source non-detection
    and prevents quality-control information from being lost during
    export.
    """

    experiment_id: str
    target_name: str
    survey: str
    experiment_type: str
    pipeline: str

    source_identifier: str
    is_target_source: bool

    x_centroid: float
    y_centroid: float

    peak_value: float

    aperture_flux: float | None
    background_level: float | None
    background_flux: float | None
    net_flux: float | None
    instrumental_magnitude: float | None

    aperture_area: float | None
    background_area: float | None

    aperture_coverage: float | None
    annulus_coverage: float | None
    background_fraction: float | None

    valid: bool
    quality_flag: str


@dataclass(frozen=True, slots=True)
class TimeSeriesMeasurementRecord:
    """
    Export record representing one source measurement at one cadence.

    A separate record is produced for each source measured at each
    cadence in a time-series experiment.

    The logical identity of a record is:

        experiment_id + cadence_index + source_identifier

    This model preserves source-level aperture-photometry measurements.
    It does not contain derived ensemble differential-photometry values.
    Those values are represented by ``DifferentialTimeSeriesMeasurementRecord``.
    """

    experiment_id: str
    target_name: str
    survey: str
    experiment_type: str
    pipeline: str

    cadence_index: int

    observation_time: datetime | float | None
    observation_time_scale: str | None
    observation_time_format: str | None

    source_identifier: str
    is_target_source: bool

    x_centroid: float
    y_centroid: float

    target_x: float | None
    target_y: float | None
    target_source_separation_pixels: float | None

    peak_value: float

    aperture_flux: float | None
    background_level: float | None
    background_flux: float | None
    net_flux: float | None
    instrumental_magnitude: float | None

    aperture_area: float | None
    background_area: float | None

    aperture_coverage: float | None
    annulus_coverage: float | None
    background_fraction: float | None

    crowded: bool

    valid: bool
    quality_flag: str


@dataclass(frozen=True, slots=True)
class DifferentialTimeSeriesMeasurementRecord:
    """
    Export record representing one differential-photometry result at one
    time-series cadence.

    One record represents one target/reference ensemble calculation.
    """

    experiment_id: str
    target_name: str
    survey: str
    experiment_type: str
    pipeline: str

    cadence_index: int

    observation_time: datetime | float | None
    observation_time_scale: str | None
    observation_time_format: str | None

    target_source_identifier: str

    reference_source_identifiers: tuple[str, ...]
    excluded_reference_source_identifiers: tuple[str, ...]

    target_net_flux: float | None
    reference_net_fluxes: tuple[float, ...]
    reference_flux: float | None

    differential_flux: float | None
    differential_magnitude: float | None

    @property
    def valid_reference_count(self) -> int:
        """Return the number of reference sources used."""
        return len(self.reference_source_identifiers)