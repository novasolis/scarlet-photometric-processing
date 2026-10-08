"""
Tests for experimental result export models.

These tests verify the structural contract of the immutable export
data-transfer objects.

The export models intentionally contain no scientific calculation
logic and no file I/O. Consequently, these tests focus on:

- construction of complete records;
- preservation of supplied values;
- support for optional values;
- representation of valid and invalid measurements;
- immutability;
- slot-based dataclass behaviour.

The tests do not duplicate scientific calculations already covered by
the evaluation and photometry test suites.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import Any

import pytest

from export.models import (
    ExperimentSummary,
    SourceMeasurementRecord,
)


def _create_experiment_summary(
    **overrides: Any,
) -> ExperimentSummary:
    """
    Create a representative complete experiment summary.

    Individual tests may override fields relevant to the behaviour
    under test.
    """

    values: dict[str, Any] = {
        "experiment_id": "tess-lhs-3844-baseline",
        "target_name": "LHS 3844",
        "survey": "TESS",
        "experiment_type": "CROWDED_FIELD",
        "source_scope": "TARGET",
        "pipeline": "BASELINE",
        "observation_time": datetime(
            2018,
            7,
            25,
            12,
            30,
            tzinfo=timezone.utc,
        ),
        "instrument": "TESS",
        "filter_name": "TESS",
        "exposure_time": 120.0,
        "dataset_identifier": "tess-sector-01-lhs-3844",
        "detected_source_count": 3,
        "detection_algorithm": "DAOStarFinder",
        "detection_threshold": 12.5,
        "target_source_identifier": "2",
        "target_x": 5.25,
        "target_y": 4.75,
        "source_x": 5.31,
        "source_y": 4.81,
        "target_source_separation_pixels": 0.085,
        "photometry_algorithm": "AperturePhotometry",
        "aperture_radius": 2.0,
        "annulus_inner_radius": 3.0,
        "annulus_outer_radius": 4.0,
        "measured_source_count": 3,
        "valid_measurement_count": 2,
        "measurement_completeness": 1.0,
        "valid_measurement_fraction": 2.0 / 3.0,
        "invalid_measurement_count": 1,
        "missing_measurement_count": 0,
        "non_positive_flux_count": 1,
        "crowded_source_count": 2,
        "crowding_fraction": 2.0 / 3.0,
        "crowding_radius": 5.0,
        "median_net_flux": 1250.5,
        "median_background_level": 18.75,
        "median_background_fraction": 0.042,
        "evaluation_algorithm": "PhotometryEvaluator",
        "evaluation_timestamp": datetime(
            2026,
            8,
            18,
            9,
            30,
            tzinfo=timezone.utc,
        ),
    }

    values.update(overrides)

    return ExperimentSummary(**values)


def _create_source_measurement_record(
    **overrides: Any,
) -> SourceMeasurementRecord:
    """
    Create a representative valid source measurement record.

    Individual tests may override fields relevant to the behaviour
    under test.
    """

    values: dict[str, Any] = {
        "experiment_id": "tess-lhs-3844-baseline",
        "target_name": "LHS 3844",
        "survey": "TESS",
        "experiment_type": "CROWDED_FIELD",
        "source_scope": "TARGET",
        "pipeline": "BASELINE",
        "source_identifier": "2",
        "is_target_source": True,
        "x_centroid": 5.31,
        "y_centroid": 4.81,
        "peak_value": 87.5,
        "aperture_flux": 1520.25,
        "background_level": 18.75,
        "background_flux": 150.0,
        "net_flux": 1370.25,
        "instrumental_magnitude": -7.841,
        "aperture_area": 12.57,
        "background_area": 37.70,
        "aperture_coverage": 1.0,
        "annulus_coverage": 1.0,
        "background_fraction": 0.0987,
        "valid": True,
        "quality_flag": "VALID",
    }

    values.update(overrides)

    return SourceMeasurementRecord(**values)


def test_experiment_summary_can_be_created_with_complete_values() -> None:
    """
    A complete experiment summary can be constructed successfully.
    """

    summary = _create_experiment_summary()

    assert summary.experiment_id == "tess-lhs-3844-baseline"
    assert summary.target_name == "LHS 3844"
    assert summary.survey == "TESS"
    assert summary.pipeline == "BASELINE"

    assert summary.detected_source_count == 3
    assert summary.measured_source_count == 3
    assert summary.valid_measurement_count == 2

    assert summary.target_source_identifier == "2"
    assert summary.target_source_separation_pixels == pytest.approx(
        0.085,
    )

    assert summary.median_net_flux == pytest.approx(
        1250.5,
    )


def test_experiment_summary_preserves_optional_none_values() -> None:
    """
    Optional observation and target-association values may be absent.
    """

    summary = _create_experiment_summary(
        observation_time=None,
        instrument=None,
        filter_name=None,
        exposure_time=None,
        target_source_identifier=None,
        target_x=None,
        target_y=None,
        source_x=None,
        source_y=None,
        target_source_separation_pixels=None,
        median_net_flux=None,
        median_background_level=None,
        median_background_fraction=None,
    )

    assert summary.observation_time is None
    assert summary.instrument is None
    assert summary.filter_name is None
    assert summary.exposure_time is None

    assert summary.target_source_identifier is None
    assert summary.target_x is None
    assert summary.target_y is None
    assert summary.source_x is None
    assert summary.source_y is None
    assert summary.target_source_separation_pixels is None

    assert summary.median_net_flux is None
    assert summary.median_background_level is None
    assert summary.median_background_fraction is None


def test_experiment_summary_can_represent_field_experiment() -> None:
    """
    A field-level experiment can be represented without a target match.
    """

    summary = _create_experiment_summary(
        source_scope="FIELD",
        target_source_identifier=None,
        target_x=None,
        target_y=None,
        source_x=None,
        source_y=None,
        target_source_separation_pixels=None,
    )

    assert summary.source_scope == "FIELD"
    assert summary.target_source_identifier is None
    assert summary.target_x is None
    assert summary.target_y is None
    assert summary.source_x is None
    assert summary.source_y is None
    assert summary.target_source_separation_pixels is None


def test_experiment_summary_preserves_pipeline_identity() -> None:
    """
    The export model preserves the processing pipeline identifier.

    This allows baseline and future SCARLET results to be compared
    using the same export schema.
    """

    baseline = _create_experiment_summary(
        pipeline="BASELINE",
    )

    scarlet = _create_experiment_summary(
        pipeline="SCARLET",
    )

    assert baseline.pipeline == "BASELINE"
    assert scarlet.pipeline == "SCARLET"


def test_experiment_summary_is_immutable() -> None:
    """
    Experiment summaries cannot be modified after construction.
    """

    summary = _create_experiment_summary()

    with pytest.raises(FrozenInstanceError):
        summary.target_name = "Changed Target"  # type: ignore[misc]


def test_experiment_summary_uses_slots() -> None:
    """
    ExperimentSummary uses slots and therefore does not expose a
    per-instance __dict__.
    """

    summary = _create_experiment_summary()

    assert not hasattr(summary, "__dict__")


def test_source_measurement_record_can_represent_valid_measurement() -> None:
    """
    A valid source measurement can be represented with its complete
    photometric information.
    """

    record = _create_source_measurement_record()

    assert record.source_identifier == "2"
    assert record.is_target_source is True

    assert record.x_centroid == pytest.approx(5.31)
    assert record.y_centroid == pytest.approx(4.81)

    assert record.aperture_flux == pytest.approx(1520.25)
    assert record.background_flux == pytest.approx(150.0)
    assert record.net_flux == pytest.approx(1370.25)

    assert record.instrumental_magnitude == pytest.approx(
        -7.841,
    )

    assert record.aperture_coverage == pytest.approx(1.0)
    assert record.annulus_coverage == pytest.approx(1.0)

    assert record.valid is True
    assert record.quality_flag == "VALID"


def test_source_measurement_record_can_represent_invalid_measurement() -> None:
    """
    Invalid measurements remain representable in the export model.

    Exporting invalid measurements is important because evaluation
    statistics must distinguish invalid measurements from missing
    measurements.
    """

    record = _create_source_measurement_record(
        valid=False,
        quality_flag="NON_POSITIVE_FLUX",
        net_flux=None,
        instrumental_magnitude=None,
    )

    assert record.valid is False
    assert record.quality_flag == "NON_POSITIVE_FLUX"
    assert record.net_flux is None
    assert record.instrumental_magnitude is None


def test_source_measurement_record_supports_missing_optional_values() -> None:
    """
    Optional measurement values may legitimately be unavailable.
    """

    record = _create_source_measurement_record(
        aperture_flux=None,
        background_level=None,
        background_flux=None,
        net_flux=None,
        instrumental_magnitude=None,
        aperture_area=None,
        background_area=None,
        aperture_coverage=None,
        annulus_coverage=None,
        background_fraction=None,
    )

    assert record.aperture_flux is None
    assert record.background_level is None
    assert record.background_flux is None
    assert record.net_flux is None
    assert record.instrumental_magnitude is None
    assert record.aperture_area is None
    assert record.background_area is None
    assert record.aperture_coverage is None
    assert record.annulus_coverage is None
    assert record.background_fraction is None


def test_source_measurement_record_preserves_target_source_flag() -> None:
    """
    The export record distinguishes the resolved target source from
    other detected field sources.
    """

    target_source = _create_source_measurement_record(
        source_identifier="1",
        is_target_source=True,
    )

    neighbouring_source = _create_source_measurement_record(
        source_identifier="2",
        is_target_source=False,
    )

    assert target_source.is_target_source is True
    assert neighbouring_source.is_target_source is False


def test_source_measurement_record_preserves_quality_classification() -> None:
    """
    The export model preserves the evaluator's quality classification
    without recalculating or interpreting it.
    """

    record = _create_source_measurement_record(
        valid=False,
        quality_flag="INSUFFICIENT_APERTURE_COVERAGE",
    )

    assert record.valid is False
    assert record.quality_flag == (
        "INSUFFICIENT_APERTURE_COVERAGE"
    )


def test_source_measurement_record_is_immutable() -> None:
    """
    Source measurement records cannot be modified after construction.
    """

    record = _create_source_measurement_record()

    with pytest.raises(FrozenInstanceError):
        record.net_flux = 9999.0  # type: ignore[misc]


def test_source_measurement_record_uses_slots() -> None:
    """
    SourceMeasurementRecord uses slots and therefore does not expose
    a per-instance __dict__.
    """

    record = _create_source_measurement_record()

    assert not hasattr(record, "__dict__")


def test_export_models_do_not_perform_scientific_calculations() -> None:
    """
    Export models preserve supplied scientific values exactly.

    Scientific calculations belong to photometry and evaluation
    components, not the export layer.
    """

    summary = _create_experiment_summary(
        measurement_completeness=0.123,
        valid_measurement_fraction=0.456,
        crowding_fraction=0.789,
        median_net_flux=9876.54321,
    )

    record = _create_source_measurement_record(
        net_flux=1234.56789,
        background_fraction=0.321,
    )

    assert summary.measurement_completeness == 0.123
    assert summary.valid_measurement_fraction == 0.456
    assert summary.crowding_fraction == 0.789
    assert summary.median_net_flux == 9876.54321

    assert record.net_flux == 1234.56789
    assert record.background_fraction == 0.321


def test_export_models_are_value_equal_when_values_are_equal() -> None:
    """
    Two records containing identical values compare equal as expected
    for immutable dataclass-based value objects.
    """

    summary_a = _create_experiment_summary()
    summary_b = _create_experiment_summary()

    record_a = _create_source_measurement_record()
    record_b = _create_source_measurement_record()

    assert summary_a == summary_b
    assert record_a == record_b