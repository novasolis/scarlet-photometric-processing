"""Unit tests for the differential-photometry calculation layer."""

from datetime import datetime, timezone
from math import log10

import pytest

from core.exceptions import ObservationError
from domain.models import (
    PhotometryMeasurement,
    PhotometryResult,
    ReferenceSourceMatch,
    TargetSourceMatch,
)
from photometry.differential import (
    DifferentialPhotometer,
    DifferentialPhotometryResult,
)


def measurement(identifier: str, flux: float | None, *, valid: bool = True, flag: str = "VALID") -> PhotometryMeasurement:
    return PhotometryMeasurement(
        source_identifier=identifier,
        x_centroid=10.0,
        y_centroid=10.0,
        aperture_flux=flux,
        background_level=0.0,
        background_flux=0.0,
        net_flux=flux,
        instrumental_magnitude=None,
        aperture_area=1.0,
        background_area=1.0,
        aperture_coverage=1.0,
        annulus_coverage=1.0,
        background_fraction=0.0,
        valid=valid,
        quality_flag=flag,
    )


def photometry(*items: PhotometryMeasurement) -> PhotometryResult:
    return PhotometryResult(
        measurements=tuple(items),
        algorithm="Test Aperture Photometry",
        aperture_radius=2.0,
        annulus_inner_radius=3.0,
        annulus_outer_radius=4.0,
        photometry_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def target(source: str = "target") -> TargetSourceMatch:
    return TargetSourceMatch(
        target_name="Test Target",
        source_identifier=source,
        target_x=10.0,
        target_y=10.0,
        source_x=10.0,
        source_y=10.0,
        separation_pixels=0.0,
    )


def reference(name: str, source: str) -> ReferenceSourceMatch:
    return ReferenceSourceMatch(
        reference_identifier=name,
        source_identifier=source,
        reference_x=10.0,
        reference_y=10.0,
        source_x=10.0,
        source_y=10.0,
        separation_pixels=0.0,
    )


@pytest.fixture
def calculator() -> DifferentialPhotometer:
    return DifferentialPhotometer()


def test_single_reference(calculator: DifferentialPhotometer) -> None:
    result = calculator.calculate(
        photometry(measurement("target", 1000.0), measurement("ref-1", 500.0)),
        target(),
        (reference("Reference 1", "ref-1"),),
    )
    assert isinstance(result, DifferentialPhotometryResult)
    assert result.target_source_identifier == "target"
    assert result.reference_source_identifiers == ("ref-1",)
    assert result.reference_net_fluxes == (500.0,)
    assert result.reference_flux == pytest.approx(500.0)
    assert result.differential_flux == pytest.approx(2.0)
    assert result.differential_magnitude == pytest.approx(-2.5 * log10(2.0))
    assert result.valid_reference_count == 1


def test_multiple_references_use_arithmetic_mean(calculator: DifferentialPhotometer) -> None:
    result = calculator.calculate(
        photometry(
            measurement("target", 1200.0),
            measurement("ref-1", 400.0),
            measurement("ref-2", 600.0),
            measurement("ref-3", 800.0),
        ),
        target(),
        (reference("Reference 1", "ref-1"), reference("Reference 2", "ref-2"), reference("Reference 3", "ref-3")),
    )
    assert result.reference_net_fluxes == (400.0, 600.0, 800.0)
    assert result.reference_flux == pytest.approx(600.0)
    assert result.differential_flux == pytest.approx(2.0)


def test_reference_order_is_preserved(calculator: DifferentialPhotometer) -> None:
    result = calculator.calculate(
        photometry(measurement("target", 1000.0), measurement("ref-a", 300.0), measurement("ref-b", 500.0)),
        target(),
        (reference("Reference B", "ref-b"), reference("Reference A", "ref-a")),
    )
    assert result.reference_source_identifiers == ("ref-b", "ref-a")
    assert result.reference_net_fluxes == (500.0, 300.0)
    assert result.reference_flux == pytest.approx(400.0)


def test_invalid_reference_is_excluded(calculator: DifferentialPhotometer) -> None:
    result = calculator.calculate(
        photometry(
            measurement("target", 1000.0),
            measurement("ref-1", 500.0),
            measurement("ref-2", 700.0, valid=False, flag="NON_POSITIVE_FLUX"),
            measurement("ref-3", 800.0),
        ),
        target(),
        (reference("Reference 1", "ref-1"), reference("Reference 2", "ref-2"), reference("Reference 3", "ref-3")),
    )
    assert result.reference_source_identifiers == ("ref-1", "ref-3")
    assert result.excluded_reference_source_identifiers == ("ref-2",)
    assert result.reference_flux == pytest.approx(650.0)


@pytest.mark.parametrize("flux", [None, 0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_unusable_reference_flux_is_excluded(calculator: DifferentialPhotometer, flux: float | None) -> None:
    result = calculator.calculate(
        photometry(measurement("target", 1000.0), measurement("bad", flux, flag="INVALID_FLUX"), measurement("good", 500.0)),
        target(),
        (reference("Bad", "bad"), reference("Good", "good")),
    )
    assert result.reference_source_identifiers == ("good",)
    assert result.excluded_reference_source_identifiers == ("bad",)


def test_missing_reference_measurement_is_excluded(calculator: DifferentialPhotometer) -> None:
    result = calculator.calculate(
        photometry(measurement("target", 1000.0), measurement("good", 500.0)),
        target(),
        (reference("Missing", "missing"), reference("Good", "good")),
    )
    assert result.reference_source_identifiers == ("good",)
    assert result.excluded_reference_source_identifiers == ("missing",)


def test_all_invalid_references_are_rejected(calculator: DifferentialPhotometer) -> None:
    with pytest.raises(ObservationError, match="at least one valid"):
        calculator.calculate(
            photometry(measurement("target", 1000.0), measurement("ref-1", 0.0, valid=False), measurement("ref-2", None, valid=False)),
            target(),
            (reference("Reference 1", "ref-1"), reference("Reference 2", "ref-2")),
        )


def test_invalid_target_is_rejected(calculator: DifferentialPhotometer) -> None:
    with pytest.raises(ObservationError, match="target.*marked invalid"):
        calculator.calculate(
            photometry(measurement("target", 1000.0, valid=False, flag="INVALID"), measurement("ref", 500.0)),
            target(),
            (reference("Reference", "ref"),),
        )


@pytest.mark.parametrize("flux", [None, 0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_unusable_target_flux_is_rejected(calculator: DifferentialPhotometer, flux: float | None) -> None:
    with pytest.raises(ObservationError):
        calculator.calculate(
            photometry(measurement("target", flux), measurement("ref", 500.0)),
            target(),
            (reference("Reference", "ref"),),
        )


def test_no_reference_matches_are_rejected(calculator: DifferentialPhotometer) -> None:
    with pytest.raises(ObservationError, match="at least one resolved"):
        calculator.calculate(photometry(measurement("target", 1000.0)), target(), ())


def test_missing_target_measurement_is_rejected(calculator: DifferentialPhotometer) -> None:
    with pytest.raises(ObservationError, match="could not find the target"):
        calculator.calculate(
            photometry(measurement("ref", 500.0)),
            target("missing"),
            (reference("Reference", "ref"),),
        )


def test_empty_photometry_is_rejected(calculator: DifferentialPhotometer) -> None:
    with pytest.raises(ObservationError, match="contains no measurements"):
        calculator.calculate(photometry(), target(), (reference("Reference", "ref"),))


def test_duplicate_measurement_identifier_is_rejected(calculator: DifferentialPhotometer) -> None:
    with pytest.raises(ObservationError, match="multiple photometry"):
        calculator.calculate(
            photometry(measurement("target", 1000.0), measurement("target", 900.0), measurement("ref", 500.0)),
            target(),
            (reference("Reference", "ref"),),
        )


def test_duplicate_reference_source_is_rejected(calculator: DifferentialPhotometer) -> None:
    with pytest.raises(ObservationError, match="same detected source 'ref'"):
        calculator.calculate(
            photometry(measurement("target", 1000.0), measurement("ref", 500.0)),
            target(),
            (reference("Reference 1", "ref"), reference("Reference 2", "ref")),
        )


def test_none_photometry_is_rejected(calculator: DifferentialPhotometer) -> None:
    with pytest.raises(ObservationError, match="completed photometry"):
        calculator.calculate(None, target(), (reference("Reference", "ref"),))  # type: ignore[arg-type]


def test_none_target_match_is_rejected(calculator: DifferentialPhotometer) -> None:
    with pytest.raises(ObservationError, match="resolved target"):
        calculator.calculate(
            photometry(measurement("target", 1000.0), measurement("ref", 500.0)),
            None,  # type: ignore[arg-type]
            (reference("Reference", "ref"),),
        )


def test_result_is_immutable(calculator: DifferentialPhotometer) -> None:
    result = calculator.calculate(
        photometry(measurement("target", 1000.0), measurement("ref", 500.0)),
        target(),
        (reference("Reference", "ref"),),
    )
    with pytest.raises(AttributeError):
        result.differential_flux = 123.0  # type: ignore[misc]
