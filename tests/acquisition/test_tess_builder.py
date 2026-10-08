"""
Tests for the TESS Observation Builder.

These tests use a lightweight synthetic TESS Target Pixel File-shaped object.
The object is explicitly cast to TessTargetPixelFile for static typing because
the production builder only needs the small structural interface exercised
below; constructing a real Lightkurve TPF would make these unit tests depend
on FITS I/O and a real mission product.

The tests verify:
    * first-valid-cadence behaviour of build();
    * cadence-specific Observation construction;
    * build_sequence() selection;
    * uncertainty propagation;
    * TESS metadata/provenance;
    * invalid cadence handling;
    * rejection of malformed datasets.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from lightkurve import TessTargetPixelFile

from core.exceptions import ObservationError
from core.models import TargetConfig
from domain.enums import ObservationSource, Survey
from domain.models import Observation
from domain.tess_builder import TESSObservationBuilder


TARGET = TargetConfig(
    target_name="Synthetic TESS Target",
    right_ascension=120.0,
    declination=-30.0,
)


def _synthetic_wcs() -> WCS:
    """Return a minimal valid celestial WCS for the synthetic TPF."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [2.0, 2.0]
    wcs.wcs.crval = [120.0, -30.0]
    wcs.wcs.cdelt = [-0.01, 0.01]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _synthetic_header() -> fits.Header:
    """Return the metadata header expected by TESSObservationBuilder."""
    header = fits.Header()
    header["INSTRUME"] = "Synthetic TESS"
    header["FILTER"] = "TESS"
    header["EXPOSURE"] = 120.0
    return header


def _synthetic_dataset(
    *,
    flux: np.ndarray | None = None,
    flux_err: np.ndarray | None = None,
    times: np.ndarray | None = None,
    sector: int = 42,
    path: str = "synthetic_tess.fits",
) -> TessTargetPixelFile:
    """
    Construct a minimal TESS-TPF-shaped test object.

    The cast is intentional: unit tests should not instantiate a real
    Lightkurve TPF simply to exercise the builder's structural contract.
    """
    if flux is None:
        flux = np.array(
            [
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                ],
                [
                    [10.0, 11.0],
                    [12.0, 13.0],
                ],
                [
                    [20.0, 21.0],
                    [22.0, 23.0],
                ],
            ],
            dtype=np.float64,
        )

    if flux_err is None:
        flux_err = np.full(
            flux.shape,
            0.5,
            dtype=np.float64,
        )

    if times is None:
        times = np.array(
            [100.0, 101.0, 102.0],
            dtype=np.float64,
        )

    if len(times) != flux.shape[0]:
        raise ValueError("Synthetic times must match the flux cadence count.")

    dataset = SimpleNamespace(
        flux=np.asarray(flux, dtype=np.float64),
        flux_err=np.asarray(flux_err, dtype=np.float64),
        time=Time(
            np.asarray(times, dtype=np.float64),
            format="btjd",
            scale="tdb",
        ),
        wcs=_synthetic_wcs(),
        ra=120.0,
        dec=-30.0,
        sector=sector,
        camera=2,
        ccd=3,
        column=100,
        row=200,
        targetid=123456789,
        meta={
            "OBJECT": "Synthetic TESS Target",
            "SECTOR": sector,
            "CAMERA": 2,
            "CCD": 3,
        },
        path=path,
        get_header=lambda ext=1: _synthetic_header(),
    )

    return cast(
        TessTargetPixelFile,
        dataset,
    )


def _builder() -> TESSObservationBuilder:
    """Construct a builder using the synthetic target configuration.

    Native TESS PRF acquisition is deliberately disabled for these unit tests.
    The PRF acquisition path is covered independently by ``test_tess_psf.py``;
    these tests focus on Observation construction and cadence handling.
    """
    builder = TESSObservationBuilder(
        TARGET,
    )
    builder._try_acquire_psf = lambda dataset: None
    return builder


def test_build_returns_first_valid_cadence() -> None:
    """build() must skip invalid leading cadences and use the first valid one."""
    dataset = _synthetic_dataset()

    observation = _builder().build(
        dataset,
    )

    assert isinstance(
        observation,
        Observation,
    )

    assert observation.metadata.target_name == "Synthetic TESS Target"
    assert observation.metadata.observation_time == pytest.approx(101.0)
    assert observation.metadata.observation_time_scale == "tdb"
    assert observation.metadata.observation_time_format == "btjd"

    np.testing.assert_array_equal(
        observation.images["raw"].image.data,
        np.array(
            [
                [10.0, 11.0],
                [12.0, 13.0],
            ],
            dtype=np.float64,
        ),
    )


def test_build_preserves_cadence_uncertainty() -> None:
    """build() must attach the uncertainty belonging to the selected cadence."""
    flux_err = np.array(
        [
            [
                [9.0, 9.0],
                [9.0, 9.0],
            ],
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            [
                [0.5, 0.6],
                [0.7, 0.8],
            ],
        ],
        dtype=np.float64,
    )

    dataset = _synthetic_dataset(
        flux_err=flux_err,
    )

    observation = _builder().build(
        dataset,
    )

    uncertainty = observation.images["raw"].image.uncertainty

    assert uncertainty is not None

    np.testing.assert_array_equal(
        uncertainty.array,
        flux_err[1],
    )


def test_build_populates_tess_provenance() -> None:
    """The Observation must identify the source as space-based TESS data."""
    observation = _builder().build(
        _synthetic_dataset(
            sector=42,
            path="C:/synthetic/tess_sector_42.fits",
        ),
    )

    assert observation.provenance.source is ObservationSource.SPACE_BASED
    assert observation.provenance.survey is Survey.TESS
    assert observation.provenance.dataset_identifier == "TESS Sector 42"
    assert observation.provenance.filename == "C:/synthetic/tess_sector_42.fits"


def test_build_uses_configured_target_coordinates() -> None:
    """Configured target coordinates must be retained in Observation metadata."""
    observation = _builder().build(
        _synthetic_dataset(),
    )

    assert observation.metadata.right_ascension == pytest.approx(120.0)
    assert observation.metadata.declination == pytest.approx(-30.0)
    assert observation.metadata.instrument == "Synthetic TESS"
    assert observation.metadata.filter_name == "TESS"
    assert observation.metadata.exposure_time == pytest.approx(120.0)


def test_build_sequence_returns_all_valid_cadences() -> None:
    """build_sequence() must construct one Observation per valid cadence."""
    dataset = _synthetic_dataset()

    observations = _builder().build_sequence(
        dataset,
    )

    assert len(observations) == 2
    assert all(
        isinstance(observation, Observation)
        for observation in observations
    )

    assert observations[0].metadata.observation_time == pytest.approx(101.0)
    assert observations[1].metadata.observation_time == pytest.approx(102.0)

    np.testing.assert_array_equal(
        observations[0].images["raw"].image.data,
        dataset.flux[1],
    )

    np.testing.assert_array_equal(
        observations[1].images["raw"].image.data,
        dataset.flux[2],
    )


def test_build_sequence_respects_requested_cadence_indices() -> None:
    """build_sequence() must preserve the caller's requested cadence order."""
    dataset = _synthetic_dataset(
        flux=np.array(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                [
                    [10.0, 11.0],
                    [12.0, 13.0],
                ],
                [
                    [20.0, 21.0],
                    [22.0, 23.0],
                ],
            ],
            dtype=np.float64,
        ),
        times=np.array(
            [100.0, 101.0, 102.0],
            dtype=np.float64,
        ),
    )

    observations = _builder().build_sequence(
        dataset,
        cadence_indices=[2, 0],
    )

    assert len(observations) == 2
    assert observations[0].metadata.observation_time == pytest.approx(102.0)
    assert observations[1].metadata.observation_time == pytest.approx(100.0)

    np.testing.assert_array_equal(
        observations[0].images["raw"].image.data,
        dataset.flux[2],
    )

    np.testing.assert_array_equal(
        observations[1].images["raw"].image.data,
        dataset.flux[0],
    )


def test_build_rejects_out_of_range_cadence_index() -> None:
    """An explicitly requested cadence outside the TPF must be rejected."""
    dataset = _synthetic_dataset()

    with pytest.raises(
        ObservationError,
        match="not a usable cadence",
    ):
        _builder().build_sequence(
            dataset,
            cadence_indices=[99],
        )


def test_build_rejects_dataset_with_no_valid_cadences() -> None:
    """A TPF containing no finite cadence must not produce an Observation."""
    flux = np.full(
        (3, 2, 2),
        np.nan,
        dtype=np.float64,
    )

    dataset = _synthetic_dataset(
        flux=flux,
    )

    with pytest.raises(
        ObservationError,
        match="no usable finite cadences",
    ):
        _builder().build(
            dataset,
        )


def test_build_rejects_mismatched_time_and_flux_lengths() -> None:
    """Flux cadence count and time count must agree."""
    dataset = SimpleNamespace(
        flux=np.ones(
            (3, 2, 2),
            dtype=np.float64,
        ),
        flux_err=np.ones(
            (3, 2, 2),
            dtype=np.float64,
        ),
        time=Time(
            np.array(
                [100.0, 101.0],
                dtype=np.float64,
            ),
            format="btjd",
            scale="tdb",
        ),
        wcs=_synthetic_wcs(),
        ra=120.0,
        dec=-30.0,
        sector=42,
        path="synthetic_tess.fits",
        get_header=lambda ext=1: _synthetic_header(),
    )

    typed_dataset = cast(
        TessTargetPixelFile,
        dataset,
    )

    with pytest.raises(
        ObservationError,
        match="TESS time and flux cadence counts do not match",
    ):
        _builder().build(
            typed_dataset,
        )


def test_build_sequence_returns_immutable_tuple() -> None:
    """The sequence API must return the documented immutable tuple."""
    observations = _builder().build_sequence(
        _synthetic_dataset(),
    )

    assert isinstance(
        observations,
        tuple,
    )
    assert len(observations) == 2
