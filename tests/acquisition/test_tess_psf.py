from __future__ import annotations

import numpy as np
import pytest

from astropy.io import fits

from acquisition.tess_psf import (
    DEFAULT_PIXEL_SCALE_ARCSEC,
    PRF_COLS,
    PRF_ROWS,
    TESSPSFAcquirer,
    TESSPSFDownloadError,
)


# ============================================================================
# Synthetic PRF fixture
# ============================================================================


def _synthetic_interleaved(value: float) -> np.ndarray:
    """
    Build a synthetic 117x117 interleaved TESS PRF product.

    Each 13x13 sub-pixel PRF is encoded at one of the 9x9 intra-pixel
    offsets. The value uniquely identifies the requested offset so that
    extraction can be tested deterministically.
    """
    data = np.zeros((117, 117), dtype=np.float64)

    for row_offset in range(9):
        for col_offset in range(9):
            rows = row_offset + np.arange(13) * 9
            cols = col_offset + np.arange(13) * 9

            data[np.ix_(rows, cols)] = (
                value
                + row_offset * 100
                + col_offset
            )

    return data


# ============================================================================
# Basic coordinate / grid tests
# ============================================================================


def test_fraction_offsets() -> None:
    """Verify extraction of the 9x9 intra-pixel PRF offset."""
    acquirer = TESSPSFAcquirer(
        sector=61,
        camera=4,
        ccd=1,
    )

    col_offset, row_offset = acquirer._fraction_offsets(
        95.4,
        1137.6,
    )

    assert (col_offset, row_offset) == (4, 3)


def test_four_nearest_locations() -> None:
    """Verify the four detector PRF grid locations surrounding a target."""
    acquirer = TESSPSFAcquirer(
        sector=61,
        camera=4,
        ccd=1,
    )

    locations = acquirer._four_nearest_locations(
        col=95.4,
        row=1137.6,
    )

    assert len(locations) == 4

    assert all(
        location.row in PRF_ROWS
        for location in locations
    )

    assert all(
        location.col in PRF_COLS
        for location in locations
    )


# ============================================================================
# Interleaved PRF extraction
# ============================================================================


def test_extract_subpixel_prf() -> None:
    """
    Verify extraction of one 13x13 sub-pixel PRF from the 117x117 product.
    """
    acquirer = TESSPSFAcquirer(
        sector=61,
        camera=4,
        ccd=1,
    )

    data = _synthetic_interleaved(1.0)

    prf = acquirer._extract_subpixel_prf(
        data,
        col_offset=2,
        row_offset=4,
    )

    assert prf.shape == (13, 13)

    # value = base + row_offset * 100 + col_offset
    #       = 1 + 4 * 100 + 2
    #       = 403
    assert np.all(prf == 403.0)


# ============================================================================
# Reconstruction / interpolation
# ============================================================================


def test_reconstruct_interpolates_and_normalises(tmp_path) -> None:
    """
    Verify that four PRF products are acquired, interpolated and normalised.
    """
    acquirer = TESSPSFAcquirer(
        sector=61,
        camera=4,
        ccd=1,
        cache_dir=tmp_path,
    )

    locations = acquirer._four_nearest_locations(
        col=95.4,
        row=1137.6,
    )

    paths = {}

    for index, location in enumerate(locations, start=1):
        path = tmp_path / f"{index}.fits"

        fits.PrimaryHDU(
            _synthetic_interleaved(float(index))
        ).writeto(path)

        paths[(location.row, location.col)] = path

    acquirer.acquire_file = (
        lambda *, row, col, force=False:
        paths[(row, col)]
    )  # type: ignore[method-assign]

    result = acquirer.reconstruct(
        row=1137.6,
        col=95.4,
    )

    assert result.data.shape == (13, 13)

    assert np.all(
        np.isfinite(result.data)
    )

    assert np.isclose(
        result.total_flux,
        1.0,
    )

    assert len(result.source_paths) == 4

    assert result.sector == 61
    assert result.camera == 4
    assert result.ccd == 1

    assert result.row == pytest.approx(
        1137.6
    )

    assert result.col == pytest.approx(
        95.4
    )

    assert result.pixel_scale_arcsec == DEFAULT_PIXEL_SCALE_ARCSEC


# ============================================================================
# Cache behaviour
# ============================================================================


def test_acquire_file_uses_cached_product(tmp_path) -> None:
    """
    Verify that an existing cached PRF is returned without downloading it.
    """
    acquirer = TESSPSFAcquirer(
        sector=10,
        camera=3,
        ccd=1,
        cache_dir=tmp_path,
    )

    path = acquirer._local_path(
        row=1,
        col=1580,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        b"cached-prf"
    )

    result = acquirer.acquire_file(
        row=1,
        col=1580,
    )

    assert result == path
    assert result.read_bytes() == b"cached-prf"


# ============================================================================
# Mission URL construction
# ============================================================================


def test_ngc_3532_prf_url() -> None:
    """
    Verify the PRF URL generated for the NGC 3532 observation.

    NGC 3532:
        Sector : 10
        Camera : 3
        CCD    : 1
    """
    acquirer = TESSPSFAcquirer(
        sector=10,
        camera=3,
        ccd=1,
    )

    url = acquirer._remote_url(
        row=1,
        col=1580,
    )

    assert url.startswith(
        "https://archive.stsci.edu/"
    )

    assert (
        "missions/tess/models/prf_fitsfiles/"
        in url
    )

    assert (
        "cam3_ccd1/"
        in url
    )

    assert (
        "prf-3-1-row0001-col1580.fits"
        in url
    )


# ============================================================================
# PRF grid validation
# ============================================================================


def test_invalid_prf_row_rejected(tmp_path) -> None:
    """Verify that non-mission PRF grid rows are rejected."""
    acquirer = TESSPSFAcquirer(
        sector=10,
        camera=3,
        ccd=1,
        cache_dir=tmp_path,
    )

    invalid_row = 999

    with pytest.raises(
        TESSPSFDownloadError
    ):
        acquirer.acquire_file(
            row=invalid_row,
            col=1580,
        )


def test_invalid_prf_column_rejected(tmp_path) -> None:
    """Verify that non-mission PRF grid columns are rejected."""
    acquirer = TESSPSFAcquirer(
        sector=10,
        camera=3,
        ccd=1,
        cache_dir=tmp_path,
    )

    invalid_col = 9999

    with pytest.raises(
        TESSPSFDownloadError
    ):
        acquirer.acquire_file(
            row=1,
            col=invalid_col,
        )


# ============================================================================
# Real NGC 3532 native PRF integration test
# ============================================================================


@pytest.mark.integration
def test_ngc_3532_native_prf_integration() -> None:
    """
    Acquire and reconstruct the native TESS PRF for the NGC 3532 target.

    This is deliberately an integration test because it accesses the
    mission-provided TESS PRF calibration products.

    NGC 3532 observation:
        Sector : 10
        Camera : 3
        CCD    : 1
        Row    : 451.735681
        Col    : 1615.616024
    """
    acquirer = TESSPSFAcquirer(
        sector=10,
        camera=3,
        ccd=1,
    )

    try:
        result = acquirer.reconstruct(
            row=451.735681,
            col=1615.616024,
        )

    except TESSPSFDownloadError as exc:
        pytest.fail(
            "Native NGC 3532 TESS PRF acquisition failed.\n\n"
            f"{exc}\n\n"
            "Check the production TESS PRF URL/calibration-set "
            "mapping before proceeding to the SCARLET production run."
        )

    assert result.shape == (13, 13)

    assert result.sector == 10
    assert result.camera == 3
    assert result.ccd == 1

    assert result.row == pytest.approx(
        451.735681
    )

    assert result.col == pytest.approx(
        1615.616024
    )

    assert np.all(
        np.isfinite(result.data)
    )

    assert result.total_flux == pytest.approx(
        1.0,
        abs=1e-10,
    )

    assert result.peak > 0.0

    assert len(result.source_paths) == 4

    assert all(
        path.exists()
        for path in result.source_paths
    )