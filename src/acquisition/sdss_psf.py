"""
SDSS PSF acquisition and reconstruction.

This module retrieves the SDSS psField product and reconstructs the
position-dependent 2-D PSF for a specified imaging field and band.

The reconstruction follows the algorithm documented by the Sloan Digital
Sky Survey:

    https://www.sdss4.org/dr17/imaging/images/

The SDSS psField contains spatially varying PSF eigen-images and
position-dependent coefficients. The PSF at a particular (row, col)
position is reconstructed as a linear combination of those eigen-images.

This module deliberately does NOT depend on SCARLET. It provides a native
SDSS PSF image which can subsequently be consumed by the SCARLET adapter.

Scientific rationale
--------------------
For this project, using the reconstructed SDSS psField PSF is preferable to
approximating the SDSS PSF using psfWidth, a single Gaussian sigma, or an
assumed Moffat profile. SDSS explicitly provides the information required to
reconstruct the spatially varying PSF.

References
----------
SDSS Collaboration / SDSS Imaging Data Files:
    https://www.sdss4.org/dr17/imaging/images/

SDSS PSF reconstruction:
    https://www.sdss4.org/dr17/algorithms/read_psf/

The implementation has been independently validated against
pydl.photoop.image.sdss_psf_recon for the current SDSS test psField.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_BANDS: Final[frozenset[str]] = frozenset("ugriz")

# SDSS PSF basis extensions:
#
#   u -> 1
#   g -> 2
#   r -> 3
#   i -> 4
#   z -> 5
#
# Extension 0 is the primary HDU.
_BAND_EXTENSIONS: Final[dict[str, int]] = {
    "u": 1,
    "g": 2,
    "r": 3,
    "i": 4,
    "z": 5,
}

# SDSS imaging native pixel scale.
#
# This is descriptive metadata only. It is NOT used in PSF reconstruction.
DEFAULT_PIXEL_SCALE_ARCSEC: Final[float] = 0.396


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SDSSPSFError(RuntimeError):
    """Base exception for SDSS PSF acquisition/reconstruction errors."""


class SDSSPSFDownloadError(SDSSPSFError):
    """Raised when the psField product cannot be downloaded."""


class SDSSPSFFormatError(SDSSPSFError):
    """Raised when a psField product has an unexpected structure."""


class SDSSPSFReconstructionError(SDSSPSFError):
    """Raised when a PSF cannot be reconstructed."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SDSSPSF:
    """
    Reconstructed SDSS point-spread function.

    Attributes
    ----------
    data:
        Normalised 2-D PSF image. The sum of all pixels is approximately 1.
    band:
        SDSS filter band.
    run:
        SDSS imaging run.
    rerun:
        SDSS imaging rerun.
    camcol:
        SDSS camera column.
    field:
        SDSS field number.
    row:
        Image row at which the PSF was reconstructed.
    col:
        Image column at which the PSF was reconstructed.
    pixel_scale_arcsec:
        SDSS native pixel scale.
    source_path:
        Local psField file used for reconstruction.
    """

    data: np.ndarray
    band: str
    run: int
    rerun: int
    camcol: int
    field: int
    row: float
    col: float
    pixel_scale_arcsec: float
    source_path: Path

    @property
    def shape(self) -> tuple[int, int]:
        """Return the PSF image shape."""
        return self.data.shape

    @property
    def total_flux(self) -> float:
        """Return the total PSF flux."""
        return float(np.sum(self.data))

    @property
    def peak(self) -> float:
        """Return the maximum PSF value."""
        return float(np.max(self.data))


@dataclass(frozen=True, slots=True)
class _MaterialisedPSFTable:
    """FITS-independent representation of one SDSS PSF band table."""

    nrow_b: np.ndarray
    ncol_b: np.ndarray
    rnrow: np.ndarray
    rncol: np.ndarray
    c: np.ndarray
    rrows: np.ndarray


# ---------------------------------------------------------------------------
# Main acquisition class
# ---------------------------------------------------------------------------


class SDSSPSFAcquirer:
    """
    Acquire and reconstruct a position-dependent SDSS PSF.

    Parameters
    ----------
    run:
        SDSS imaging run.
    rerun:
        SDSS imaging rerun. For the current project this is normally 301.
    camcol:
        SDSS camera column.
    field:
        SDSS field number.
    band:
        One of ``u``, ``g``, ``r``, ``i`` or ``z``.
    cache_dir:
        Directory in which the psField FITS file is cached.
    data_release:
        SDSS data release. The current implementation targets DR17.
    timeout:
        HTTP download timeout in seconds.
    """

    def __init__(
        self,
        *,
        run: int,
        rerun: int,
        camcol: int,
        field: int,
        band: str,
        cache_dir: str | Path = "data/raw/sdss/psf",
        data_release: int = 17,
        timeout: float = 60.0,
    ) -> None:
        self.run = int(run)
        self.rerun = int(rerun)
        self.camcol = int(camcol)
        self.field = int(field)
        self.band = band.lower()
        self.cache_dir = Path(cache_dir)
        self.data_release = int(data_release)
        self.timeout = float(timeout)

        self._validate_configuration()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        if self.band not in SUPPORTED_BANDS:
            raise ValueError(
                f"Unsupported SDSS band {self.band!r}. "
                f"Expected one of {sorted(SUPPORTED_BANDS)}."
            )

        if self.run < 0:
            raise ValueError("run must be non-negative.")

        if self.rerun < 0:
            raise ValueError("rerun must be non-negative.")

        if not 1 <= self.camcol <= 6:
            raise ValueError("camcol must be between 1 and 6.")

        if self.field < 0:
            raise ValueError("field must be non-negative.")

        if self.data_release < 1:
            raise ValueError("data_release must be positive.")

        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")

    # ------------------------------------------------------------------
    # Paths / URLs
    # ------------------------------------------------------------------

    @property
    def filename(self) -> str:
        """Return the canonical SDSS psField filename."""
        return (
            f"psField-{self.run:06d}-"
            f"{self.camcol}-"
            f"{self.field:04d}.fit"
        )

    @property
    def local_path(self) -> Path:
        """
        Return the local cache path.

        The run/rerun/camcol hierarchy is retained to avoid collisions
        between fields with otherwise similar identifiers.
        """
        return (
            self.cache_dir
            / f"dr{self.data_release}"
            / str(self.rerun)
            / str(self.run)
            / str(self.camcol)
            / self.filename
        )

    @property
    def sas_url(self) -> str:
        """
        Return the SDSS SAS URL for the psField product.
        """
        return (
            "https://data.sdss.org/sas/"
            f"dr{self.data_release}/"
            "eboss/photo/redux/"
            f"{self.rerun}/{self.run}/objcs/{self.camcol}/"
            f"{self.filename}"
        )

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def acquire_file(self, *, force: bool = False) -> Path:
        """
        Download the psField FITS file if it is not already cached.

        Parameters
        ----------
        force:
            If True, download the file even when a cached copy exists.

        Returns
        -------
        pathlib.Path
            Path to the local psField file.
        """
        if self.local_path.exists() and not force:
            return self.local_path

        self.local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import requests
        except ImportError as exc:
            raise SDSSPSFDownloadError(
                "The 'requests' package is required to download SDSS "
                "psField products."
            ) from exc

        temp_path = self.local_path.with_suffix(
            self.local_path.suffix + ".part"
        )

        try:
            response = requests.get(
                self.sas_url,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()

            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        handle.write(chunk)

            # Atomic replacement prevents a partially downloaded FITS file
            # from becoming the cached production file.
            temp_path.replace(self.local_path)

        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink()

            raise SDSSPSFDownloadError(
                "Unable to download SDSS psField product.\n"
                f"URL: {self.sas_url}\n"
                f"Destination: {self.local_path}"
            ) from exc

        return self.local_path

    # ------------------------------------------------------------------
    # FITS validation
    # ------------------------------------------------------------------

    @staticmethod
    def _column_lookup(data: Any) -> dict[str, str]:
        """
        Build a case-insensitive FITS column-name lookup.

        FITS files conventionally use uppercase column names, while code
        may naturally use lowercase names. NumPy record arrays preserve the
        actual FITS field names, so we resolve the names without mutating
        the record-array dtype.

        Returns
        -------
        dict
            Mapping of lowercase canonical names to actual FITS names.
        """
        names = list(data.dtype.names or ())

        return {
            str(name).lower(): str(name)
            for name in names
        }

    @staticmethod
    def _get_column(
        data: Any,
        lookup: dict[str, str],
        name: str,
    ) -> np.ndarray:
        """
        Retrieve a FITS table column case-insensitively.
        """
        actual_name = lookup.get(name.lower())

        if actual_name is None:
            available = list(data.dtype.names or ())

            raise SDSSPSFFormatError(
                f"SDSS psField PSF extension is missing required field "
                f"{name!r}. Available fields: {available!r}"
            )

        return data[actual_name]

    def _load_band_table(self, psfield_path: Path) -> _MaterialisedPSFTable:
        """
        Load and fully materialise the selected psField band table.

        SDSS stores RROWS as a FITS variable-length-array column. The VLA
        values reference the FITS heap, so returning an Astropy FITS_rec (or
        merely calling ``copy()`` on it) is unsafe once the FITS file closes.

        Every field needed for reconstruction is therefore converted to an
        independent NumPy array while the HDU is open. In particular, RROWS
        is explicitly stacked into a normal 2-D float64 array.
        """
        extension = _BAND_EXTENSIONS[self.band]

        try:
            with fits.open(
                psfield_path,
                mode="readonly",
                memmap=False,
            ) as hdul:
                if len(hdul) <= extension:
                    raise SDSSPSFFormatError(
                        f"psField file contains only {len(hdul)} HDUs; "
                        f"band {self.band!r} requires extension {extension}."
                    )

                data = getattr(hdul[extension], "data", None)
                if data is None:
                    raise SDSSPSFFormatError(
                        f"PSF FITS extension {extension} contains no data."
                    )

                lookup = self._column_lookup(data)
                required = ("nrow_b", "ncol_b", "rnrow", "rncol", "c", "rrows")
                missing = [name for name in required if name.lower() not in lookup]
                if missing:
                    raise SDSSPSFFormatError(
                        "SDSS psField PSF extension is missing required fields: "
                        f"{missing}. Available fields: {list(data.dtype.names or ())}"
                    )

                def copy_fixed(name: str) -> np.ndarray:
                    return np.array(
                        self._get_column(data, lookup, name),
                        dtype=np.float64,
                        copy=True,
                    )

                # RROWS is a variable-length FITS column. Materialise each
                # heap-backed vector before the FITS HDU is closed.
                rrows_column = self._get_column(data, lookup, "rrows")
                rows = [
                    np.array(item, dtype=np.float64, copy=True).reshape(-1)
                    for item in rrows_column
                ]

                if not rows:
                    rrows = np.empty((0, 0), dtype=np.float64)
                else:
                    lengths = {row.size for row in rows}
                    if len(lengths) != 1:
                        raise SDSSPSFFormatError(
                            "RROWS contains vectors of inconsistent lengths: "
                            f"{sorted(lengths)}"
                        )
                    rrows = np.stack(rows, axis=0)

                return _MaterialisedPSFTable(
                    nrow_b=copy_fixed("nrow_b"),
                    ncol_b=copy_fixed("ncol_b"),
                    rnrow=copy_fixed("rnrow"),
                    rncol=copy_fixed("rncol"),
                    c=copy_fixed("c"),
                    rrows=rrows,
                )

        except SDSSPSFError:
            raise
        except Exception as exc:
            raise SDSSPSFFormatError(
                f"Unable to read SDSS psField file: {psfield_path}"
            ) from exc


    # ------------------------------------------------------------------
    # Utility conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scalar(value: Any) -> int:
        """
        Convert an SDSS FITS scalar/one-element value to int.
        """
        array = np.asarray(value)

        if array.size == 0:
            raise SDSSPSFReconstructionError(
                "Encountered an empty scalar PSF parameter."
            )

        return int(array.reshape(-1)[0])

    @staticmethod
    def _as_float_array(value: Any) -> np.ndarray:
        """
        Convert an SDSS FITS field to a float NumPy array.
        """
        return np.asarray(value, dtype=np.float64)

    # ------------------------------------------------------------------
    # PSF reconstruction
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        *,
        row: float,
        col: float,
        psfield_path: str | Path | None = None,
        normalise: bool = True,
        clip_negative: bool = True,
    ) -> SDSSPSF:
        """Reconstruct the position-dependent native SDSS PSF."""
        row = float(row)
        col = float(col)

        if not np.isfinite(row) or not np.isfinite(col):
            raise ValueError("row and col must be finite.")
        if row < 0 or col < 0:
            raise ValueError("row and col must be non-negative.")

        path = Path(psfield_path) if psfield_path is not None else self.acquire_file()
        if not path.exists():
            raise FileNotFoundError(f"psField file not found: {path}")

        pstruct = self._load_band_table(path)

        try:
            nrow_b = self._scalar(pstruct.nrow_b[0])
            ncol_b = self._scalar(pstruct.ncol_b[0])
            rnrow = self._scalar(pstruct.rnrow[0])
            rncol = self._scalar(pstruct.rncol[0])
        except Exception as exc:
            raise SDSSPSFReconstructionError(
                "Unable to read PSF basis dimensions from psField."
            ) from exc

        if nrow_b <= 0 or ncol_b <= 0:
            raise SDSSPSFReconstructionError(
                f"Invalid PSF coefficient dimensions: nrow_b={nrow_b}, ncol_b={ncol_b}"
            )
        if rnrow <= 0 or rncol <= 0:
            raise SDSSPSFReconstructionError(
                f"Invalid reconstructed PSF dimensions: rnrow={rnrow}, rncol={rncol}"
            )

        # SDSS polynomial basis ordering:
        # index = row_power + nrow_b * col_power.
        nb = nrow_b * ncol_b
        row_scaled = row * 0.001
        col_scaled = col * 0.001
        coeffs = np.empty(nb, dtype=np.float64)

        for index in range(nb):
            row_power = index % nrow_b
            col_power = index // nrow_b
            coeffs[index] = (
                row_scaled ** row_power
                * col_scaled ** col_power
            )

        cmat = np.asarray(pstruct.c, dtype=np.float64)
        if cmat.ndim != 3:
            raise SDSSPSFReconstructionError(
                f"Unexpected SDSS coefficient matrix shape: {cmat.shape}"
            )

        n_templates = cmat.shape[0]
        if n_templates <= 0:
            raise SDSSPSFReconstructionError(
                "SDSS psField contains no PSF eigen-components."
            )

        # The psField C array is stored as:
        #
        #     C[component, col_power, row_power]
        #
        # while ``coeffs`` is ordered as:
        #
        #     index = row_power + nrow_b * col_power
        #
        # The physical polynomial basis can therefore be extracted from the
        # leading ncol_b x nrow_b block and flattened in C's native
        # component-col-row ordering.  This makes row_power the fastest-varying
        # coefficient index, matching the ``coeffs`` construction above.
        #
        # The FITS C array may be padded (e.g. 5x5 for a declared 3x3 basis),
        # so do NOT simply flatten the complete C array and take the first nb
        # elements.

        if cmat.shape[1] < ncol_b or cmat.shape[2] < nrow_b:
            raise SDSSPSFReconstructionError(
                "SDSS PSF coefficient matrix is smaller than the declared "
                f"polynomial basis: C={cmat.shape}, "
                f"required=({n_templates}, {ncol_b}, {nrow_b})"
            )

        c_coeff = cmat[:, :ncol_b, :nrow_b]

        # C is [component, col_power, row_power].
        # Flattening this 3-D block makes row_power the fastest-varying
        # coefficient index, matching:
        #
        #     index = row_power + nrow_b * col_power
        #
        c_flat = c_coeff.reshape(n_templates, nb)

        ecoeff = coeffs @ c_flat.T

        rrows = np.asarray(pstruct.rrows, dtype=np.float64)
        if rrows.ndim == 1:
            rrows = rrows.reshape(1, -1)
        elif rrows.ndim != 2:
            raise SDSSPSFReconstructionError(
                f"Unexpected SDSS eigen-PSF shape: {rrows.shape}"
            )

        if rrows.shape[0] != n_templates:
            if rrows.shape[1] == n_templates:
                rrows = rrows.T
            else:
                raise SDSSPSFReconstructionError(
                    "Number of RROWS eigen-images does not match coefficient "
                    f"components: RROWS={rrows.shape}, coefficients={n_templates}"
                )

        expected_pixels = rnrow * rncol
        if rrows.shape[1] != expected_pixels:
            raise SDSSPSFReconstructionError(
                "SDSS PSF eigen-image size is inconsistent with rnrow/rncol: "
                f"expected {expected_pixels} pixels but found {rrows.shape[1]}"
            )

        psf = (ecoeff @ rrows).reshape(rnrow, rncol)

        if not np.all(np.isfinite(psf)):
            raise SDSSPSFReconstructionError(
                "Reconstructed SDSS PSF contains non-finite values."
            )

        if clip_negative:
            psf = np.maximum(psf, 0.0)

        total = float(np.sum(psf))
        if not np.isfinite(total) or total <= 0:
            raise SDSSPSFReconstructionError(
                f"Reconstructed SDSS PSF has invalid total flux: {total}"
            )

        if normalise:
            psf = psf / total

        return SDSSPSF(
            data=np.asarray(psf, dtype=np.float64),
            band=self.band,
            run=self.run,
            rerun=self.rerun,
            camcol=self.camcol,
            field=self.field,
            row=row,
            col=col,
            pixel_scale_arcsec=DEFAULT_PIXEL_SCALE_ARCSEC,
            source_path=path,
        )


    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    def acquire(
        self,
        *,
        row: float,
        col: float,
        force_download: bool = False,
    ) -> SDSSPSF:
        """
        Acquire and reconstruct the PSF in one operation.

        This is the preferred high-level entry point for the acquisition
        layer.
        """
        path = self.acquire_file(
            force=force_download
        )

        return self.reconstruct(
            row=row,
            col=col,
            psfield_path=path,
        )


# ---------------------------------------------------------------------------
# Functional convenience API
# ---------------------------------------------------------------------------


def reconstruct_sdss_psf(
    *,
    run: int,
    rerun: int,
    camcol: int,
    field: int,
    band: str,
    row: float,
    col: float,
    cache_dir: str | Path = "data/raw/sdss/psf",
    data_release: int = 17,
) -> SDSSPSF:
    """
    Convenience function for one-shot SDSS PSF acquisition.

    Example
    -------
    >>> psf = reconstruct_sdss_psf(
    ...     run=5972,
    ...     rerun=301,
    ...     camcol=6,
    ...     field=117,
    ...     band="g",
    ...     row=700.0,
    ...     col=1000.0,
    ... )
    >>> psf.data.shape
    (51, 51)
    """
    acquirer = SDSSPSFAcquirer(
        run=run,
        rerun=rerun,
        camcol=camcol,
        field=field,
        band=band,
        cache_dir=cache_dir,
        data_release=data_release,
    )

    return acquirer.acquire(
        row=row,
        col=col,
    )


__all__ = [
    "SDSSPSF",
    "SDSSPSFAcquirer",
    "SDSSPSFError",
    "SDSSPSFDownloadError",
    "SDSSPSFFormatError",
    "SDSSPSFReconstructionError",
    "reconstruct_sdss_psf",
]