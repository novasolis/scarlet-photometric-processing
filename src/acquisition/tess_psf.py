"""
TESS Pixel Response Function (PRF) acquisition and reconstruction.

This module provides a small, survey-specific acquisition boundary for the
mission-provided TESS PRF products.  The implementation follows the MAST TESS
PRF retrieval procedure: each detector position is represented by a 117x117
interleaved FITS image containing 13x13 PRFs sampled on a 9x9 intra-pixel
grid, and the four nearest detector grid locations are linearly interpolated
in detector position.

The resulting PRF is returned as a normalised 13x13 image suitable for the
project's survey-independent PSFData model.

The PRF is a mission calibration product, not an empirical fit to the science
cadence.  It therefore belongs in the acquisition/domain boundary and is
consumed by SCARLET as the observation PSF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import re

import numpy as np
from astropy.io import fits


DEFAULT_CACHE_DIR: Final[Path] = Path("data/raw/tess/psf")
DEFAULT_MAST_BASE_URL: Final[str] = (
    "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles"
)
DEFAULT_PIXEL_SCALE_ARCSEC: Final[float] = 21.0
PRF_IMAGE_SIZE: Final[int] = 13
PRF_INTERLEAVED_SIZE: Final[int] = 117
PRF_GRID_SIZE: Final[int] = 9

# The PRF coordinate system is offset from the FFI/TPF science-product
# coordinate system by 44 columns.  This is documented by MAST for the
# mission PRF products.
PRF_COLUMN_OFFSET_FROM_SCIENCE: Final[float] = 44.0

# TESS PRF archive directory indexes are authoritative for the actual FITS
# filenames.  We discover the filename rather than hard-coding a historical
# filename prefix, because MAST may retain the products while changing or
# correcting the filename prefix.
_PRF_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'(?P<name>tess[^"<>\s]+?-prf-(?P<camera>\d)-(?P<ccd>\d)-'
    r'row(?P<row>\d{4})-col(?P<col>\d{4})\.fits)',
    re.IGNORECASE,
)

# Detector locations at which the mission provides PRFs.
PRF_ROWS: Final[tuple[int, ...]] = (1, 513, 1025, 1536, 2048)
PRF_COLS: Final[tuple[int, ...]] = (45, 557, 1069, 1580, 2092)


class TESSPSFError(RuntimeError):
    """Base exception for TESS PRF acquisition/reconstruction errors."""


class TESSPSFDownloadError(TESSPSFError):
    """Raised when a mission PRF FITS product cannot be acquired."""


class TESSPSFFormatError(TESSPSFError):
    """Raised when a mission PRF FITS product has an unexpected structure."""


class TESSPSFReconstructionError(TESSPSFError):
    """Raised when a position-specific TESS PRF cannot be reconstructed."""


@dataclass(frozen=True, slots=True)
class TESSPSF:
    """Reconstructed, normalised TESS pixel response function."""

    data: np.ndarray
    sector: int
    camera: int
    ccd: int
    row: float
    col: float
    pixel_scale_arcsec: float
    source_paths: tuple[Path, ...]

    @property
    def shape(self) -> tuple[int, int]:
        if self.data.ndim != 2:
            raise ValueError(f"TESS PSF data must be 2-D; received ndim={self.data.ndim}.")
        return int(self.data.shape[0]), int(self.data.shape[1])

    @property
    def total_flux(self) -> float:
        return float(np.sum(self.data))

    @property
    def peak(self) -> float:
        return float(np.max(self.data))


@dataclass(frozen=True, slots=True)
class _PRFLocation:
    col: int
    row: int


class TESSPSFAcquirer:
    """
    Acquire and reconstruct a position-dependent TESS PRF.

    Parameters
    ----------
    sector
        TESS sector number. Sectors 1-3 use the mission ``start_s0001`` PRF
        calibration set; Sector 4 onward uses ``start_s0004``.
    camera
        TESS camera number, 1-4.
    ccd
        TESS CCD number, 1-4.
    cache_dir
        Local directory used to cache PRF FITS files.
    data_release
        Reserved for future PRF calibration-set selection. The current MAST
        PRF archive exposes the mission's start_s0001/start_s0004 products.
    base_url
        Root URL of the MAST TESS PRF archive.
    timeout
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        *,
        sector: int,
        camera: int,
        ccd: int,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        data_release: int = 17,
        base_url: str = DEFAULT_MAST_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.sector = self._validate_positive_int("sector", sector)
        self.camera = self._validate_range("camera", camera, 1, 4)
        self.ccd = self._validate_range("ccd", ccd, 1, 4)
        self.cache_dir = Path(cache_dir)
        self.data_release = int(data_release)
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        self._remote_filename_cache: dict[tuple[int, int], str] = {}
        self._remote_directory_index: dict[tuple[int, int, int], dict[tuple[int, int], str]] = {}

        if self.timeout <= 0 or not np.isfinite(self.timeout):
            raise ValueError("timeout must be finite and greater than zero.")

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> int:
        """Validate and return a strictly positive integer."""

        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer; received {value!r}.")

        result = int(value)
        if result <= 0:
            raise ValueError(f"{name} must be a positive integer; received {value!r}.")

        return result

    @staticmethod
    def _validate_range(name: str, value: int, minimum: int, maximum: int) -> int:
        """Validate an integer against an inclusive range."""

        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"{name} must be an integer in [{minimum}, {maximum}]; "
                f"received {value!r}."
            )

        result = int(value)
        if not minimum <= result <= maximum:
            raise ValueError(
                f"{name} must be an integer in [{minimum}, {maximum}]; "
                f"received {value!r}."
            )

        return result

    # ------------------------------------------------------------------
    # PRF calibration-set naming
    # ------------------------------------------------------------------

    def _calibration_directory(self) -> str:
        """Return the MAST PRF calibration directory for this sector."""
        return "start_s0001" if self.sector <= 3 else "start_s0004"

    def _relative_directory(self) -> Path:
        """Return the remote/local directory containing this CCD's PRFs."""
        return Path(self._calibration_directory()) / (
            f"cam{self.camera}_ccd{self.ccd}"
        )

    def _remote_directory_url(self) -> str:
        """Return the public MAST directory index URL."""
        return f"{self.base_url}/{self._relative_directory().as_posix()}/"

    def _legacy_filename_candidates(self, *, row: int, col: int) -> tuple[str, ...]:
        """Return historical filename candidates as a final fallback."""
        if self.sector <= 3:
            prefixes = ("tess2018243163600", "tess2018243163601")
        else:
            prefixes = (
                "tess2019107181900",
                "tess2019107181901",
                "tess2019107181902",
            )
        return tuple(
            f"{prefix}-prf-{self.camera}-{self.ccd}-"
            f"row{row:04d}-col{col:04d}.fits"
            for prefix in prefixes
        )

    def _resolve_remote_filename(self, *, row: int, col: int) -> str:
        """Resolve the exact FITS filename published by the MAST directory."""
        key = (int(row), int(col))
        cached = self._remote_filename_cache.get(key)
        if cached is not None:
            return cached

        directory_key = (self.sector, self.camera, self.ccd)
        index = self._remote_directory_index.get(directory_key)
        if index is None:
            directory_url = self._remote_directory_url()
            request = Request(
                directory_url,
                headers={"User-Agent": "photometry-tess-prf/1.0"},
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    html = response.read().decode("utf-8", errors="replace")
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                for filename in self._legacy_filename_candidates(row=row, col=col):
                    self._remote_filename_cache[key] = filename
                    return filename
                raise TESSPSFDownloadError(
                    f"Unable to inspect TESS PRF directory '{directory_url}'."
                ) from exc

            index = {}
            for match in _PRF_FILENAME_PATTERN.finditer(html):
                if (
                    int(match.group("camera")) != self.camera
                    or int(match.group("ccd")) != self.ccd
                ):
                    continue
                location = (
                    int(match.group("row")),
                    int(match.group("col")),
                )
                index[location] = match.group("name")
            self._remote_directory_index[directory_key] = index

        filename = index.get(key)
        if filename is None:
            directory_url = self._remote_directory_url()
            raise TESSPSFDownloadError(
                "MAST PRF directory contains no FITS product for "
                f"camera={self.camera}, ccd={self.ccd}, row={row}, col={col}: "
                f"{directory_url}"
            )

        self._remote_filename_cache[key] = filename
        return filename

    def _relative_path(self, *, row: int, col: int) -> Path:
        """Return the local/remote relative path for one PRF FITS product."""
        return self._relative_directory() / self._resolve_remote_filename(
            row=row,
            col=col,
        )

    def _remote_url(self, *, row: int, col: int) -> str:
        return f"{self.base_url}/{self._relative_path(row=row, col=col).as_posix()}"

    def _local_path(self, *, row: int, col: int) -> Path:
        """
        Return the cache path for one PRF.

        Cached products are located without contacting MAST first.  This is
        important for reproducible/offline reruns once the mission products
        have already been acquired.
        """
        directory = self.cache_dir / self._relative_directory()

        if directory.exists():
            pattern = (
                f"*-prf-{self.camera}-{self.ccd}-"
                f"row{row:04d}-col{col:04d}.fits"
            )
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]

        return directory / self._resolve_remote_filename(row=row, col=col)

    # ------------------------------------------------------------------
    # Download / cache
    # ------------------------------------------------------------------

    def acquire_file(
        self,
        *,
        row: int,
        col: int,
        force: bool = False,
    ) -> Path:
        """Download one mission PRF FITS file, using the local cache."""

        if row not in PRF_ROWS or col not in PRF_COLS:
            raise TESSPSFDownloadError(
                "PRF file coordinates must be one of the mission calibration "
                f"grid positions: rows={PRF_ROWS}, cols={PRF_COLS}."
            )

        destination = self._local_path(row=row, col=col)

        if destination.exists() and not force:
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        filename = self._resolve_remote_filename(row=row, col=col)
        url = f"{self.base_url}/{self._relative_directory().as_posix()}/{filename}"

        try:
            with urlopen(
                Request(url, headers={"User-Agent": "photometry-tess-prf/1.0"}),
                timeout=self.timeout,
            ) as response:
                payload = response.read()

        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            # If directory discovery selected a stale/incorrect entry, retry
            # the historical prefixes.  This is deliberately limited to the
            # known mission naming variants and never fabricates a PRF.
            payload = None
            last_exc: Exception = exc
            for candidate in self._legacy_filename_candidates(row=row, col=col):
                candidate_url = (
                    f"{self.base_url}/{self._relative_directory().as_posix()}/"
                    f"{candidate}"
                )
                try:
                    with urlopen(
                        Request(
                            candidate_url,
                            headers={"User-Agent": "photometry-tess-prf/1.0"},
                        ),
                        timeout=self.timeout,
                    ) as response:
                        payload = response.read()
                    filename = candidate
                    self._remote_filename_cache[(row, col)] = candidate
                    break
                except (HTTPError, URLError, TimeoutError, OSError) as retry_exc:
                    last_exc = retry_exc

            if payload is None:
                raise TESSPSFDownloadError(
                    "Unable to download TESS PRF product for "
                    f"camera={self.camera}, ccd={self.ccd}, row={row}, col={col} "
                    f"from '{url}'."
                ) from last_exc

        if not payload:
            raise TESSPSFDownloadError(
                f"TESS PRF download returned an empty file: '{url}'."
            )

        try:
            destination.write_bytes(payload)
        except OSError as exc:
            raise TESSPSFDownloadError(
                f"Unable to cache TESS PRF product at '{destination}'."
            ) from exc

        return destination

    # ------------------------------------------------------------------
    # FITS parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _read_interleaved(path: Path) -> np.ndarray:
        """Read and validate one 117x117 interleaved PRF image."""

        try:
            # Use astropy's public getdata() helper rather than accessing
            # ``hdul[0].data`` directly. This avoids an incomplete/unknown
            # ``data`` attribute in some Astropy/Pylance stub versions while
            # preserving the same primary-HDU semantics.
            raw_data = fits.getdata(path, memmap=False)

        except (OSError, ValueError, TypeError) as exc:
            raise TESSPSFFormatError(
                f"Unable to read TESS PRF FITS product '{path}'."
            ) from exc

        if raw_data is None:
            raise TESSPSFFormatError(
                f"Primary HDU of TESS PRF product '{path}' contains no data."
            )

        data = np.asarray(raw_data, dtype=np.float64)

        if data.shape != (PRF_INTERLEAVED_SIZE, PRF_INTERLEAVED_SIZE):
            raise TESSPSFFormatError(
                "Expected a 117x117 interleaved TESS PRF image; "
                f"received shape={data.shape} from '{path}'."
            )

        if not np.all(np.isfinite(data)):
            raise TESSPSFFormatError(
                f"TESS PRF product '{path}' contains non-finite values."
            )

        return data

    @staticmethod
    def _fraction_offsets(col: float, row: float) -> tuple[int, int]:
        """Map intra-pixel fractions onto the mission's 9x9 PRF grid."""

        col_fraction = float(np.remainder(col, 1.0))
        row_fraction = float(np.remainder(row, 1.0))

        col_offset = (
            PRF_GRID_SIZE
            - int(np.round(PRF_GRID_SIZE * col_fraction))
            - 1
        )
        row_offset = (
            PRF_GRID_SIZE
            - int(np.round(PRF_GRID_SIZE * row_fraction))
            - 1
        )

        return (
            int(np.clip(col_offset, 0, PRF_GRID_SIZE - 1)),
            int(np.clip(row_offset, 0, PRF_GRID_SIZE - 1)),
        )

    @staticmethod
    def _extract_subpixel_prf(
        interleaved: np.ndarray,
        *,
        col_offset: int,
        row_offset: int,
    ) -> np.ndarray:
        """Extract one 13x13 PRF from a 117x117 interleaved image."""

        if not (
            0 <= col_offset < PRF_GRID_SIZE
            and 0 <= row_offset < PRF_GRID_SIZE
        ):
            raise TESSPSFFormatError(
                "TESS PRF sub-pixel offsets are outside the 9x9 grid."
            )

        indices = np.arange(PRF_IMAGE_SIZE, dtype=int) * PRF_GRID_SIZE
        row_indices = row_offset + indices
        col_indices = col_offset + indices

        return np.asarray(
            interleaved[np.ix_(row_indices, col_indices)],
            dtype=np.float64,
        )

    @staticmethod
    def _four_nearest_locations(
        *,
        col: float,
        row: float,
    ) -> tuple[_PRFLocation, _PRFLocation, _PRFLocation, _PRFLocation]:
        """Return the four mission grid positions surrounding the detector point."""

        row_candidates = sorted(PRF_ROWS, key=lambda value: abs(value - row))[:2]
        col_candidates = sorted(PRF_COLS, key=lambda value: abs(value - col))[:2]

        row_candidates.sort()
        col_candidates.sort()

        if len(row_candidates) != 2 or len(col_candidates) != 2:
            raise TESSPSFReconstructionError(
                "Unable to determine four surrounding TESS PRF grid positions."
            )

        return (
            _PRFLocation(col=col_candidates[0], row=row_candidates[0]),
            _PRFLocation(col=col_candidates[1], row=row_candidates[0]),
            _PRFLocation(col=col_candidates[0], row=row_candidates[1]),
            _PRFLocation(col=col_candidates[1], row=row_candidates[1]),
        )

    @staticmethod
    def _interpolate(
        prfs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        *,
        col: float,
        row: float,
        locations: tuple[_PRFLocation, _PRFLocation, _PRFLocation, _PRFLocation],
    ) -> np.ndarray:
        """Bilinearly interpolate four spatially adjacent PRFs."""

        p11, p21, p12, p22 = prfs
        c0 = float(locations[0].col)
        c1 = float(locations[1].col)
        r0 = float(locations[0].row)
        r1 = float(locations[2].row)

        if c0 == c1 or r0 == r1:
            raise TESSPSFReconstructionError(
                "TESS PRF interpolation requires two distinct row and column "
                "grid coordinates."
            )

        d_col = (col - c0) / (c1 - c0)
        d_row = (row - r0) / (r1 - r0)

        tmp1 = p11 + (p21 - p11) * d_col
        tmp2 = p12 + (p22 - p12) * d_col

        return np.asarray(
            tmp1 + (tmp2 - tmp1) * d_row,
            dtype=np.float64,
        )

    @staticmethod
    def _to_prf_coordinates(*, row: float, col: float) -> tuple[float, float]:
        """
        Convert TESS science-product coordinates to mission PRF coordinates.

        TESS documents a 44-column offset between the FFI/TPF coordinate
        system and the PRF coordinate system. Rows are unchanged.
        """
        return row, col - PRF_COLUMN_OFFSET_FROM_SCIENCE

    # ------------------------------------------------------------------
    # Public reconstruction API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        *,
        row: float,
        col: float,
        force_download: bool = False,
    ) -> TESSPSF:
        """
        Acquire and reconstruct the mission PRF at one CCD location.

        ``row`` and ``col`` are detector coordinates in the TESS
        science-product (FFI/TPF) coordinate system.  The documented 44-column
        offset is applied before looking up and interpolating the mission PRF.
        The resulting PRF is normalised to unit total flux.
        """

        try:
            row = float(row)
            col = float(col)
        except (TypeError, ValueError) as exc:
            raise TESSPSFReconstructionError(
                "TESS PRF row and column must be numeric."
            ) from exc

        if not np.isfinite(row) or not np.isfinite(col):
            raise TESSPSFReconstructionError(
                f"TESS PRF position must be finite: row={row!r}, col={col!r}."
            )

        # Science-product coordinates include 44 collateral columns relative
        # to the mission PRF coordinate system.
        prf_row, prf_col = self._to_prf_coordinates(row=row, col=col)

        # The mission PRF grid spans the detector focal plane.  The PRF
        # coordinate system may legitimately extend slightly beyond the
        # light-sensitive 2048-pixel array, so validate against the documented
        # PRF grid envelope rather than the science-image envelope.
        if not (0.0 <= prf_row <= 2135.0 and -43.0 <= prf_col <= 2135.0):
            raise TESSPSFReconstructionError(
                "TESS PRF position lies outside the expected detector range "
                f"after science->PRF conversion: row={prf_row:.6f}, "
                f"col={prf_col:.6f}."
            )

        col_offset, row_offset = self._fraction_offsets(prf_col, prf_row)
        locations = self._four_nearest_locations(col=prf_col, row=prf_row)

        extracted: list[np.ndarray] = []
        source_paths: list[Path] = []

        for location in locations:
            path = self.acquire_file(
                row=location.row,
                col=location.col,
                force=force_download,
            )

            interleaved = self._read_interleaved(path)
            extracted.append(
                self._extract_subpixel_prf(
                    interleaved,
                    col_offset=col_offset,
                    row_offset=row_offset,
                )
            )
            source_paths.append(path)

        if len(extracted) != 4:
            raise TESSPSFReconstructionError(
                f"Expected four extracted TESS PRFs; received {len(extracted)}."
            )

        prfs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = (
            extracted[0],
            extracted[1],
            extracted[2],
            extracted[3],
        )

        psf = self._interpolate(
            prfs,
            col=prf_col,
            row=prf_row,
            locations=locations,
        )

        if psf.shape != (PRF_IMAGE_SIZE, PRF_IMAGE_SIZE):
            raise TESSPSFReconstructionError(
                "TESS PRF reconstruction produced an unexpected shape: "
                f"{psf.shape}."
            )

        if not np.all(np.isfinite(psf)):
            raise TESSPSFReconstructionError(
                "TESS PRF reconstruction produced non-finite values."
            )

        total = float(np.sum(psf))
        if not np.isfinite(total) or total <= 0.0:
            raise TESSPSFReconstructionError(
                f"TESS PRF has invalid total flux: {total!r}."
            )

        psf = psf / total

        return TESSPSF(
            data=np.asarray(psf, dtype=np.float64),
            sector=self.sector,
            camera=self.camera,
            ccd=self.ccd,
            row=row,
            col=col,
            pixel_scale_arcsec=DEFAULT_PIXEL_SCALE_ARCSEC,
            source_paths=tuple(source_paths),
        )

    def acquire(
        self,
        *,
        row: float,
        col: float,
        force_download: bool = False,
    ) -> TESSPSF:
        """Preferred high-level API for one-shot TESS PRF acquisition."""

        return self.reconstruct(
            row=row,
            col=col,
            force_download=force_download,
        )


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MAST_BASE_URL",
    "DEFAULT_PIXEL_SCALE_ARCSEC",
    "TESSPSF",
    "TESSPSFAcquirer",
    "TESSPSFError",
    "TESSPSFDownloadError",
    "TESSPSFFormatError",
    "TESSPSFReconstructionError",
]
