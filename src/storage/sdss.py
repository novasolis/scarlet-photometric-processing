"""
SDSS dataset storage.

Provides persistent local storage for SDSS imaging datasets.

Datasets are stored in their native FITS format, preserving the
original image and metadata returned by the SDSS archive.

The storage identity includes the target coordinates and requested
acquisition mode so that different SDSS acquisition configurations
cannot accidentally share the same cached dataset.
"""

from __future__ import annotations

from pathlib import Path
import re

from astropy.io.fits import (
    HDUList,
    open as fits_open,
)


class SDSSStorage:
    """
    Local storage for SDSS imaging datasets.

    Storage identity is determined by the configured target name,
    celestial coordinates and requested image acquisition mode.

    Parameters
    ----------
    target_name
        Benchmark target identifier.

    right_ascension
        Target right ascension in decimal degrees (ICRS).

    declination
        Target declination in decimal degrees (ICRS).

    cutout_size
        Requested image cutout size in pixels. ``None`` indicates that
        the native/full SDSS image should be preserved.
    """

    _FILENAME = "sdss_image.fits"

    def __init__(
        self,
        *,
        target_name: str,
        right_ascension: float,
        declination: float,
        cutout_size: int | None,
    ) -> None:
        """
        Initialise SDSS dataset storage.

        Parameters
        ----------
        target_name
            Benchmark target identifier.

        right_ascension
            Target right ascension in decimal degrees.

        declination
            Target declination in decimal degrees.

        cutout_size
            Requested cutout size in pixels, or ``None`` for the
            native/full SDSS image.

        Raises
        ------
        ValueError
            If a storage parameter is invalid.
        """

        self._validate_parameters(
            target_name=target_name,
            right_ascension=right_ascension,
            declination=declination,
            cutout_size=cutout_size,
        )

        self._directory = (
            Path("data")
            / "raw"
            / "sdss"
            / self._normalise_name(
                target_name,
            )
            / self._configuration_name(
                right_ascension=right_ascension,
                declination=declination,
                cutout_size=cutout_size,
            )
        )

    @property
    def directory(
        self,
    ) -> Path:
        """
        Return the dataset directory.

        Returns
        -------
        Path
            Directory containing the cached dataset.
        """

        return self._directory

    @property
    def file_path(
        self,
    ) -> Path:
        """
        Return the dataset file path.

        Returns
        -------
        Path
            Full path to the cached FITS dataset.
        """

        return self.directory / self._FILENAME

    def exists(
        self,
    ) -> bool:
        """
        Determine whether a cached SDSS dataset exists.

        Returns
        -------
        bool
            ``True`` when the cached FITS file exists.
        """

        return self.file_path.is_file()

    def load(
        self,
    ) -> HDUList:
        """
        Load a cached SDSS dataset.

        Returns
        -------
        HDUList
            Cached FITS dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset does not exist.

        OSError
            If the FITS file cannot be opened.
        """

        if not self.exists():
            raise FileNotFoundError(
                f"SDSS dataset not found: {self.file_path}"
            )

        return fits_open(
            self.file_path,
            mode="readonly",
            memmap=False,
        )

    def save(
        self,
        dataset: HDUList,
    ) -> None:
        """
        Persist an SDSS dataset.

        Parameters
        ----------
        dataset
            FITS dataset to store.

        Raises
        ------
        ValueError
            If the supplied dataset is invalid.
        """

        if dataset is None:
            raise ValueError(
                "SDSS dataset cannot be None."
            )

        if not isinstance(
            dataset,
            HDUList,
        ):
            raise ValueError(
                "SDSS dataset must be an astropy.io.fits.HDUList."
            )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset.writeto(
            self.file_path,
            overwrite=True,
        )

    @staticmethod
    def _validate_parameters(
        *,
        target_name: str,
        right_ascension: float,
        declination: float,
        cutout_size: int | None,
    ) -> None:
        """
        Validate SDSS storage identity parameters.

        Parameters
        ----------
        target_name
            Benchmark target identifier.

        right_ascension
            Target right ascension in decimal degrees.

        declination
            Target declination in decimal degrees.

        cutout_size
            Requested cutout size, or ``None`` for native/full-image
            storage.

        Raises
        ------
        ValueError
            If a storage parameter is invalid.
        """

        if not target_name.strip():
            raise ValueError(
                "SDSS storage target name must not be empty."
            )

        if not (
            0.0
            <= right_ascension
            < 360.0
        ):
            raise ValueError(
                "SDSS right ascension must be in the range "
                "[0, 360) degrees."
            )

        if not (
            -90.0
            <= declination
            <= 90.0
        ):
            raise ValueError(
                "SDSS declination must be in the range "
                "[-90, 90] degrees."
            )

        if cutout_size is None:
            return

        if (
            not isinstance(
                cutout_size,
                int,
            )
            or isinstance(
                cutout_size,
                bool,
            )
        ):
            raise ValueError(
                "SDSS cutout size must be an integer or None."
            )

        if cutout_size <= 0:
            raise ValueError(
                "SDSS cutout size must be greater than zero "
                "when specified."
            )

    @classmethod
    def _configuration_name(
        cls,
        *,
        right_ascension: float,
        declination: float,
        cutout_size: int | None,
    ) -> str:
        """
        Build a deterministic filesystem-safe configuration name.

        Examples
        --------
        Cutout mode::

            ra-132.825000_dec-11.817000_cutout-128

        Native/full-image mode::

            ra-132.825000_dec-11.817000_native
        """

        coordinate_prefix = (
            f"ra-{right_ascension:.6f}"
            f"_dec-{declination:.6f}"
        )

        if cutout_size is None:
            return (
                f"{coordinate_prefix}"
                "_native"
            )

        return (
            f"{coordinate_prefix}"
            f"_cutout-{cutout_size}"
        )

    @staticmethod
    def _normalise_name(
        target_name: str,
    ) -> str:
        """
        Convert a target name into a filesystem-safe directory name.

        Parameters
        ----------
        target_name
            Benchmark target name.

        Returns
        -------
        str
            Filesystem-safe directory name.
        """

        normalised = (
            target_name
            .strip()
            .lower()
        )

        normalised = re.sub(
            r"[^a-z0-9._-]+",
            "_",
            normalised,
        )

        normalised = re.sub(
            r"_+",
            "_",
            normalised,
        )

        return normalised.strip(
            "._-"
        )