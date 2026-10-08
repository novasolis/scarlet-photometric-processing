"""
TESS dataset storage.

Provides persistent local storage for TESS Target Pixel Files.

Datasets are stored using the native FITS format produced by
Lightkurve, ensuring compatibility with astronomy tools and
reproducibility of experiments.

The storage identity includes the acquisition parameters that can
change the selected dataset or downloaded cutout. This prevents
different experiment configurations from accidentally sharing or
overwriting the same cached observation.
"""

from __future__ import annotations

from pathlib import Path
import re

from lightkurve import TessTargetPixelFile


class TESSStorage:
    """
    Local storage for TESS Target Pixel Files.

    Storage identity is determined by the target and acquisition
    parameters that influence the selected dataset.

    Parameters
    ----------
    target_name
        Benchmark target identifier.

    sector
        Requested TESS observing sector, or ``None`` when no specific
        sector is requested.

    author
        Requested processing pipeline/author, or ``None`` when no
        specific author is requested.

    cadence_seconds
        Preferred observation cadence in seconds, or ``None`` when
        no cadence preference is configured.

    cutout_size
        Requested TESS cutout size in pixels. ``None`` means that
        the native spatial footprint of the selected TESS product
        is preserved.
    """

    _FILENAME = "tess_tpf.fits"

    def __init__(
        self,
        *,
        target_name: str,
        sector: int | None,
        author: str | None,
        cadence_seconds: int | None,
        cutout_size: int | None,
    ) -> None:
        """
        Initialise TESS dataset storage.

        Parameters
        ----------
        target_name
            Benchmark target identifier.

        sector
            Requested TESS observing sector.

        author
            Requested processing author.

        cadence_seconds
            Preferred cadence in seconds.

        cutout_size
            Requested cutout size in pixels, or ``None`` to preserve
            the native spatial footprint.
        """

        self._validate_parameters(
            target_name=target_name,
            sector=sector,
            author=author,
            cadence_seconds=cadence_seconds,
            cutout_size=cutout_size,
        )

        self._directory = (
            Path("data")
            / "raw"
            / "tess"
            / self._normalise_name(
                target_name,
            )
            / self._configuration_name(
                sector=sector,
                author=author,
                cadence_seconds=cadence_seconds,
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
            Full path to the cached Target Pixel File.
        """

        return self.directory / self._FILENAME

    def exists(
        self,
    ) -> bool:
        """
        Determine whether the dataset exists locally.

        Returns
        -------
        bool
            ``True`` when the cached FITS file exists.
        """

        return self.file_path.is_file()

    def load(
        self,
    ) -> TessTargetPixelFile:
        """
        Load a cached TESS dataset.

        Returns
        -------
        TessTargetPixelFile
            Cached Target Pixel File.

        Raises
        ------
        FileNotFoundError
            If the dataset does not exist.
        """

        if not self.exists():
            raise FileNotFoundError(
                f"TESS dataset not found: {self.file_path}"
            )

        return TessTargetPixelFile(
            self.file_path,
        )

    def save(
        self,
        dataset: TessTargetPixelFile,
    ) -> None:
        """
        Persist a TESS dataset.

        Parameters
        ----------
        dataset
            Target Pixel File to store.

        Raises
        ------
        ValueError
            If the supplied dataset is invalid.

        Notes
        -----
        The dataset is stored in native FITS format using Lightkurve's
        ``to_fits`` implementation.
        """

        if dataset is None:
            raise ValueError(
                "TESS dataset cannot be None."
            )

        if not isinstance(
            dataset,
            TessTargetPixelFile,
        ):
            raise ValueError(
                "TESS dataset must be a "
                "TessTargetPixelFile."
            )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset.to_fits(
            self.file_path,
            overwrite=True,
        )

    @staticmethod
    def _validate_parameters(
        *,
        target_name: str,
        sector: int | None,
        author: str | None,
        cadence_seconds: int | None,
        cutout_size: int | None,
    ) -> None:
        """
        Validate storage identity parameters.

        Raises
        ------
        ValueError
            If a storage parameter is invalid.
        """

        if not target_name.strip():
            raise ValueError(
                "TESS storage target name must not be empty."
            )

        if (
            sector is not None
            and sector <= 0
        ):
            raise ValueError(
                "TESS storage sector must be greater than zero."
            )

        if (
            cadence_seconds is not None
            and cadence_seconds <= 0
        ):
            raise ValueError(
                "TESS storage cadence must be greater than zero."
            )

        if (
            author is not None
            and not author.strip()
        ):
            raise ValueError(
                "TESS storage author must not be empty."
            )

        if (
            cutout_size is not None
            and cutout_size <= 0
        ):
            raise ValueError(
                "TESS storage cutout size must be greater than zero "
                "when specified."
            )

    @classmethod
    def _configuration_name(
        cls,
        *,
        sector: int | None,
        author: str | None,
        cadence_seconds: int | None,
        cutout_size: int | None,
    ) -> str:
        """
        Build a deterministic filesystem-safe configuration name.

        Examples
        --------
        Native TPF::

            sector-42_author-spoc_cadence-120_native

        Cutout TPF::

            sector-42_author-spoc_cadence-120_cutout-15

        Sector-independent native TPF::

            sector-any_author-spoc_cadence-120_native
        """

        sector_value = (
            "any"
            if sector is None
            else str(sector)
        )

        author_value = (
            "any"
            if author is None
            else cls._normalise_component(
                author,
            )
        )

        cadence_value = (
            "any"
            if cadence_seconds is None
            else str(cadence_seconds)
        )

        cutout_value = (
            "native"
            if cutout_size is None
            else f"cutout-{cutout_size}"
        )

        return (
            f"sector-{sector_value}"
            f"_author-{author_value}"
            f"_cadence-{cadence_value}"
            f"_{cutout_value}"
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

    @staticmethod
    def _normalise_component(
        value: str,
    ) -> str:
        """
        Convert a configuration component into a filesystem-safe value.

        Parameters
        ----------
        value
            Configuration value.

        Returns
        -------
        str
            Filesystem-safe representation.
        """

        normalised = (
            value
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