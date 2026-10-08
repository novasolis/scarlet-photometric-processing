"""
TESS dataset acquisition.

Production implementation of the acquisition layer for TESS Target Pixel
Files (TPFs) and TESSCut products.

This component is responsible only for locating, downloading and validating
observations. No scientific processing is performed here.

Acquisition modes
-----------------
Standard target-centred experiments use the existing TESS Target Pixel File
search:

    lightkurve.search_targetpixelfile()

Target-centred local-field experiments use TESSCut:

    lightkurve.search_tesscut()

A local field is no longer a scientific experiment type. It is a derived
spatial context around the configured target. When an experiment requests a
local-field acquisition, TESSCut provides the spatial footprint from which
the downstream source detector and LocalFieldSelector construct:

    target + nearest neighbouring sources

No scientific source detection, background subtraction or photometry is
performed here.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import lightkurve as lk
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from lightkurve import SearchResult
from lightkurve.targetpixelfile import TessTargetPixelFile

from acquisition.base import BaseAcquirer
from core.exceptions import (
    AcquisitionError,
    ConfigurationError,
)
from core.models import ExperimentConfig
from storage.tess import TESSStorage
from utils.logger import get_logger


class TESSAcquirer(BaseAcquirer[TessTargetPixelFile]):
    """
    Acquire TESS Target Pixel Files or TESSCut products from MAST.

    Acquisition mode is determined by the semantic experiment configuration.

    Standard target-centred experiments
    -----------------------------------
    Existing behaviour is preserved:

        lightkurve.search_targetpixelfile()

    Target-centred local-field experiments
    ---------------------------------------
    Coordinate-based TESSCut acquisition is used:

        lightkurve.search_tesscut()

    This provides the spatial context required by LocalFieldSelector while
    keeping the astronomical target as the centre of the experiment.

    The acquirer:

    1. validates the experiment configuration;
    2. resolves a deterministic local cache location;
    3. loads and validates an existing cached dataset when available;
    4. searches MAST when no valid cache exists;
    5. deterministically selects the preferred observation;
    6. downloads the selected product;
    7. validates the resulting TPF; and
    8. persists the validated dataset.

    No scientific image processing is performed here.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the TESS acquirer.

        Parameters
        ----------
        config
            Complete experiment configuration.

        logger
            Optional logger. If omitted, the project logger is used.
        """
        self._config = config
        self._logger = logger or get_logger(__name__)

    # ==================================================================
    # Public acquisition API
    # ==================================================================

    def acquire(
        self,
    ) -> TessTargetPixelFile:
        """
        Acquire and validate a TESS dataset.

        A validated cached dataset is reused when available. Otherwise,
        the preferred observation is located in MAST, downloaded,
        validated and persisted locally.

        Returns
        -------
        TessTargetPixelFile
            Validated TESS Target Pixel File.

        Raises
        ------
        ConfigurationError
            If the acquisition configuration is invalid.

        AcquisitionError
            If the requested observation cannot be found, downloaded,
            validated or cached.
        """
        #
        # Always validate configuration before checking the cache.
        #
        self._validate_configuration()

        storage = self._create_storage()
        acquisition = self._config.acquisition

        self._logger.info(
            "TESS acquisition configuration:"
        )

        self._logger.info(
            "  Target       : %s",
            self._config.target.target_name,
        )

        self._logger.info(
            "  Sector       : %s",
            (
                acquisition.sector
                if acquisition.sector is not None
                else "any"
            ),
        )

        self._logger.info(
            "  Author       : %s",
            (
                acquisition.author
                if acquisition.author is not None
                else "any"
            ),
        )

        self._logger.info(
            "  Cadence      : %s s",
            (
                acquisition.cadence_seconds
                if acquisition.cadence_seconds is not None
                else "any"
            ),
        )

        self._logger.info(
            "  Cutout size  : %s",
            (
                acquisition.cutout_size
                if acquisition.cutout_size is not None
                else "native product footprint"
            ),
        )

        self._logger.info(
            "  Acquisition  : %s",
            (
                "TESSCut local field"
                if self._uses_tesscut_acquisition()
                else "Target Pixel File"
            ),
        )

        #
        # Attempt to load cached dataset.
        #
        if storage.exists():
            self._logger.info(
                "Loading cached TESS dataset from '%s'.",
                storage.file_path,
            )

            try:
                tpf = storage.load()

                self._validate_dataset(
                    tpf,
                )

            except Exception as exc:
                self._logger.warning(
                    "Cached TESS dataset is unavailable or invalid: %s",
                    exc,
                )

                self._logger.info(
                    "Re-acquiring TESS dataset from MAST."
                )

            else:
                self._logger.info(
                    "Cached TESS dataset validated successfully."
                )

                return tpf

        #
        # Acquire a new dataset from MAST.
        #
        self._logger.info(
            "Acquiring TESS data for '%s'.",
            self._config.target.target_name,
        )

        search_result = self._search()

        product = self._select_product(
            search_result,
        )

        tpf = self._download(
            product,
        )

        self._validate_dataset(
            tpf,
        )

        self._logger.info(
            "Saving TESS dataset to '%s'.",
            storage.file_path,
        )

        try:
            storage.save(
                tpf,
            )

        except Exception as exc:
            raise AcquisitionError(
                "Failed to save TESS dataset to "
                f"'{storage.file_path}'."
            ) from exc

        self._logger.info(
            "Successfully acquired TESS Sector %s "
            "(shape=%s, cadences=%d).",
            tpf.sector,
            tpf.shape,
            len(tpf.time),
        )

        return tpf

    # ==================================================================
    # Storage
    # ==================================================================

    def _create_storage(
        self,
    ) -> TESSStorage:
        """
        Create storage using the complete acquisition identity.

        Returns
        -------
        TESSStorage
            Configuration-aware local TESS storage.
        """
        acquisition = self._config.acquisition

        return TESSStorage(
            target_name=self._config.target.target_name,
            sector=acquisition.sector,
            author=acquisition.author,
            cadence_seconds=acquisition.cadence_seconds,
            cutout_size=acquisition.cutout_size,
        )

    # ==================================================================
    # Configuration validation
    # ==================================================================

    def _validate_configuration(
        self,
    ) -> None:
        """
        Validate the TESS acquisition configuration.

        Raises
        ------
        ConfigurationError
            If an acquisition parameter is invalid.
        """
        target = self._config.target
        acquisition = self._config.acquisition

        if not target.target_name.strip():
            raise ConfigurationError(
                "TESS target name must not be empty."
            )

        if (
            acquisition.sector is not None
            and acquisition.sector <= 0
        ):
            raise ConfigurationError(
                "TESS sector must be greater than zero."
            )

        if (
            acquisition.cadence_seconds is not None
            and acquisition.cadence_seconds <= 0
        ):
            raise ConfigurationError(
                "TESS cadence must be greater than zero."
            )

        if (
            acquisition.author is not None
            and not acquisition.author.strip()
        ):
            raise ConfigurationError(
                "TESS author must not be empty."
            )

        if (
            acquisition.cutout_size is not None
            and acquisition.cutout_size <= 0
        ):
            raise ConfigurationError(
                "TESS cutout size must be greater than zero when specified."
            )

        #
        # Coordinate-based acquisition is required for experiments that
        # acquire a spatial field rather than a target-specific TPF.
        #
        if self._uses_tesscut_acquisition():

            if target.right_ascension is None:
                raise ConfigurationError(
                    "TESS field experiment target "
                    f"'{target.target_name}' requires right ascension."
                )

            if target.declination is None:
                raise ConfigurationError(
                    "TESS field experiment target "
                    f"'{target.target_name}' requires declination."
                )

            if not (
                -360.0
                <= target.right_ascension
                <= 360.0
            ):
                raise ConfigurationError(
                    "TESS field experiment right ascension must be "
                    "between -360 and 360 degrees."
                )

            if not (
                -90.0
                <= target.declination
                <= 90.0
            ):
                raise ConfigurationError(
                    "TESS field experiment declination must be "
                    "between -90 and 90 degrees."
                )

    # ==================================================================
    # Experiment mode
    # ==================================================================

    def _uses_tesscut_acquisition(
        self,
    ) -> bool:
        """
        Determine whether this target-centred experiment uses TESSCut.

        TESSCut is selected by the acquisition configuration rather than by
        scientific experiment type. This is essential to the target-centred
        architecture: ``TARGET_PHOTOMETRY`` may use either a standard SPOC
        Target Pixel File or a larger target-centred TESSCut product.

        The project's standard TESS acquisition configuration identifies a
        SPOC product through an explicit author and cadence preference. The
        local-field acquisition configuration deliberately leaves those
        preferences unspecified because TESSCut is the acquisition product.

        Therefore the current configuration contract is:

            author is None
            cadence_seconds is None

        means that coordinate-based TESSCut acquisition is requested.

        Returns
        -------
        bool
            ``True`` when the configured acquisition requires TESSCut.
        """
        acquisition = self._config.acquisition

        return (
            acquisition.author is None
            and acquisition.cadence_seconds is None
        )

    # ==================================================================
    # Search
    # ==================================================================

    def _search(
        self,
    ) -> SearchResult:
        """
        Search MAST for the appropriate TESS product.

        Target-centred experiments configured for local-field acquisition
        use TESSCut. All other experiments retain the existing
        ``search_targetpixelfile()`` mechanism.

        Returns
        -------
        SearchResult
            TESS observations returned by Lightkurve.

        Raises
        ------
        AcquisitionError
            If the MAST search fails or returns no observations.
        """
        if self._uses_tesscut_acquisition():
            return self._search_field_by_coordinates()

        #
        # Existing standard target-centred acquisition path.
        #
        return self._search_target_by_name()

    def _search_target_by_name(
        self,
    ) -> SearchResult:
        """
        Search MAST for an individual TESS target.

        This is the existing acquisition path used by target-oriented
        experiments such as HD 209458 and WASP-12.

        IMPORTANT
        ---------
        This method intentionally retains ``search_targetpixelfile()``
        so existing target experiments are not changed.
        """
        target = self._config.target.target_name

        self._logger.info(
            "Searching MAST for target '%s'.",
            target,
        )

        acquisition = self._config.acquisition

        try:
            if acquisition.sector is None:
                # Preserve the existing broad target search when no
                # experiment-level sector override is configured.
                result = lk.search_targetpixelfile(
                    target,
                    mission="TESS",
                )
            else:
                # An explicitly configured sector is a hard acquisition
                # constraint. Passing it to MAST prevents another sector
                # from being returned as an unintended fallback.
                result = lk.search_targetpixelfile(
                    target,
                    mission="TESS",
                    sector=acquisition.sector,
                )

        except Exception as exc:
            if acquisition.sector is not None:
                raise AcquisitionError(
                    f"Failed to search MAST for target '{target}' "
                    f"in requested Sector {acquisition.sector}."
                ) from exc

            raise AcquisitionError(
                f"Failed to search MAST for target '{target}'."
            ) from exc

        if len(result) == 0:
            if acquisition.sector is not None:
                raise AcquisitionError(
                    f"No TESS observations found for target '{target}' "
                    f"in requested Sector {acquisition.sector}."
                )

            raise AcquisitionError(
                f"No TESS observations found for target '{target}'."
            )

        self._logger.info(
            "Found %d available TESS observations.",
            len(result),
        )

        return result

    def _search_field_by_coordinates(
        self,
    ) -> SearchResult:
        """
        Search MAST for a TESSCut field using sky coordinates.

        TESSCut is specifically intended for obtaining a spatial cutout
        around a sky position from the TESS Full Frame Images.

        This is the correct acquisition mechanism for extended fields
        such as open clusters.

        Returns
        -------
        SearchResult
            TESSCut observations returned by Lightkurve.

        Raises
        ------
        AcquisitionError
            If the coordinates are unavailable, invalid, the MAST search
            fails, or no observations are found.
        """
        target = self._config.target

        if (
            target.right_ascension is None
            or target.declination is None
        ):
            raise AcquisitionError(
                "Coordinate-based TESS field acquisition requires both "
                "right ascension and declination for "
                f"'{target.target_name}'."
            )

        try:
            skycoord = SkyCoord(
                ra=target.right_ascension,
                dec=target.declination,
                unit=(u.deg, u.deg),
                frame="icrs",
            )

        except Exception as exc:
            raise AcquisitionError(
                "Failed to construct sky coordinates for TESS field "
                f"'{target.target_name}'."
            ) from exc

        self._logger.info(
            "Searching MAST for target-centred local field '%s' using TESSCut.",
            target.target_name,
        )

        self._logger.info(
            "  RA           : %.6f deg",
            target.right_ascension,
        )

        self._logger.info(
            "  Dec          : %.6f deg",
            target.declination,
        )

        try:
            #
            # IMPORTANT:
            #
            # TESSCut is deliberately used here instead of
            # search_targetpixelfile().
            #
            # Extended fields such as NGC 2516 can have TESSCut
            # observations even when no SPOC TPF exists for the
            # field itself.
            #
            result = lk.search_tesscut(
                skycoord,
            )

        except Exception as exc:
            raise AcquisitionError(
                "Failed to search MAST for TESSCut field "
                f"'{target.target_name}' at "
                f"RA={target.right_ascension:.6f}, "
                f"Dec={target.declination:.6f}."
            ) from exc

        if len(result) == 0:
            raise AcquisitionError(
                "No TESSCut observations found for target-centred local field "
                f"'{target.target_name}' at "
                f"RA={target.right_ascension:.6f}, "
                f"Dec={target.declination:.6f}."
            )

        self._logger.info(
            "Found %d available TESSCut field observations.",
            len(result),
        )

        return result

    # ==================================================================
    # Product selection
    # ==================================================================

    def _select_product(
        self,
        search_result: SearchResult,
    ) -> SearchResult:
        """
        Select the most appropriate TESS observation.

        Selection follows this deterministic rule and preference order:

        1. Requested sector is a hard constraint when configured.
        2. Appropriate processing author, when configured.
        3. Exact requested cadence/exposure time, when available.
        4. Closest available cadence/exposure time.
        5. Original archive ordering as the final tie-breaker.

        For TESSCut local-field searches the archive author is ``TESScut``.
        Because TESSCut is the acquisition mechanism itself, the standard
        SPOC author preference must not prevent a valid TESSCut product from
        being selected.

        Sector is a strict requirement when explicitly configured. Author
        and cadence remain preferences. When no sector is configured, the
        existing broad search and deterministic preference behaviour is
        preserved.

        Parameters
        ----------
        search_result
            Search results returned by Lightkurve.

        Returns
        -------
        SearchResult
            Single-row SearchResult containing the selected product.

        Raises
        ------
        AcquisitionError
            If no usable observation metadata is available.
        """
        acquisition = self._config.acquisition

        table = search_result.table

        if table is None:
            raise AcquisitionError(
                "TESS search results did not contain a metadata table."
            )

        table = cast(
            Table,
            table,
        )

        if len(table) == 0:
            raise AcquisitionError(
                "No TESS observations were available."
            )

        is_local_field = self._uses_tesscut_acquisition()

        candidates: list[
            tuple[
                tuple[float, ...],
                int,
            ]
        ] = []

        for index in range(len(table)):

            row = cast(
                Any,
                table[index],
            )

            sector = self._extract_sector(
                row,
                table,
            )

            # An explicitly configured sector is a hard acquisition
            # constraint. Do not allow the deterministic preference logic
            # below to fall back to a different sector.
            if (
                acquisition.sector is not None
                and sector != acquisition.sector
            ):
                self._logger.debug(
                    "Skipping TESS observation at index %d because "
                    "Sector %s does not match requested Sector %s.",
                    index,
                    sector if sector is not None else "Unknown",
                    acquisition.sector,
                )
                continue

            author = self._extract_author(
                row,
                table,
            )

            cadence = self._extract_cadence(
                row,
                table,
            )

            sector_match = (
                1
                if (
                    acquisition.sector is not None
                    and sector == acquisition.sector
                )
                else 0
            )

            #
            # Author handling differs slightly for TESSCut.
            #
            # Normal target searches continue to honour the configured
            # author preference exactly as before.
            #
            # Local-field searches return TESScut products. "SPOC" is not
            # a meaningful alternative acquisition author for TESSCut, so
            # TESScut is considered the correct acquisition source.
            #
            if is_local_field:
                author_match = (
                    1
                    if (
                        author is not None
                        and author.upper() == "TESSCUT"
                    )
                    else 0
                )
            else:
                author_match = (
                    1
                    if (
                        acquisition.author is not None
                        and author is not None
                        and author.upper()
                        == acquisition.author.strip().upper()
                    )
                    else 0
                )

            #
            # Cadence/exposure preference.
            #
            if (
                is_local_field
                and acquisition.cadence_seconds is not None
                and cadence is not None
            ):
                #
                # TESSCut search results expose the source FFI exposure
                # time rather than the 120-second SPOC cadence requested
                # by the normal target configuration.
                #
                # Therefore cadence remains a useful diagnostic value,
                # but it must not cause the field acquisition to reject
                # a valid TESSCut product.
                #
                cadence_match = 0
                cadence_distance = 0.0

            elif acquisition.cadence_seconds is None:
                cadence_match = 0
                cadence_distance = 0.0

            elif cadence is None:
                cadence_match = 0
                cadence_distance = float("inf")

            else:
                requested_cadence = float(
                    acquisition.cadence_seconds,
                )

                cadence_match = (
                    1
                    if cadence == requested_cadence
                    else 0
                )

                cadence_distance = abs(
                    cadence - requested_cadence,
                )

            #
            # Lexicographic preference:
            #
            #   sector match
            #       >
            #   author/acquisition-source match
            #       >
            #   exact cadence
            #       >
            #   cadence proximity
            #       >
            #   original archive ordering
            #
            preference_key = (
                float(sector_match),
                float(author_match),
                float(cadence_match),
                -cadence_distance,
                -float(index),
            )

            candidates.append(
                (
                    preference_key,
                    index,
                )
            )

        if not candidates:
            if acquisition.sector is not None:
                raise AcquisitionError(
                    "No usable TESS observations were available for "
                    f"the requested Sector {acquisition.sector}."
                )

            raise AcquisitionError(
                "No usable TESS observations were available."
            )

        _, best_index = max(
            candidates,
            key=lambda item: item[0],
        )

        selected_sector = self._extract_sector(
            table[best_index],
            table,
        )

        selected_author = self._extract_author(
            table[best_index],
            table,
        )

        selected_cadence = self._extract_cadence(
            table[best_index],
            table,
        )

        self._logger.info(
            "Selected TESS observation:"
        )

        self._logger.info(
            "  Sector  : %s",
            (
                selected_sector
                if selected_sector is not None
                else "Unknown"
            ),
        )

        self._logger.info(
            "  Author  : %s",
            (
                selected_author
                if selected_author is not None
                else "Unknown"
            ),
        )

        self._logger.info(
            "  Cadence : %s s",
            (
                selected_cadence
                if selected_cadence is not None
                else "Unknown"
            ),
        )

        if acquisition.sector is not None:
            self._logger.info(
                "  Requested sector : %s",
                acquisition.sector,
            )

        if acquisition.author is not None:
            self._logger.info(
                "  Requested author : %s",
                acquisition.author,
            )

        if acquisition.cadence_seconds is not None:
            self._logger.info(
                "  Requested cadence: %s s",
                acquisition.cadence_seconds,
            )

        if is_local_field:
            self._logger.info(
                "  Acquisition mode : TESSCut local-field product"
            )

        return search_result[
            best_index : best_index + 1
        ]

    # ==================================================================
    # Search-result metadata extraction
    # ==================================================================

    @staticmethod
    def _extract_sector(
        row: Any,
        table: Table,
    ) -> int | None:
        """
        Extract the TESS sector from a search-result row.

        Returns
        -------
        int | None
            Sector number when available.
        """
        if "sequence_number" not in table.colnames:
            return None

        try:
            value = row["sequence_number"]

            if value is None:
                return None

            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _extract_author(
        row: Any,
        table: Table,
    ) -> str | None:
        """
        Extract the processing/acquisition author from a search-result row.
        """
        if "author" not in table.colnames:
            return None

        value = row["author"]

        if value is None:
            return None

        author = str(value).strip()

        return author or None

    @staticmethod
    def _extract_cadence(
        row: Any,
        table: Table,
    ) -> float | None:
        """
        Extract the exposure/cadence value from a search-result row.

        Lightkurve exposes TESS search-product exposure time through the
        ``exptime`` field.

        For normal SPOC TPF products this is used as the cadence-selection
        proxy.

        For TESSCut products it represents the underlying FFI exposure time
        and is retained primarily as metadata.
        """
        if "exptime" not in table.colnames:
            return None

        try:
            value = row["exptime"]

            if value is None:
                return None

            cadence = float(value)

            if not cadence > 0.0:
                return None

            return cadence

        except (
            TypeError,
            ValueError,
        ):
            return None

    # ==================================================================
    # Download
    # ==================================================================

    def _download(
        self,
        product: SearchResult,
    ) -> TessTargetPixelFile:
        """
        Download the selected TESS product.

        For normal Target Pixel File products, the archive-native footprint
        is preserved and ``cutout_size`` is not passed to ``download()``.

        For TESSCut field products, ``cutout_size`` is supported by the
        TESSCut download mechanism. If it is ``None``, the native/default
        TESSCut footprint is preserved.

        Parameters
        ----------
        product
            Single-row SearchResult containing the selected product.

        Returns
        -------
        TessTargetPixelFile
            Downloaded Target Pixel File.

        Raises
        ------
        AcquisitionError
            If the download fails or returns no dataset.
        """
        cutout_size = self._config.acquisition.cutout_size

        is_local_field = self._uses_tesscut_acquisition()

        if is_local_field:

            if cutout_size is not None:
                self._logger.info(
                    "Downloading TESSCut local-field product with "
                    "cutout_size=%d.",
                    cutout_size,
                )

            else:
                self._logger.info(
                    "Downloading TESSCut local-field product using "
                    "the native/default spatial footprint."
                )

        else:

            if cutout_size is not None:
                self._logger.warning(
                    "Requested cutout_size=%d is not applicable to the "
                    "selected TESS Target Pixel File product. "
                    "Preserving the native TPF spatial footprint.",
                    cutout_size,
                )

            else:
                self._logger.info(
                    "Downloading native TESS Target Pixel File "
                    "without requesting a spatial cutout."
                )

        try:

            if is_local_field and cutout_size is not None:
                #
                # TESSCut products support cutout_size at download time.
                #
                tpf = product.download(
                    cutout_size=cutout_size,
                )

            else:
                #
                # Existing target-oriented mechanism.
                #
                tpf = product.download()

        except Exception as exc:
            raise AcquisitionError(
                "Failed to download the selected TESS product."
            ) from exc

        if tpf is None:
            raise AcquisitionError(
                "Download returned no Target Pixel File."
            )

        if not isinstance(
            tpf,
            TessTargetPixelFile,
        ):
            raise AcquisitionError(
                "Downloaded TESS product is not a "
                "TessTargetPixelFile."
            )

        return tpf

    # ==================================================================
    # Dataset validation
    # ==================================================================

    def _validate_dataset(
        self,
        tpf: TessTargetPixelFile,
    ) -> None:
        """
        Validate a TESS Target Pixel File.

        Validation is structural rather than scientific. Scientific
        suitability is assessed by downstream processing and evaluation.

        Parameters
        ----------
        tpf
            TESS Target Pixel File.

        Raises
        ------
        AcquisitionError
            If the TPF is incomplete or structurally invalid.
        """
        if tpf is None:
            raise AcquisitionError(
                "TESS dataset cannot be None."
            )

        # Validate explicit acquisition constraints against the dataset
        # itself. This is essential for cache correctness: a cache path
        # identifies the requested configuration, but its contents must
        # independently prove that they satisfy that configuration.
        requested_sector = self._config.acquisition.sector
        if requested_sector is not None:
            try:
                actual_sector = tpf.sector
            except Exception as exc:
                raise AcquisitionError(
                    "Unable to access TESS dataset sector information "
                    "required to validate the requested acquisition "
                    f"Sector {requested_sector}."
                ) from exc

            if actual_sector != requested_sector:
                raise AcquisitionError(
                    "TESS dataset sector does not match the requested "
                    f"acquisition sector: requested Sector {requested_sector}, "
                    f"dataset contains Sector {actual_sector}."
                )

        try:
            time = tpf.time

        except Exception as exc:
            raise AcquisitionError(
                "Unable to access TESS observation times."
            ) from exc

        if len(time) == 0:
            raise AcquisitionError(
                "TESS dataset contains no cadences."
            )

        try:
            flux = tpf.flux

        except Exception as exc:
            raise AcquisitionError(
                "Unable to access TESS flux data."
            ) from exc

        if flux is None:
            raise AcquisitionError(
                "TESS dataset contains no flux cube."
            )

        try:
            flux_shape = tuple(
                flux.shape,
            )

        except Exception as exc:
            raise AcquisitionError(
                "Unable to determine TESS flux cube shape."
            ) from exc

        if len(flux_shape) != 3:
            raise AcquisitionError(
                "TESS flux cube has unexpected dimensionality."
            )

        cadence_count = flux_shape[0]

        if cadence_count <= 0:
            raise AcquisitionError(
                "TESS flux cube contains no cadence frames."
            )

        if len(time) != cadence_count:
            raise AcquisitionError(
                "TESS time and flux cadence counts do not match."
            )

        spatial_shape = flux_shape[1:]

        if (
            spatial_shape[0] <= 0
            or spatial_shape[1] <= 0
        ):
            raise AcquisitionError(
                "TESS flux cube contains an invalid spatial footprint."
            )

        #
        # Individual invalid pixels/cadences are permitted. The dataset
        # must nevertheless contain some finite flux values.
        #
        flux_array = getattr(
            flux,
            "value",
            flux,
        )

        try:
            finite_flux_count = int(
                np.count_nonzero(
                    np.isfinite(
                        flux_array,
                    )
                )
            )

        except Exception as exc:
            raise AcquisitionError(
                "Unable to validate numerical TESS flux data."
            ) from exc

        if finite_flux_count == 0:
            raise AcquisitionError(
                "TESS dataset contains no finite flux values."
            )

        #
        # WCS is required by downstream target/source resolution.
        #
        try:
            wcs = tpf.wcs

        except Exception as exc:
            raise AcquisitionError(
                "Unable to access TESS Target Pixel File WCS."
            ) from exc

        if wcs is None:
            raise AcquisitionError(
                "TESS Target Pixel File does not contain WCS information."
            )

        #
        # Target coordinates are required by the downstream
        # TargetSourceResolver.
        #
        try:
            target_ra = tpf.ra
            target_dec = tpf.dec

        except Exception as exc:
            raise AcquisitionError(
                "Unable to access TESS target coordinates."
            ) from exc

        if target_ra is None or target_dec is None:
            raise AcquisitionError(
                "TESS Target Pixel File does not contain target "
                "right ascension and declination."
            )

        self._logger.info(
            "Dataset validated successfully "
            "(sector=%s, shape=%s, cadences=%d).",
            tpf.sector,
            tpf.shape,
            len(tpf.time),
        )