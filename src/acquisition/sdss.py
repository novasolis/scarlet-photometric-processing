"""
SDSS image acquisition.

Acquires calibrated Sloan Digital Sky Survey (SDSS) imaging data using
Astroquery.

The acquisition process follows the native SDSS workflow:

    Sky coordinates
        ↓
    SDSS catalogue search
        ↓
    Nearest catalogue entry selection
        ↓
    FITS image download
        ↓
    Acquisition metadata preservation
        ↓
    Optional WCS-aware image cutout
        ↓
    Local FITS storage

When ``cutout_size`` is ``None``, the native SDSS image returned by the
archive is preserved without spatial cropping.

Scientific processing such as background subtraction, source detection,
photometry, deblending and evaluation is deliberately excluded from this
component.

The acquisition layer also preserves the SDSS imaging identifiers required
by downstream Observation construction and native psField PSF acquisition:

    RUN
    RERUN
    CAMCOL
    FIELD
    FILTER

These identifiers are written to the final image HDU header so that the
ObservationBuilder remains independent of Astroquery catalogue objects.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io.fits import HDUList
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from astroquery.sdss import SDSS

from acquisition.base import BaseAcquirer
from core.exceptions import AcquisitionError
from core.models import ExperimentConfig
from storage.sdss import SDSSStorage
from utils.logger import get_logger


class SDSSAcquirer(BaseAcquirer[HDUList]):
    """
    Acquire calibrated SDSS imaging data.

    The acquirer performs archive retrieval and optional deterministic
    spatial extraction only. It does not perform scientific image
    processing.

    Parameters
    ----------
    config
        Complete experiment configuration.

    logger
        Optional application logger.
    """

    # ------------------------------------------------------------------
    # Acquisition constants
    # ------------------------------------------------------------------

    _SEARCH_RADIUS_ARCMIN = 2.0

    _FALLBACK_SEARCH_RADIUS_ARCMIN = 3.0

    # Current production experiment uses the SDSS g band.
    _BAND = "g"

    # Fixed data release for reproducibility.
    _DATA_RELEASE = 17

    # ------------------------------------------------------------------
    # Documented imaging-field fallbacks
    # ------------------------------------------------------------------

    _KNOWN_IMAGING_FIELD_FALLBACKS: dict[
        str,
        tuple[int, int, int, int],
    ] = {
        # M13:
        # run, rerun, camcol, field
        "m13": (3226, 301, 5, 127),
    }

    # Internal marker used only for synthetic fallback tables.
    _FIELD_FALLBACK_MARKER = "__sdss_field_fallback"

    # FITS keywords required by downstream SDSS ObservationBuilder /
    # native SDSS PSF acquisition.
    _PSF_METADATA_KEYS = (
        "RUN",
        "RERUN",
        "CAMCOL",
        "FIELD",
        "FILTER",
    )

    def __init__(
        self,
        config: ExperimentConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or get_logger(__name__)

    # ==================================================================
    # Public API
    # ==================================================================

    def acquire(self) -> HDUList:
        """
        Acquire and validate an SDSS imaging dataset.

        A valid cached dataset is reused when available.

        If the cached dataset does not contain the metadata required for
        downstream SDSS Observation construction, it is reacquired.

        Returns
        -------
        HDUList
            Validated SDSS FITS dataset.

        Raises
        ------
        AcquisitionError
            If acquisition, cutout creation, validation or storage fails.
        """
        self._validate_configuration()

        target = self._config.target

        right_ascension = float(
            cast(float, target.right_ascension)
        )
        declination = float(
            cast(float, target.declination)
        )

        cutout_size = self._config.acquisition.cutout_size

        storage = SDSSStorage(
            target_name=target.target_name,
            right_ascension=right_ascension,
            declination=declination,
            cutout_size=cutout_size,
        )

        self._logger.info(
            "SDSS acquisition configuration:"
        )
        self._logger.info(
            "  Target      : %s",
            target.target_name,
        )
        self._logger.info(
            "  RA          : %.6f°",
            right_ascension,
        )
        self._logger.info(
            "  Dec         : %.6f°",
            declination,
        )

        if cutout_size is None:
            self._logger.info(
                "  Image mode  : native/full SDSS image"
            )
        else:
            self._logger.info(
                "  Cutout size : %d pixels",
                cutout_size,
            )

        self._logger.info(
            "  Band        : %s",
            self._BAND,
        )
        self._logger.info(
            "  Data release: DR%d",
            self._DATA_RELEASE,
        )

        # --------------------------------------------------------------
        # Cached dataset
        # --------------------------------------------------------------

        if storage.exists():
            self._logger.info(
                "Loading cached SDSS dataset from '%s'.",
                storage.file_path,
            )

            try:
                dataset = storage.load()

                self._validate_dataset(dataset)

                if not self._has_required_psf_metadata(dataset):
                    raise AcquisitionError(
                        "Cached SDSS dataset does not contain the imaging "
                        "metadata required for native PSF acquisition."
                    )

            except Exception as exc:
                self._logger.warning(
                    "Cached SDSS dataset is unavailable, invalid, or "
                    "missing required PSF metadata: %s",
                    exc,
                )
                self._logger.info(
                    "Re-acquiring SDSS dataset from the archive."
                )
            else:
                self._logger.info(
                    "Cached SDSS dataset validated successfully."
                )
                return dataset

        # --------------------------------------------------------------
        # Archive acquisition
        # --------------------------------------------------------------

        try:
            catalogue = self._search_catalog()

            catalogue_entry = self._select_catalog_entry(
                catalogue,
            )

            dataset = self._download_dataset(
                catalogue_entry,
            )

            # Preserve the SDSS imaging identifiers before any spatial
            # transformation is performed.
            dataset = self._preserve_acquisition_metadata(
                dataset,
                catalogue_entry,
            )

            if cutout_size is None:
                self._logger.info(
                    "Preserving native SDSS image without spatial cutout."
                )
            else:
                dataset = self._create_cutout(
                    dataset,
                )

                # Cutout construction may replace WCS keywords. Reapply
                # the acquisition metadata afterwards so the final dataset
                # remains self-describing.
                dataset = self._preserve_acquisition_metadata(
                    dataset,
                    catalogue_entry,
                )

            self._validate_dataset(
                dataset,
            )

            self._logger.info(
                "Saving SDSS dataset to '%s'.",
                storage.file_path,
            )

            storage.save(
                dataset,
            )

            self._logger.info(
                "Successfully acquired SDSS dataset for '%s' "
                "(RA=%.6f°, Dec=%.6f°, shape=%s).",
                target.target_name,
                right_ascension,
                declination,
                self._image_shape(dataset),
            )

            return dataset

        except AcquisitionError:
            raise

        except Exception as exc:
            self._logger.exception(
                "Unexpected error during SDSS acquisition."
            )

            raise AcquisitionError(
                f"Failed to acquire SDSS dataset for "
                f"'{target.target_name}'."
            ) from exc

    # ==================================================================
    # Configuration validation
    # ==================================================================

    def _validate_configuration(self) -> None:
        """
        Validate the SDSS acquisition configuration.
        """
        target = self._config.target
        acquisition = self._config.acquisition

        if not target.target_name.strip():
            raise AcquisitionError(
                "SDSS target name is required."
            )

        if target.right_ascension is None:
            raise AcquisitionError(
                "SDSS target right ascension is required."
            )

        if target.declination is None:
            raise AcquisitionError(
                "SDSS target declination is required."
            )

        try:
            right_ascension = float(
                target.right_ascension
            )
            declination = float(
                target.declination
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise AcquisitionError(
                "SDSS target coordinates must be numeric."
            ) from exc

        if not np.isfinite(right_ascension):
            raise AcquisitionError(
                "SDSS target right ascension must be finite."
            )

        if not np.isfinite(declination):
            raise AcquisitionError(
                "SDSS target declination must be finite."
            )

        if not 0.0 <= right_ascension < 360.0:
            raise AcquisitionError(
                "SDSS target right ascension must be in the "
                "range [0, 360) degrees."
            )

        if not -90.0 <= declination <= 90.0:
            raise AcquisitionError(
                "SDSS target declination must be in the "
                "range [-90, 90] degrees."
            )

        cutout_size = acquisition.cutout_size

        if cutout_size is None:
            return

        if (
            not isinstance(cutout_size, int)
            or isinstance(cutout_size, bool)
        ):
            raise AcquisitionError(
                "SDSS cutout size must be an integer or None."
            )

        if cutout_size <= 0:
            raise AcquisitionError(
                "SDSS cutout size must be greater than zero "
                "when specified."
            )

    # ==================================================================
    # Catalogue search
    # ==================================================================

    def _search_catalog(self) -> Table:
        """
        Resolve the SDSS imaging source for the configured target.

        Normal targets use a preferred two-arcminute search followed by
        a bounded three-arcminute fallback.

        Explicitly registered crowded-field exceptions use a deterministic
        imaging-field descriptor.
        """
        target = self._config.target

        right_ascension = float(
            cast(float, target.right_ascension)
        )
        declination = float(
            cast(float, target.declination)
        )

        coordinates = SkyCoord(
            ra=right_ascension * u.deg,
            dec=declination * u.deg,
            frame="icrs",
        )

        self._logger.info(
            "Searching SDSS catalogue around RA=%.6f°, Dec=%.6f°.",
            right_ascension,
            declination,
        )

        search_radii_arcmin = (
            self._SEARCH_RADIUS_ARCMIN,
            self._FALLBACK_SEARCH_RADIUS_ARCMIN,
        )

        last_empty_radius_arcmin: float | None = None

        for index, radius_arcmin in enumerate(
            search_radii_arcmin
        ):
            radius = radius_arcmin * u.arcmin

            if index == 0:
                self._logger.info(
                    "Searching SDSS catalogue within %.1f arcmin.",
                    radius_arcmin,
                )
            else:
                self._logger.warning(
                    "No SDSS catalogue entries found within %.1f "
                    "arcmin; retrying within bounded fallback radius "
                    "%.1f arcmin.",
                    self._SEARCH_RADIUS_ARCMIN,
                    radius_arcmin,
                )

            try:
                raw_catalogue = SDSS.query_region(  # pyright: ignore[reportAttributeAccessIssue]
                    coordinates,
                    radius=radius,
                    data_release=self._DATA_RELEASE,
                )

            except Exception as exc:
                fallback = self._known_imaging_field_fallback()

                if fallback is not None:
                    self._logger.warning(
                        "SDSS catalogue query failed for target '%s'; "
                        "using documented imaging-field fallback: %s.",
                        target.target_name,
                        exc,
                    )
                    return fallback

                raise AcquisitionError(
                    "SDSS catalogue search failed within "
                    f"{radius_arcmin:.1f} arcmin."
                ) from exc

            catalogue = cast(
                Table | None,
                raw_catalogue,
            )

            if catalogue is None or len(catalogue) == 0:
                last_empty_radius_arcmin = radius_arcmin
                continue

            required_columns = {
                "ra",
                "dec",
            }

            missing_columns = (
                required_columns
                - set(catalogue.colnames)
            )

            if missing_columns:
                raise AcquisitionError(
                    "SDSS catalogue response is missing required "
                    f"columns: {', '.join(sorted(missing_columns))}."
                )

            self._logger.info(
                "Found %d SDSS catalogue entr%s within %.1f arcmin%s.",
                len(catalogue),
                "y" if len(catalogue) == 1 else "ies",
                radius_arcmin,
                "" if index == 0 else " fallback radius",
            )

            return catalogue

        # Explicit crowded-field fallback.
        fallback = self._known_imaging_field_fallback()

        if fallback is not None:
            self._logger.warning(
                "No SDSS catalogue entries found within %.1f arcmin; "
                "using documented imaging-field fallback for target '%s'.",
                last_empty_radius_arcmin
                or self._FALLBACK_SEARCH_RADIUS_ARCMIN,
                target.target_name,
            )
            return fallback

        maximum_radius = (
            last_empty_radius_arcmin
            or self._FALLBACK_SEARCH_RADIUS_ARCMIN
        )

        raise AcquisitionError(
            "No SDSS catalogue entries found within the maximum "
            f"supported search radius of {maximum_radius:.1f} arcmin."
        )

    def _known_imaging_field_fallback(self) -> Table | None:
        """
        Return a deterministic imaging-field descriptor for a known target.
        """
        target = self._config.target

        key = "".join(
            character
            for character in target.target_name.casefold()
            if character.isalnum()
        )

        fallback = self._KNOWN_IMAGING_FIELD_FALLBACKS.get(
            key
        )

        if fallback is None:
            return None

        run, rerun, camcol, field = fallback

        self._logger.info(
            "Resolved target '%s' to SDSS imaging field "
            "run=%d, rerun=%d, camcol=%d, field=%d.",
            target.target_name,
            run,
            rerun,
            camcol,
            field,
        )

        return Table(
            rows=[
                (
                    float(
                        cast(float, target.right_ascension)
                    ),
                    float(
                        cast(float, target.declination)
                    ),
                    run,
                    rerun,
                    camcol,
                    field,
                    True,
                )
            ],
            names=[
                "ra",
                "dec",
                "run",
                "rerun",
                "camcol",
                "field",
                self._FIELD_FALLBACK_MARKER,
            ],
        )

    # ==================================================================
    # Catalogue selection
    # ==================================================================

    def _select_catalog_entry(
        self,
        catalogue: Table,
    ) -> Table:
        """
        Select the catalogue entry nearest the configured target.
        """
        target = self._config.target

        right_ascension = float(
            cast(float, target.right_ascension)
        )
        declination = float(
            cast(float, target.declination)
        )

        try:
            target_coordinates = SkyCoord(
                ra=right_ascension * u.deg,
                dec=declination * u.deg,
                frame="icrs",
            )

            catalogue_ra = np.asarray(
                catalogue["ra"],
                dtype=float,
            )

            catalogue_dec = np.asarray(
                catalogue["dec"],
                dtype=float,
            )

            if catalogue_ra.size == 0:
                raise ValueError(
                    "Catalogue contains no right ascension values."
                )

            if catalogue_dec.size != catalogue_ra.size:
                raise ValueError(
                    "Catalogue right ascension and declination "
                    "columns have different lengths."
                )

            if not np.all(
                np.isfinite(catalogue_ra)
            ):
                raise ValueError(
                    "Catalogue contains non-finite right ascension values."
                )

            if not np.all(
                np.isfinite(catalogue_dec)
            ):
                raise ValueError(
                    "Catalogue contains non-finite declination values."
                )

            catalogue_coordinates = SkyCoord(
                ra=catalogue_ra * u.deg,
                dec=catalogue_dec * u.deg,
                frame="icrs",
            )

            separations = target_coordinates.separation(
                catalogue_coordinates,
            )

        except Exception as exc:
            raise AcquisitionError(
                "Unable to calculate SDSS catalogue "
                "source separations."
            ) from exc

        if len(separations) == 0:
            raise AcquisitionError(
                "SDSS catalogue contains no usable coordinates."
            )

        nearest_index = int(
            np.argmin(
                separations.to_value(
                    u.arcsec,
                )
            )
        )

        nearest_separation = cast(
            Angle,
            separations[nearest_index],
        )

        self._logger.info(
            "Selected nearest SDSS catalogue entry "
            "(%.2f arcsec from requested coordinates).",
            nearest_separation.arcsec,
        )

        return cast(
            Table,
            catalogue[
                nearest_index : nearest_index + 1
            ],
        )

    # ==================================================================
    # FITS acquisition
    # ==================================================================

    def _download_dataset(
        self,
        catalogue_entry: Table,
    ) -> HDUList:
        """
        Download the SDSS image corresponding to a selected entry.
        """
        self._logger.info(
            "Downloading SDSS %s-band FITS image.",
            self._BAND,
        )

        try:
            entry: Any = catalogue_entry

            use_field_fallback = (
                self._FIELD_FALLBACK_MARKER
                in catalogue_entry.colnames
                and bool(
                    entry[
                        self._FIELD_FALLBACK_MARKER
                    ][0]
                )
            )

            if use_field_fallback:
                run = int(entry["run"][0])
                rerun = int(entry["rerun"][0])
                camcol = int(entry["camcol"][0])
                field = int(entry["field"][0])

                self._logger.info(
                    "Downloading SDSS %s-band image directly from "
                    "run=%d, rerun=%d, camcol=%d, field=%d.",
                    self._BAND,
                    run,
                    rerun,
                    camcol,
                    field,
                )

                fallback_matches = Table(
                    rows=[
                        (
                            run,
                            rerun,
                            camcol,
                            field,
                        )
                    ],
                    names=[
                        "run",
                        "rerun",
                        "camcol",
                        "field",
                    ],
                )

                raw_datasets = SDSS.get_images(
                    matches=fallback_matches,
                    band=self._BAND,
                    data_release=self._DATA_RELEASE,
                )

            else:
                raw_datasets = SDSS.get_images(
                    matches=catalogue_entry,
                    band=self._BAND,
                    data_release=self._DATA_RELEASE,
                )

        except Exception as exc:
            raise AcquisitionError(
                "Failed to download SDSS FITS image."
            ) from exc

        if raw_datasets is None:
            raise AcquisitionError(
                "SDSS image download returned no datasets."
            )

        datasets = cast(
            list[Any],
            raw_datasets,
        )

        if not datasets:
            raise AcquisitionError(
                "SDSS image download returned no datasets."
            )

        self._logger.info(
            "Downloaded %d SDSS FITS dataset%s.",
            len(datasets),
            "" if len(datasets) == 1 else "s",
        )

        dataset = cast(
            HDUList,
            datasets[0],
        )

        if len(dataset) == 0:
            raise AcquisitionError(
                "SDSS image download returned an empty FITS dataset."
            )

        self._logger.info(
            "Downloaded native SDSS image with shape %s.",
            self._image_shape(dataset),
        )

        return dataset

    # ==================================================================
    # Acquisition metadata preservation
    # ==================================================================

    @staticmethod
    def _table_value(
        table: Table,
        keyword: str,
    ) -> Any:
        """
        Safely retrieve a scalar value from a one-row Astropy table.
        """
        if keyword not in table.colnames:
            return None

        try:
            value = table[keyword][0]
        except (
            IndexError,
            KeyError,
            TypeError,
        ):
            return None

        return value

    def _preserve_acquisition_metadata(
        self,
        dataset: HDUList,
        catalogue_entry: Table,
    ) -> HDUList:
        """
        Preserve SDSS imaging identifiers on the final image HDU.

        This is important because the ObservationBuilder operates on the
        FITS dataset alone and must not depend on the temporary Astroquery
        catalogue object.

        Existing archive header values are retained when present.
        Catalogue/fallback values are used to fill missing values.
        """
        image_hdu = self._find_image_hdu(
            dataset,
        )

        header = self._image_header(
            image_hdu,
        )

        entry: Any = catalogue_entry

        fallback_values: dict[str, Any] = {}

        for keyword in (
            "RUN",
            "RERUN",
            "CAMCOL",
            "FIELD",
        ):
            value = self._table_value(
                catalogue_entry,
                keyword.lower(),
            )

            if value is None:
                value = self._table_value(
                    catalogue_entry,
                    keyword,
                )

            fallback_values[keyword] = value

        filter_value = self._table_value(
            catalogue_entry,
            "filter",
        )

        if filter_value is None:
            filter_value = self._table_value(
                catalogue_entry,
                "FILTER",
            )

        # --------------------------------------------------------------
        # Only fill missing values.
        #
        # Archive-provided values remain authoritative.
        # --------------------------------------------------------------

        for keyword, value in fallback_values.items():
            if header.get(keyword) is None and value is not None:
                try:
                    header[keyword] = (
                        int(value),
                        f"SDSS {keyword.lower()} identifier",
                    )
                except (
                    TypeError,
                    ValueError,
                    OverflowError,
                ) as exc:
                    raise AcquisitionError(
                        f"Invalid SDSS {keyword} value: {value!r}."
                    ) from exc

        if header.get("FILTER") is None and filter_value is not None:
            header["FILTER"] = (
                str(filter_value).strip().lower(),
                "SDSS imaging filter",
            )

        # Explicit project-level metadata.
        header["SDSSBAND"] = (
            self._BAND,
            "SDSS imaging band used by experiment",
        )

        header["SDSSDR"] = (
            self._DATA_RELEASE,
            "SDSS data release",
        )

        self._logger.debug(
            "Preserved SDSS acquisition metadata: "
            "RUN=%s RERUN=%s CAMCOL=%s FIELD=%s FILTER=%s.",
            header.get("RUN"),
            header.get("RERUN"),
            header.get("CAMCOL"),
            header.get("FIELD"),
            header.get("FILTER"),
        )

        return dataset

    def _has_required_psf_metadata(
        self,
        dataset: HDUList,
    ) -> bool:
        """
        Return whether the dataset contains all identifiers required for
        native SDSS PSF reconstruction.
        """
        try:
            image_hdu = self._find_image_hdu(
                dataset,
            )

            header = self._image_header(
                image_hdu,
            )

        except AcquisitionError:
            return False

        for keyword in self._PSF_METADATA_KEYS:
            value = header.get(keyword)

            if value is None:
                return False

            if isinstance(value, str) and not value.strip():
                return False

        return True

    # ==================================================================
    # Cutout
    # ==================================================================

    def _create_cutout(
        self,
        dataset: HDUList,
    ) -> HDUList:
        """
        Create a target-centred WCS-aware image cutout.
        """
        target = self._config.target
        cutout_size = self._config.acquisition.cutout_size

        if cutout_size is None:
            raise AcquisitionError(
                "Cannot create an SDSS cutout when cutout_size is None."
            )

        image_hdu = self._find_image_hdu(
            dataset,
        )

        image_data = self._image_data(
            image_hdu,
        )

        header = self._image_header(
            image_hdu,
        )

        try:
            wcs = WCS(
                header,
            )
        except Exception as exc:
            raise AcquisitionError(
                "Unable to construct WCS from the SDSS image header."
            ) from exc

        if not wcs.has_celestial:
            raise AcquisitionError(
                "SDSS image does not contain a celestial WCS."
            )

        position = SkyCoord(
            ra=float(
                cast(float, target.right_ascension)
            ) * u.deg,
            dec=float(
                cast(float, target.declination)
            ) * u.deg,
            frame="icrs",
        )

        self._logger.info(
            "Creating %d × %d pixel SDSS cutout "
            "centred on RA=%.6f°, Dec=%.6f°.",
            cutout_size,
            cutout_size,
            float(
                cast(float, target.right_ascension)
            ),
            float(
                cast(float, target.declination)
            ),
        )

        try:
            cutout = Cutout2D(
                image_data,
                position=position,
                size=(
                    cutout_size,
                    cutout_size,
                ),
                wcs=wcs,
                mode="strict",
                copy=True,
            )
        except Exception as exc:
            raise AcquisitionError(
                f"Unable to create a {cutout_size} × "
                f"{cutout_size} pixel SDSS cutout centred on "
                f"target '{target.target_name}'. The target may "
                f"fall outside the downloaded image footprint."
            ) from exc

        if cutout.wcs is None:
            raise AcquisitionError(
                "SDSS cutout does not contain a valid celestial WCS."
            )

        cutout_dataset = cast(
            HDUList,
            dataset.copy(),
        )

        target_hdu_index: int | None = None

        for index, hdu in enumerate(dataset):
            if hdu is image_hdu:
                target_hdu_index = index
                break

        if target_hdu_index is None:
            for index, hdu in enumerate(cutout_dataset):
                candidate = cast(
                    Any,
                    hdu,
                )

                candidate_data = getattr(
                    candidate,
                    "data",
                    None,
                )

                if (
                    candidate_data is not None
                    and np.ndim(candidate_data) == 2
                ):
                    target_hdu_index = index
                    break

        if target_hdu_index is None:
            raise AcquisitionError(
                "Unable to locate the SDSS image HDU in the "
                "copied FITS dataset."
            )

        output_hdu = cast(
            Any,
            cutout_dataset[target_hdu_index],
        )

        output_hdu.data = np.asarray(
            cutout.data,
        ).copy()

        output_hdu.header.update(
            cutout.wcs.to_header(),
        )

        output_hdu.header["CUTOUTSZ"] = (
            cutout_size,
            "Requested SDSS cutout size in pixels",
        )

        output_hdu.header["TARGRA"] = (
            float(
                cast(float, target.right_ascension)
            ),
            "Configured target RA, degrees",
        )

        output_hdu.header["TARGDEC"] = (
            float(
                cast(float, target.declination)
            ),
            "Configured target Dec, degrees",
        )

        output_hdu.header["SDSSBAND"] = (
            self._BAND,
            "SDSS imaging band",
        )

        output_hdu.header["SDSSDR"] = (
            self._DATA_RELEASE,
            "SDSS data release",
        )

        self._logger.info(
            "Created SDSS cutout with shape %s.",
            cutout.data.shape,
        )

        return cutout_dataset

    # ==================================================================
    # FITS image helpers
    # ==================================================================

    @staticmethod
    def _find_image_hdu(
        dataset: HDUList,
    ) -> Any:
        """
        Locate the first suitable two-dimensional image HDU.
        """
        if dataset is None or len(dataset) == 0:
            raise AcquisitionError(
                "SDSS FITS dataset contains no HDUs."
            )

        for hdu in dataset:
            candidate = cast(
                Any,
                hdu,
            )

            data = getattr(
                candidate,
                "data",
                None,
            )

            if data is None:
                continue

            try:
                dimensions = np.ndim(data)
            except Exception:
                continue

            if dimensions != 2:
                continue

            shape = getattr(
                data,
                "shape",
                None,
            )

            if shape is None or len(shape) != 2:
                continue

            if (
                shape[0] <= 0
                or shape[1] <= 0
            ):
                continue

            return candidate

        raise AcquisitionError(
            "SDSS FITS dataset does not contain a usable "
            "two-dimensional image HDU."
        )

    @staticmethod
    def _image_data(
        image_hdu: Any,
    ) -> np.ndarray:
        """
        Extract and validate image data from an image HDU.
        """
        data = getattr(
            image_hdu,
            "data",
            None,
        )

        if data is None:
            raise AcquisitionError(
                "SDSS image HDU does not contain image data."
            )

        try:
            image_data = np.asarray(
                data,
            )
        except Exception as exc:
            raise AcquisitionError(
                "Unable to convert SDSS image data to a NumPy array."
            ) from exc

        if image_data.ndim != 2:
            raise AcquisitionError(
                "SDSS image data must be two-dimensional."
            )

        if (
            image_data.shape[0] <= 0
            or image_data.shape[1] <= 0
        ):
            raise AcquisitionError(
                "SDSS image has an invalid spatial shape."
            )

        return image_data

    @staticmethod
    def _image_header(
        image_hdu: Any,
    ) -> Any:
        """
        Return the FITS header associated with an image HDU.
        """
        header = getattr(
            image_hdu,
            "header",
            None,
        )

        if header is None:
            raise AcquisitionError(
                "SDSS image HDU does not contain a FITS header."
            )

        return header

    # ==================================================================
    # Dataset validation
    # ==================================================================

    def _validate_dataset(
        self,
        dataset: HDUList,
    ) -> None:
        """
        Validate the final SDSS FITS dataset.

        Validation is structural rather than scientific.
        """
        if dataset is None:
            raise AcquisitionError(
                "SDSS dataset cannot be None."
            )

        if len(dataset) == 0:
            raise AcquisitionError(
                "SDSS FITS dataset contains no HDUs."
            )

        image_hdu = self._find_image_hdu(
            dataset,
        )

        image_data = self._image_data(
            image_hdu,
        )

        expected_size = self._config.acquisition.cutout_size

        if expected_size is not None:
            if image_data.shape != (
                expected_size,
                expected_size,
            ):
                raise AcquisitionError(
                    "SDSS image shape does not match the configured "
                    f"cutout size of {expected_size} pixels: "
                    f"received {image_data.shape}."
                )

        if not np.any(
            np.isfinite(image_data)
        ):
            raise AcquisitionError(
                "SDSS image contains no finite pixel values."
            )

        header = self._image_header(
            image_hdu,
        )

        try:
            wcs = WCS(
                header,
            )
        except Exception as exc:
            raise AcquisitionError(
                "Unable to read SDSS image WCS."
            ) from exc

        if not wcs.has_celestial:
            raise AcquisitionError(
                "SDSS image does not contain a valid celestial WCS."
            )

        self._logger.info(
            "SDSS dataset validated successfully "
            "(%d HDU%s, image shape=%s).",
            len(dataset),
            "" if len(dataset) == 1 else "s",
            image_data.shape,
        )

    @staticmethod
    def _image_shape(
        dataset: HDUList,
    ) -> tuple[int, int]:
        """
        Return the shape of the first two-dimensional image HDU.
        """
        for hdu in dataset:
            candidate = cast(
                Any,
                hdu,
            )

            data = getattr(
                candidate,
                "data",
                None,
            )

            if data is None:
                continue

            if np.ndim(data) != 2:
                continue

            shape = getattr(
                data,
                "shape",
                None,
            )

            if shape is None or len(shape) != 2:
                continue

            return (
                int(shape[0]),
                int(shape[1]),
            )

        raise AcquisitionError(
            "SDSS dataset contains no two-dimensional image."
        )