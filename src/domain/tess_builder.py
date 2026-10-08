"""
TESS Observation Builder.

Converts Lightkurve Target Pixel File objects into the project's internal
Observation domain model.

This builder forms the boundary between the external Lightkurve acquisition
library and the remainder of the photometric analysis pipeline.

The builder performs structural translation only. It does not perform
background subtraction, source detection, photometry, or evaluation.

Two construction modes are supported:

    build()
        Builds a single Observation from the first valid cadence. This
        preserves the behaviour of the existing single-image workflow.

    build_sequence()
        Builds a tuple of independent Observation objects, one per selected
        valid TESS cadence. This provides the foundation for time-series
        photometric analysis without changing the existing single-image
        Observation model.

The TESS pipeline aperture is deliberately not used as an NDData mask.
It represents the TESS pipeline's photometric aperture rather than invalid
image pixels. The complete acquired image remains available to the
independent background-estimation and source-detection stages.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from os import PathLike
from typing import Any, Sequence

import numpy as np
from astropy.nddata import NDData, StdDevUncertainty
from lightkurve import TessTargetPixelFile

from core.exceptions import ObservationError
from core.models import TargetConfig
from acquisition.tess_psf import (
    TESSPSFAcquirer,
    TESSPSFError,
)
from domain.builders import ObservationBuilder
from domain.enums import ObservationSource, Survey
from domain.models import (
    ImageData,
    Observation,
    ObservationMetadata,
    ObservationProvenance,
    PSFData,
)
from utils.logger import get_logger


class TESSObservationBuilder(
    ObservationBuilder[TessTargetPixelFile],
):
    """
    Build survey-independent Observations from a TESS TPF.

    The existing ``build()`` method preserves the current single-image
    behaviour by using the first valid cadence.

    ``build_sequence()`` provides a time-series construction path by
    converting selected valid cadences into independent Observation
    objects.

    No scientific processing is performed by this builder.
    """

    def __init__(
        self,
        target: TargetConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the TESS Observation builder.

        Parameters
        ----------
        target
            Configured astronomical target.

        logger
            Optional logger.
        """

        super().__init__(
            logger,
        )

        self._target = target

        # The mission PRF is a sector/camera/CCD calibration product and is
        # independent of cadence. Cache it so a TESS time-series does not
        # download/reconstruct the same four PRF products for every cadence.
        self._psf_cache: dict[tuple[int, int, int, float, float], PSFData] = {}

    # ========================================================================
    # Existing single-observation API
    # ========================================================================

    def build(
        self,
        dataset: TessTargetPixelFile,
    ) -> Observation:
        """
        Convert a TESS Target Pixel File into one Observation.

        The first valid cadence is used. This preserves the existing
        single-image workflow behaviour.

        Parameters
        ----------
        dataset
            Acquired and validated TESS Target Pixel File.

        Returns
        -------
        Observation
            Survey-independent Observation containing the first valid
            TESS cadence.

        Raises
        ------
        ObservationError
            If the TPF cannot be converted into a valid Observation.
        """

        self._logger.info(
            "Building single Observation from TESS Target Pixel File."
        )

        self._validate_dataset(
            dataset,
        )

        cadence_indices = self._valid_cadence_indices(
            dataset,
        )

        if not cadence_indices:
            raise ObservationError(
                "TESS dataset contains no usable finite cadences."
            )

        first_cadence_index = cadence_indices[0]

        observation = self._build_observation(
            dataset=dataset,
            cadence_index=first_cadence_index,
        )

        image_shape = np.asarray(
            observation.images["raw"].image.data,
        ).shape

        self._logger.info(
            "Constructed single Observation for target '%s'.",
            observation.metadata.target_name,
        )

        self._logger.info(
            "TESS cadence index: %d.",
            first_cadence_index,
        )

        self._logger.info(
            "TESS image shape: %s.",
            image_shape,
        )

        self._logger.info(
            "TESS observation time: %s %s (%s).",
            observation.metadata.observation_time,
            observation.metadata.observation_time_format,
            observation.metadata.observation_time_scale,
        )

        self._log_dataset_context(
            dataset,
        )

        return observation

    # ========================================================================
    # New time-series API
    # ========================================================================

    def build_sequence(
        self,
        dataset: TessTargetPixelFile,
        *,
        cadence_indices: Sequence[int] | None = None,
    ) -> tuple[Observation, ...]:
        """
        Convert selected TESS cadences into independent Observations.

        This method does not alter the existing ``build()`` behaviour.
        Every returned Observation represents one TESS cadence and carries
        the cadence-specific image, uncertainty and observation time.

        Parameters
        ----------
        dataset
            Acquired and validated TESS Target Pixel File.

        cadence_indices
            Optional sequence of zero-based cadence indices to process.
            When omitted, every usable finite cadence in the TPF is used.

        Returns
        -------
        tuple[Observation, ...]
            Immutable tuple containing one Observation per selected cadence.

        Raises
        ------
        ObservationError
            If the TPF is invalid, no usable cadence exists, a cadence index
            is invalid, or a selected cadence cannot be converted.

        Notes
        -----
        This method deliberately returns ordinary Observation objects rather
        than introducing a second image-processing domain model. The existing
        preprocessing, source-detection and photometry components can
        therefore be applied to every returned Observation without
        modification.
        """

        self._logger.info(
            "Building TESS Observation sequence."
        )

        self._validate_dataset(
            dataset,
        )

        valid_indices = self._valid_cadence_indices(
            dataset,
        )

        if not valid_indices:
            raise ObservationError(
                "TESS dataset contains no usable finite cadences."
            )

        selected_indices = self._select_cadence_indices(
            valid_indices=valid_indices,
            requested_indices=cadence_indices,
        )

        self._logger.info(
            "TESS usable cadences: %d.",
            len(valid_indices),
        )

        self._logger.info(
            "TESS selected cadences: %d.",
            len(selected_indices),
        )

        observations: list[Observation] = []

        for position, cadence_index in enumerate(
            selected_indices,
            start=1,
        ):
            try:
                observation = self._build_observation(
                    dataset=dataset,
                    cadence_index=cadence_index,
                )

            except ObservationError:
                raise

            except Exception as exc:
                self._logger.exception(
                    (
                        "Failed to construct TESS Observation for "
                        "cadence index %d."
                    ),
                    cadence_index,
                )

                raise ObservationError(
                    "Failed to construct TESS Observation for "
                    f"cadence index {cadence_index}."
                ) from exc

            observations.append(
                observation,
            )

            self._logger.debug(
                (
                    "Constructed TESS cadence %d/%d "
                    "(index=%d, time=%s)."
                ),
                position,
                len(selected_indices),
                cadence_index,
                observation.metadata.observation_time,
            )

        if not observations:
            raise ObservationError(
                "TESS Observation sequence contains no observations."
            )

        self._logger.info(
            "Constructed %d TESS Observations for target '%s'.",
            len(observations),
            self._target.target_name,
        )

        return tuple(
            observations,
        )

    # ========================================================================
    # Common observation construction
    # ========================================================================

    def _build_observation(
        self,
        *,
        dataset: TessTargetPixelFile,
        cadence_index: int,
    ) -> Observation:
        """
        Build one Observation from one TESS cadence.

        Parameters
        ----------
        dataset
            TESS Target Pixel File.

        cadence_index
            Zero-based cadence index.

        Returns
        -------
        Observation
            Survey-independent single-cadence Observation.
        """

        self._validate_cadence_index(
            dataset=dataset,
            cadence_index=cadence_index,
        )

        try:
            metadata = self._extract_metadata(
                dataset=dataset,
                cadence_index=cadence_index,
            )

            provenance = self._extract_provenance(
                dataset=dataset,
                cadence_index=cadence_index,
            )

            raw_image = ImageData(
                self._create_image(
                    dataset=dataset,
                    cadence_index=cadence_index,
                ),
            )

        except ObservationError:
            raise

        except Exception as exc:
            self._logger.exception(
                (
                    "Failed to construct Observation from TESS "
                    "cadence index %d."
                ),
                cadence_index,
            )

            raise ObservationError(
                "Failed to construct Observation from TESS "
                f"cadence index {cadence_index}."
            ) from exc

        psf = self._try_acquire_psf(
            dataset,
        )

        return Observation(
            metadata=metadata,
            provenance=provenance,
            original=raw_image,
            active=raw_image,
            psf=psf,
            images={
                "raw": raw_image,
            },
        )

    # ========================================================================
    # Native TESS PRF acquisition
    # ========================================================================

    def _try_acquire_psf(
        self,
        dataset: TessTargetPixelFile,
    ) -> PSFData | None:
        """Attempt to attach the mission-provided TESS PRF to the Observation.

        Failure is deliberately non-fatal, matching the SDSS PSF architecture:
        baseline photometry remains independent of the PSF, while the SCARLET
        preprocessing contract is responsible for requiring it.
        """

        try:
            sector = int(dataset.sector)
            camera = int(dataset.camera)
            ccd = int(dataset.ccd)

            row, col = self._target_detector_position(
                dataset,
            )

            cache_key = (
                sector,
                camera,
                ccd,
                round(col, 9),
                round(row, 9),
            )

            cached = self._psf_cache.get(cache_key)
            if cached is not None:
                self._logger.debug(
                    "Reusing cached TESS PSF for sector=%d, camera=%d, "
                    "ccd=%d at (row=%.6f, col=%.6f).",
                    sector,
                    camera,
                    ccd,
                    row,
                    col,
                )
                return cached

            self._logger.info(
                "Acquiring native TESS PRF at "
                "(row=%.6f, col=%.6f; sector=%d, camera=%d, ccd=%d).",
                row,
                col,
                sector,
                camera,
                ccd,
            )

            acquirer = TESSPSFAcquirer(
                sector=sector,
                camera=camera,
                ccd=ccd,
            )

            tess_psf = acquirer.acquire(
                row=row,
                col=col,
            )

            source_paths = tuple(
                str(path)
                for path in tess_psf.source_paths
            )

            psf = PSFData(
                data=np.array(
                    tess_psf.data,
                    dtype=np.float64,
                    copy=True,
                ),
                model_type="TESS_PRF_SPOC",
                normalised=True,
                x=col,
                y=row,
                pixel_scale_arcsec=tess_psf.pixel_scale_arcsec,
                source_path=(
                    source_paths[0]
                    if len(source_paths) == 1
                    else ";".join(source_paths)
                ),
                metadata={
                    "sector": sector,
                    "camera": camera,
                    "ccd": ccd,
                    "detector_row": row,
                    "detector_col": col,
                    "source_paths": source_paths,
                    "calibration": "MAST TESS mission PRF",
                },
            )

            self._psf_cache[cache_key] = psf

            self._logger.info(
                "TESS PRF acquired successfully "
                "(shape=%s, total_flux=%.12f).",
                psf.shape,
                psf.total_flux,
            )

            return psf

        except (
            ObservationError,
            TESSPSFError,
        ) as exc:
            self._logger.warning(
                "Native TESS PRF unavailable for target '%s': %s",
                self._target.target_name,
                exc,
            )
            return None

        except Exception as exc:
            self._logger.warning(
                "Unexpected error while acquiring native TESS PRF "
                "for target '%s': %s",
                self._target.target_name,
                exc,
            )
            return None

    def _target_detector_position(
        self,
        dataset: TessTargetPixelFile,
    ) -> tuple[float, float]:
        """Resolve the target into the full-CCD coordinates used by TESS PRFs.

        TESSCut/TPF products expose the lower-left CCD pixel through
        ``dataset.column``/``dataset.row``.  Their aperture header also
        contains ``CRPIX1``/``CRPIX2`` describing the target/reference pixel
        within the cutout.  This is the same convention documented by the
        Lightkurve project for TESSCut products and avoids a one-pixel
        ambiguity between FITS (1-based) and Astropy (0-based) WCS coordinates.

        The WCS path remains the fallback for products which do not carry the
        aperture reference-pixel keywords.
        """

        try:
            column = float(dataset.column)
            row = float(dataset.row)
        except (TypeError, ValueError) as exc:
            raise ObservationError(
                "TESS dataset does not expose valid CCD column/row origin "
                "metadata for PRF positioning."
            ) from exc

        # Preferred path: TESS TPF/TESSCut aperture reference pixel.
        try:
            aperture_header = dataset.get_header(
                ext=2,
            )
            crpix1 = aperture_header.get("CRPIX1")
            crpix2 = aperture_header.get("CRPIX2")

            if crpix1 is not None and crpix2 is not None:
                detector_col = column + float(crpix1)
                detector_row = row + float(crpix2)

                if np.isfinite(detector_col) and np.isfinite(detector_row):
                    return detector_row, detector_col

        except Exception:
            # Fall through to the celestial WCS path.
            pass

        # Fallback: resolve the configured target sky position using the TPF
        # WCS.  Astropy's high-level WCS coordinates are zero-based, whereas
        # the mission PRF detector grid uses the FITS-style CCD coordinate
        # convention, hence the explicit +1 here.
        ra = (
            self._target.right_ascension
            if self._target.right_ascension is not None
            else self._finite_float(dataset.ra)
        )
        dec = (
            self._target.declination
            if self._target.declination is not None
            else self._finite_float(dataset.dec)
        )

        if ra is None or dec is None:
            raise ObservationError(
                "TESS PRF acquisition requires finite target RA and Dec."
            )

        try:
            wcs = dataset.wcs
            if wcs is None or not wcs.has_celestial:
                raise ObservationError(
                    "TESS dataset does not contain a valid celestial WCS "
                    "for PRF positioning."
                )

            pixel = wcs.world_to_pixel_values(
                float(ra),
                float(dec),
            )

            if len(pixel) < 2:
                raise ObservationError(
                    "TESS WCS returned fewer than two pixel coordinates."
                )

            local_col = float(np.asarray(pixel[0]))
            local_row = float(np.asarray(pixel[1]))

        except ObservationError:
            raise
        except Exception as exc:
            raise ObservationError(
                "Unable to transform the TESS target sky coordinate into "
                "detector pixel coordinates for PRF acquisition."
            ) from exc

        detector_row = row + local_row + 1.0
        detector_col = column + local_col + 1.0

        if not np.isfinite(detector_row) or not np.isfinite(detector_col):
            raise ObservationError(
                "TESS PRF detector coordinates are non-finite: "
                f"row={detector_row!r}, col={detector_col!r}."
            )

        return detector_row, detector_col

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            result = float(value)
        except (TypeError, ValueError):
            return None

        return result if np.isfinite(result) else None

    # ========================================================================
    # Dataset validation
    # ========================================================================

    def _validate_dataset(
        self,
        dataset: TessTargetPixelFile,
    ) -> None:
        """
        Validate the TESS dataset required to build Observations.

        Parameters
        ----------
        dataset
            TESS Target Pixel File.

        Raises
        ------
        ObservationError
            If required TPF data is unavailable.
        """

        if dataset is None:
            raise ObservationError(
                "TESS dataset cannot be None."
            )

        if dataset.flux is None:
            raise ObservationError(
                "TESS dataset contains no flux cube."
            )

        flux = np.asarray(
            dataset.flux,
        )

        if flux.ndim != 3:
            raise ObservationError(
                "Expected a three-dimensional TESS flux cube."
            )

        if flux.shape[0] == 0:
            raise ObservationError(
                "TESS dataset contains no cadences."
            )

        if (
            flux.shape[1] == 0
            or flux.shape[2] == 0
        ):
            raise ObservationError(
                "TESS flux cube contains an empty spatial footprint."
            )

        try:
            time = dataset.time
        except Exception as exc:
            raise ObservationError(
                "Unable to access TESS observation times."
            ) from exc

        if time is None:
            raise ObservationError(
                "TESS dataset contains no observation times."
            )

        try:
            time_length = len(time)
        except TypeError as exc:
            raise ObservationError(
                "TESS observation time array is not indexable."
            ) from exc

        if time_length == 0:
            raise ObservationError(
                "TESS dataset contains no observation times."
            )

        if time_length != flux.shape[0]:
            raise ObservationError(
                "TESS time and flux cadence counts do not match."
            )

        try:
            wcs = dataset.wcs
        except Exception as exc:
            raise ObservationError(
                "Unable to access TESS dataset WCS."
            ) from exc

        if wcs is None:
            raise ObservationError(
                "TESS dataset does not contain WCS information."
            )

        if not getattr(
            wcs,
            "has_celestial",
            False,
        ):
            raise ObservationError(
                "TESS dataset does not contain a valid celestial WCS."
            )

        try:
            dataset_ra = dataset.ra
            dataset_dec = dataset.dec

        except Exception as exc:
            raise ObservationError(
                "Unable to access TESS target coordinates."
            ) from exc

        if (
            self._target.right_ascension is None
            and dataset_ra is None
        ):
            raise ObservationError(
                f"Target '{self._target.target_name}' has no "
                "right ascension in either the experiment "
                "configuration or the TESS dataset."
            )

        if (
            self._target.declination is None
            and dataset_dec is None
        ):
            raise ObservationError(
                f"Target '{self._target.target_name}' has no "
                "declination in either the experiment "
                "configuration or the TESS dataset."
            )

    def _validate_cadence_index(
        self,
        *,
        dataset: TessTargetPixelFile,
        cadence_index: int,
    ) -> None:
        """
        Validate one zero-based cadence index.
        """

        if isinstance(
            cadence_index,
            bool,
        ) or not isinstance(
            cadence_index,
            int,
        ):
            raise ObservationError(
                "TESS cadence index must be an integer."
            )

        flux = np.asarray(
            dataset.flux,
        )

        cadence_count = flux.shape[0]

        if not (
            0
            <= cadence_index
            < cadence_count
        ):
            raise ObservationError(
                "TESS cadence index "
                f"{cadence_index} is outside the valid range "
                f"[0, {cadence_count - 1}]."
            )

    # ========================================================================
    # Cadence selection
    # ========================================================================

    @staticmethod
    def _valid_cadence_indices(
        dataset: TessTargetPixelFile,
    ) -> tuple[int, ...]:
        """
        Return indices of cadences containing finite time and image data.

        A cadence is considered structurally usable when:

        * the corresponding time value is finite; and
        * at least one pixel in the cadence contains a finite flux value.

        This is a structural criterion only. It does not perform scientific
        quality filtering or outlier rejection.
        """

        flux = np.asarray(
            dataset.flux,
            dtype=np.float64,
        )

        time = dataset.time

        raw_time_values: Any = getattr(
            time,
            "value",
            None,
        )

        if raw_time_values is None:
            raise ObservationError(
                "Unable to obtain numerical TESS cadence times."
            )

        time_values = np.asarray(
            raw_time_values,
            dtype=np.float64,
        )

        if (
            time_values.ndim != 1
            or len(time_values) != flux.shape[0]
        ):
            raise ObservationError(
                "TESS cadence time values do not match the "
                "flux-cube cadence dimension."
            )

        valid_indices: list[int] = []

        for index in range(
            flux.shape[0]
        ):
            cadence_image = flux[
                index
            ]

            cadence_time = time_values[
                index
            ]

            if not np.isfinite(
                cadence_time,
            ):
                continue

            if not np.isfinite(
                cadence_image,
            ).any():
                continue

            valid_indices.append(
                index,
            )

        return tuple(
            valid_indices,
        )

    def _select_cadence_indices(
        self,
        *,
        valid_indices: tuple[int, ...],
        requested_indices: Sequence[int] | None,
    ) -> tuple[int, ...]:
        """
        Validate and select cadence indices.

        When ``requested_indices`` is omitted, all valid cadence indices
        are returned.

        Requested indices must:

        * be integers;
        * refer to valid cadences;
        * contain no duplicates.

        The caller therefore has explicit control over which cadences
        enter the time-series workflow.
        """

        if requested_indices is None:
            return valid_indices

        valid_index_set = set(
            valid_indices,
        )

        selected: list[int] = []

        for cadence_index in requested_indices:
            if isinstance(
                cadence_index,
                bool,
            ) or not isinstance(
                cadence_index,
                int,
            ):
                raise ObservationError(
                    "TESS cadence indices must all be integers."
                )

            if cadence_index not in valid_index_set:
                raise ObservationError(
                    "Requested TESS cadence index "
                    f"{cadence_index} is not a usable cadence."
                )

            if cadence_index in selected:
                raise ObservationError(
                    "Duplicate TESS cadence index requested: "
                    f"{cadence_index}."
                )

            selected.append(
                cadence_index,
            )

        if not selected:
            raise ObservationError(
                "No TESS cadence indices were requested."
            )

        return tuple(
            selected,
        )

    # ========================================================================
    # Metadata extraction
    # ========================================================================

    def _extract_metadata(
        self,
        *,
        dataset: TessTargetPixelFile,
        cadence_index: int,
    ) -> ObservationMetadata:
        """
        Extract survey-independent observation metadata for one cadence.

        Configured target coordinates take precedence over coordinates
        supplied by the acquired TPF.
        """

        try:
            header = dataset.get_header(
                ext=1,
            )

        except Exception as exc:
            raise ObservationError(
                "Unable to read TESS pixel-extension metadata."
            ) from exc

        right_ascension = (
            self._target.right_ascension
            if self._target.right_ascension is not None
            else self._finite_float(
                dataset.ra,
            )
        )

        declination = (
            self._target.declination
            if self._target.declination is not None
            else self._finite_float(
                dataset.dec,
            )
        )

        if right_ascension is None:
            raise ObservationError(
                f"Unable to determine right ascension for "
                f"target '{self._target.target_name}'."
            )

        if declination is None:
            raise ObservationError(
                f"Unable to determine declination for "
                f"target '{self._target.target_name}'."
            )

        try:
            wcs = dataset.wcs

        except Exception as exc:
            raise ObservationError(
                "Unable to obtain TESS WCS."
            ) from exc

        if wcs is None or not wcs.has_celestial:
            raise ObservationError(
                "TESS dataset does not contain a valid celestial WCS."
            )

        observation_time = self._extract_observation_time(
            dataset=dataset,
            cadence_index=cadence_index,
        )

        exposure_time = self._extract_exposure_time(
            header,
        )

        instrument = self._optional_string(
            header.get(
                "INSTRUME",
            ),
        )

        filter_name = self._optional_string(
            header.get(
                "FILTER",
            ),
        )

        return ObservationMetadata(
            target_name=self._target.target_name,
            right_ascension=right_ascension,
            declination=declination,
            instrument=instrument,
            filter_name=filter_name,
            observation_time=observation_time,
            observation_time_scale="tdb",
            observation_time_format="btjd",
            exposure_time=exposure_time,
            wcs=wcs,
        )

    def _extract_provenance(
        self,
        *,
        dataset: TessTargetPixelFile,
        cadence_index: int,
    ) -> ObservationProvenance:
        """
        Extract acquisition provenance for one TESS cadence.
        """

        filename: str | None = None

        if isinstance(
            dataset.path,
            (
                str,
                PathLike,
            ),
        ):
            filename = str(
                dataset.path,
            )

        sector = dataset.sector

        dataset_identifier = (
            f"TESS Sector {sector}"
            if sector is not None
            else "TESS Sector Unknown"
        )

        return ObservationProvenance(
            source=ObservationSource.SPACE_BASED,
            survey=Survey.TESS,
            dataset_identifier=dataset_identifier,
            filename=filename,
            download_timestamp=datetime.now(
                timezone.utc,
            ),
        )

    # ========================================================================
    # Image construction
    # ========================================================================

    def _create_image(
        self,
        *,
        dataset: TessTargetPixelFile,
        cadence_index: int,
    ) -> NDData:
        """
        Create the 2-D image for one TESS cadence.

        The image preserves:

        * TESS flux values;
        * matching cadence-specific flux uncertainty;
        * WCS;
        * TPF metadata.

        The TESS pipeline aperture is not assigned to ``NDData.mask``.
        """

        flux = np.asarray(
            dataset.flux,
            dtype=np.float64,
        )

        if flux.ndim != 3:
            raise ObservationError(
                "Cannot create TESS image from a non-3-D flux cube."
            )

        if not (
            0
            <= cadence_index
            < flux.shape[0]
        ):
            raise ObservationError(
                "Requested TESS cadence index is outside the flux cube."
            )

        image_data = np.array(
            flux[cadence_index],
            dtype=np.float64,
            copy=True,
        )

        if image_data.ndim != 2:
            raise ObservationError(
                "TESS cadence image is not two-dimensional."
            )

        if not np.isfinite(
            image_data,
        ).any():
            raise ObservationError(
                "TESS cadence contains no finite image pixels."
            )

        uncertainty = self._create_uncertainty(
            dataset=dataset,
            cadence_index=cadence_index,
        )

        metadata = self._image_metadata(
            dataset=dataset,
            cadence_index=cadence_index,
        )

        return NDData(
            data=image_data,
            uncertainty=uncertainty,
            mask=None,
            wcs=dataset.wcs,
            meta=metadata,
            copy=False,
        )

    @staticmethod
    def _create_uncertainty(
        *,
        dataset: TessTargetPixelFile,
        cadence_index: int,
    ) -> Any:
        """
        Extract the flux uncertainty for one TESS cadence.

        Returns
        -------
        Any
            Two-dimensional uncertainty array, or ``None`` when no valid
            uncertainty cube is available.
        """

        try:
            flux_err = dataset.flux_err

        except Exception:
            return None

        if flux_err is None:
            return None

        flux_err_array = np.asarray(
            flux_err,
            dtype=np.float64,
        )

        if flux_err_array.ndim != 3:
            return None

        if not (
            0
            <= cadence_index
            < flux_err_array.shape[0]
        ):
            return None

        uncertainty = np.array(
            flux_err_array[cadence_index],
            dtype=np.float64,
            copy=True,
        )

        if uncertainty.ndim != 2:
            return None

        return StdDevUncertainty(uncertainty)

    # ========================================================================
    # Time extraction
    # ========================================================================

    @staticmethod
    def _extract_observation_time(
        *,
        dataset: TessTargetPixelFile,
        cadence_index: int,
    ) -> float | None:
        """
        Extract one TESS cadence time.

        The external Lightkurve/Astropy time object is converted to a plain
        Python float before entering the survey-independent domain model.

        The accompanying ObservationMetadata fields identify the value as
        BTJD on the TDB time scale.
        """

        try:
            time = dataset.time

            raw_values: Any = getattr(
                time,
                "value",
                None,
            )

            if raw_values is None:
                return None

            values = np.asarray(
                raw_values,
                dtype=np.float64,
            )

            if values.ndim != 1:
                return None

            if not (
                0
                <= cadence_index
                < len(values)
            ):
                return None

            result = float(
                values[cadence_index],
            )

            if not np.isfinite(
                result,
            ):
                return None

            return result

        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
        ):
            return None

    # ========================================================================
    # Metadata helpers
    # ========================================================================

    @staticmethod
    def _extract_exposure_time(
        header: Any,
    ) -> float | None:
        """
        Extract TESS exposure duration from the FITS header.
        """

        for keyword in (
            "EXPOSURE",
            "TELAPSE",
        ):
            value = header.get(
                keyword,
            )

            if value is None:
                continue

            try:
                result = float(
                    value,
                )

                if np.isfinite(
                    result,
                ):
                    return result

            except (
                TypeError,
                ValueError,
            ):
                continue

        return None

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        """
        Convert optional metadata into a non-empty string.
        """

        if value is None:
            return None

        result = str(
            value,
        ).strip()

        return result or None

    @staticmethod
    def _image_metadata(
        *,
        dataset: TessTargetPixelFile,
        cadence_index: int,
    ) -> dict[str, Any]:
        """
        Create metadata associated with one raw TESS cadence image.
        """

        metadata: dict[str, Any] = {}

        try:
            metadata.update(
                dict(
                    dataset.meta,
                )
            )

        except Exception:
            pass

        metadata.update(
            {
                "survey": "TESS",
                "processing_product": "raw",
                "sector": dataset.sector,
                "camera": dataset.camera,
                "ccd": dataset.ccd,
                "cadence_index": cadence_index,
            }
        )

        return metadata

    # ========================================================================
    # Logging
    # ========================================================================

    def _log_dataset_context(
        self,
        dataset: TessTargetPixelFile,
    ) -> None:
        """
        Log common TESS dataset context.
        """

        self._logger.info(
            "TESS sector: %s.",
            dataset.sector
            if dataset.sector is not None
            else "Unknown",
        )

        try:
            flux = np.asarray(
                dataset.flux,
            )

            self._logger.info(
                "TESS cadence count: %d.",
                flux.shape[0],
            )

        except Exception:
            pass