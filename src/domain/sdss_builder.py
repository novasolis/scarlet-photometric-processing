"""
SDSS Observation Builder.

Converts an SDSS FITS HDUList into the project's survey-independent
Observation domain model.

Responsibilities
----------------
1. Validate the acquired SDSS science image.
2. Extract observation metadata and SDSS provenance.
3. Preserve the original SDSS science image.
4. Resolve the target sky position into science-image pixel coordinates.
5. Acquire the native position-dependent SDSS psField PSF.
6. Convert the survey-specific SDSSPSF into the canonical PSFData model.
7. Attach the PSF to the Observation when available.

Baseline compatibility
----------------------
The raw image construction is unchanged from the original SDSS builder.

The PSF is an optional Observation-level calibration product. Baseline
photometry does not consume it, so failure to acquire the PSF does not
invalidate the Observation. SCARLET preprocessing is responsible for
requiring a PSF when its processing contract needs one.

The builder deliberately contains no photometry, source detection,
background subtraction, deblending, or SCARLET-specific processing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy.io.fits import HDUList, PrimaryHDU
from astropy.nddata import NDData
from astropy.wcs import WCS

from acquisition.sdss_psf import (
    SDSSPSFAcquirer,
    SDSSPSF,
    SDSSPSFError,
)
from core.exceptions import ObservationError
from core.models import TargetConfig
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


class SDSSObservationBuilder(ObservationBuilder[HDUList]):
    """
    Build a survey-independent Observation from an SDSS FITS HDUList.

    The builder preserves the acquired SDSS science image exactly and
    optionally attaches the native, position-dependent SDSS psField PSF.

    Parameters
    ----------
    target
        Target configuration containing the target name and sky coordinates.

    logger
        Optional application logger.

    psf_cache_dir
        Local cache directory used by SDSSPSFAcquirer.

    psf_data_release
        SDSS data release used for psField acquisition.

    psf_timeout
        Network timeout passed to SDSSPSFAcquirer.

    acquire_psf
        Whether to attempt native SDSS PSF acquisition. This defaults to
        True for production use. Setting it to False preserves the original
        baseline-only builder behaviour.
    """

    def __init__(
        self,
        target: TargetConfig,
        logger: logging.Logger | None = None,
        *,
        psf_cache_dir: str | Path = "data/raw/sdss/psf",
        psf_data_release: int = 17,
        psf_timeout: float = 60.0,
        acquire_psf: bool = True,
    ) -> None:
        self._target = target
        self._logger = logger or get_logger(__name__)

        self._psf_cache_dir = Path(psf_cache_dir)
        self._psf_data_release = int(psf_data_release)
        self._psf_timeout = float(psf_timeout)
        self._acquire_psf_enabled = bool(acquire_psf)

    # ======================================================================
    # Public API
    # ======================================================================

    def build(
        self,
        dataset: HDUList,
    ) -> Observation:
        """
        Convert an SDSS FITS HDUList into an Observation.

        The raw SDSS science image is always preserved.

        When PSF acquisition is enabled, the native SDSS PSF is reconstructed
        at the target's actual image position and attached to the Observation.

        PSF acquisition failure is non-fatal so that existing baseline
        photometry remains operational.
        """

        self._validate_dataset(dataset)

        metadata = self._extract_metadata(dataset)
        provenance = self._extract_provenance(dataset)

        # ------------------------------------------------------------------
        # Preserve the original raw SDSS science image.
        # ------------------------------------------------------------------
        raw_image = ImageData(
            self._create_image(dataset),
        )

        # ------------------------------------------------------------------
        # Optional native SDSS PSF.
        # ------------------------------------------------------------------
        psf: PSFData | None = None

        if self._acquire_psf_enabled:
            psf = self._try_acquire_psf(dataset)

        self._logger.info(
            "Constructed SDSS Observation for target '%s'. "
            "PSF available=%s.",
            metadata.target_name,
            psf is not None,
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

    # ======================================================================
    # Dataset validation
    # ======================================================================

    @staticmethod
    def _primary_hdu(
        dataset: HDUList,
    ) -> PrimaryHDU:
        """
        Return the primary science HDU.

        The explicit cast is used here because Astropy's HDUList typing
        exposes dataset indexing more broadly than the concrete PrimaryHDU
        type used by this builder.
        """

        return cast(
            PrimaryHDU,
            dataset[0],
        )

    def _validate_dataset(
        self,
        dataset: HDUList,
    ) -> None:
        """
        Validate the acquired SDSS dataset.
        """

        if dataset is None:
            raise ObservationError(
                "SDSS dataset cannot be None."
            )

        if len(dataset) == 0:
            raise ObservationError(
                "SDSS dataset contains no HDUs."
            )

        primary = self._primary_hdu(dataset)

        if primary.data is None:
            raise ObservationError(
                "SDSS primary HDU contains no science image data."
            )

        image = np.asarray(
            primary.data,
        )

        if image.ndim != 2:
            raise ObservationError(
                "SDSS science image must be two-dimensional; "
                f"received shape={image.shape}."
            )

        if image.size == 0:
            raise ObservationError(
                "SDSS science image is empty."
            )

        if not np.issubdtype(
            image.dtype,
            np.number,
        ):
            raise ObservationError(
                "SDSS science image must contain numeric data; "
                f"received dtype={image.dtype}."
            )

    # ======================================================================
    # FITS metadata
    # ======================================================================

    @staticmethod
    def _header_value(
        header: Any,
        keyword: str,
    ) -> Any:
        """
        Safely retrieve one FITS header value.

        The return type is intentionally Any because Astropy FITS header
        values may be strings, integers, floats, booleans, or other
        FITS-compatible scalar representations.
        """

        try:
            return header.get(keyword)
        except AttributeError:
            return None

    @classmethod
    def _find_metadata_value(
        cls,
        dataset: HDUList,
        keyword: str,
    ) -> Any:
        """
        Search the SDSS FITS dataset for a metadata keyword.

        The primary header is checked first, followed by all HDU headers.

        This is necessary because SDSS metadata such as FIELD can be present
        in a binary-table extension rather than in the primary header.
        """

        for hdu in dataset:
            header = getattr(
                hdu,
                "header",
                None,
            )

            if header is None:
                continue

            value = cls._header_value(
                header,
                keyword,
            )

            if value is not None:
                return value

        return None

    @classmethod
    def _required_int_metadata(
        cls,
        dataset: HDUList,
        keyword: str,
    ) -> int:
        """
        Read a required integer-valued SDSS metadata field.
        """

        value = cls._find_metadata_value(
            dataset,
            keyword,
        )

        if value is None:
            raise ObservationError(
                "SDSS dataset is missing required metadata "
                f"'{keyword}'."
            )

        try:
            return int(
                str(value).strip(),
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise ObservationError(
                f"Invalid integer value for SDSS metadata "
                f"'{keyword}': {value!r}."
            ) from exc

    @classmethod
    def _optional_float_metadata(
        cls,
        dataset: HDUList,
        keyword: str,
    ) -> float | None:
        """
        Read an optional floating-point SDSS metadata field.

        The explicit local variable and type narrowing prevent static type
        checkers from interpreting the FITS Header object itself as the
        argument to float().
        """

        value: Any = cls._find_metadata_value(
            dataset,
            keyword,
        )

        if value is None:
            return None

        try:
            converted = float(
                str(value).strip(),
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise ObservationError(
                f"Invalid floating-point value for SDSS metadata "
                f"'{keyword}': {value!r}."
            ) from exc

        if not np.isfinite(converted):
            raise ObservationError(
                f"SDSS metadata '{keyword}' is not finite: "
                f"{converted!r}."
            )

        return converted

    # ======================================================================
    # Observation metadata
    # ======================================================================

    def _extract_metadata(
        self,
        dataset: HDUList,
    ) -> ObservationMetadata:
        """
        Extract scientific metadata from the SDSS primary header.
        """

        primary = self._primary_hdu(dataset)
        header = primary.header

        instrument_value: Any = self._header_value(
            header,
            "INSTRUME",
        )

        if instrument_value is None:
            instrument_value = self._header_value(
                header,
                "TELESCOP",
            )

        instrument: str | None = None

        if instrument_value is not None:
            instrument = str(
                instrument_value,
            ).strip()

        filter_value: Any = self._header_value(
            header,
            "FILTER",
        )

        filter_name: str | None = None

        if filter_value is not None:
            filter_name = str(
                filter_value,
            ).strip()

        exposure_value: Any = self._header_value(
            header,
            "EXPTIME",
        )

        exposure_time: float | None = None

        if exposure_value is not None:
            try:
                converted_exposure = float(
                    str(exposure_value).strip(),
                )
            except (
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:
                raise ObservationError(
                    "Invalid SDSS EXPTIME value: "
                    f"{exposure_value!r}."
                ) from exc

            if not np.isfinite(
                converted_exposure,
            ):
                raise ObservationError(
                    "SDSS EXPTIME is not finite."
                )

            exposure_time = converted_exposure

        try:
            wcs = WCS(
                header,
            )
        except Exception as exc:
            raise ObservationError(
                "Unable to construct WCS from the SDSS science "
                "image header."
            ) from exc

        return ObservationMetadata(
            target_name=self._target.target_name,
            observation_time=None,
            observation_time_scale=None,
            observation_time_format=None,
            right_ascension=self._target.right_ascension,
            declination=self._target.declination,
            instrument=instrument,
            filter_name=filter_name,
            exposure_time=exposure_time,
            wcs=wcs,
        )

    # ======================================================================
    # Provenance
    # ======================================================================

    def _extract_provenance(
        self,
        dataset: HDUList,
    ) -> ObservationProvenance:
        """
        Extract SDSS provenance metadata.
        """

        primary = self._primary_hdu(dataset)
        header = primary.header

        run = self._find_metadata_value(
            dataset,
            "RUN",
        )

        rerun = self._find_metadata_value(
            dataset,
            "RERUN",
        )

        camcol = self._find_metadata_value(
            dataset,
            "CAMCOL",
        )

        field = self._find_metadata_value(
            dataset,
            "FIELD",
        )

        filter_value: Any = self._header_value(
            header,
            "FILTER",
        )

        if filter_value is None:
            filter_value = self._find_metadata_value(
                dataset,
                "FILTER",
            )

        run_text = (
            str(run).strip()
            if run is not None
            else "Unknown"
        )

        rerun_text = (
            str(rerun).strip()
            if rerun is not None
            else "Unknown"
        )

        camcol_text = (
            str(camcol).strip()
            if camcol is not None
            else "Unknown"
        )

        field_text = (
            str(field).strip()
            if field is not None
            else "Unknown"
        )

        filter_text = (
            str(filter_value).strip()
            if filter_value is not None
            else "Unknown"
        )

        dataset_identifier = (
            f"Run {run_text} / "
            f"Rerun {rerun_text} / "
            f"Camcol {camcol_text} / "
            f"Field {field_text} / "
            f"Filter {filter_text}"
        )

        return ObservationProvenance(
            source=ObservationSource.GROUND_BASED,
            survey=Survey.SDSS,
            dataset_identifier=dataset_identifier,
            filename=None,
            download_timestamp=datetime.now(
                timezone.utc,
            ),
        )

    # ======================================================================
    # Raw image construction
    # ======================================================================

    def _create_image(
        self,
        dataset: HDUList,
    ) -> NDData:
        """
        Create the initial raw SDSS science image.

        No scientific transformation is performed here.

        The acquired image values are copied into a new NDData object while
        preserving the original WCS, FITS metadata and unit information.
        """

        primary = self._primary_hdu(dataset)

        if primary.data is None:
            raise ObservationError(
                "SDSS primary HDU contains no science image data."
            )

        try:
            wcs = WCS(
                primary.header,
            )
        except Exception as exc:
            raise ObservationError(
                "Unable to construct WCS for the SDSS science image."
            ) from exc

        data = np.asarray(
            primary.data,
        ).copy()

        metadata = dict(
            primary.header,
        )

        unit = self._header_value(
            primary.header,
            "BUNIT",
        )

        return NDData(
            data=data,
            mask=None,
            uncertainty=None,
            wcs=wcs,
            meta=metadata,
            unit=unit,
        )

    # ======================================================================
    # Target -> science-image pixel coordinates
    # ======================================================================

    def _target_pixel(
        self,
        dataset: HDUList,
    ) -> tuple[float, float]:
        """
        Convert the configured target sky coordinate into SDSS pixel
        coordinates.

        Returns
        -------
        tuple[float, float]
            ``(x, y)`` pixel coordinates, where x is column and y is row.
        """

        primary = self._primary_hdu(dataset)

        if primary.data is None:
            raise ObservationError(
                "Cannot determine target pixel position because the "
                "SDSS science image contains no data."
            )

        right_ascension = self._target.right_ascension
        declination = self._target.declination

        if right_ascension is None:
            raise ObservationError(
                "SDSS target right ascension is not configured."
            )

        if declination is None:
            raise ObservationError(
                "SDSS target declination is not configured."
            )

        try:
            ra = float(
                right_ascension,
            )
            dec = float(
                declination,
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise ObservationError(
                "SDSS target coordinates must be numeric."
            ) from exc

        if not np.isfinite(ra):
            raise ObservationError(
                "SDSS target right ascension is not finite."
            )

        if not np.isfinite(dec):
            raise ObservationError(
                "SDSS target declination is not finite."
            )

        try:
            wcs = WCS(
                primary.header,
            )

            pixel = wcs.world_to_pixel_values(
                ra,
                dec,
            )
        except Exception as exc:
            raise ObservationError(
                "Unable to transform the SDSS target sky coordinate "
                "into image pixel coordinates."
            ) from exc

        if len(pixel) < 2:
            raise ObservationError(
                "SDSS WCS returned an unexpected number of pixel "
                "coordinates."
            )

        try:
            x = float(
                np.asarray(
                    pixel[0],
                ),
            )

            y = float(
                np.asarray(
                    pixel[1],
                ),
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise ObservationError(
                "SDSS WCS returned non-scalar pixel coordinates."
            ) from exc

        if not np.isfinite(x) or not np.isfinite(y):
            raise ObservationError(
                "SDSS target WCS transformation produced non-finite "
                f"pixel coordinates: x={x!r}, y={y!r}."
            )

        image = np.asarray(
            primary.data,
        )

        height = int(
            image.shape[0],
        )

        width = int(
            image.shape[1],
        )

        if not (
            0.0 <= x < width
            and 0.0 <= y < height
        ):
            raise ObservationError(
                "SDSS target position lies outside the science image: "
                f"x={x:.6f}, y={y:.6f}, "
                f"image_shape=({height}, {width})."
            )

        return (
            x,
            y,
        )

    # ======================================================================
    # Native SDSS PSF acquisition
    # ======================================================================

    def _try_acquire_psf(
        self,
        dataset: HDUList,
    ) -> PSFData | None:
        """
        Attempt to acquire and convert the native SDSS PSF.

        Failure is deliberately non-fatal.

        This preserves baseline compatibility while allowing the SCARLET
        preprocessing boundary to require a valid Observation-level PSF.
        """

        try:
            sdss_psf = self._acquire_sdss_psf(
                dataset,
            )

            return self._convert_psf(
                sdss_psf,
            )

        except ObservationError as exc:
            self._logger.warning(
                "Native SDSS PSF unavailable for target '%s': %s",
                self._target.target_name,
                exc,
            )
            return None

        except Exception as exc:
            self._logger.warning(
                "Unexpected error while acquiring native SDSS PSF "
                "for target '%s': %s",
                self._target.target_name,
                exc,
            )
            return None

    def _acquire_sdss_psf(
        self,
        dataset: HDUList,
    ) -> SDSSPSF:
        """
        Acquire the native position-dependent SDSS psField PSF.
        """

        x, y = self._target_pixel(
            dataset,
        )

        run = self._required_int_metadata(
            dataset,
            "RUN",
        )

        rerun = self._required_int_metadata(
            dataset,
            "RERUN",
        )

        camcol = self._required_int_metadata(
            dataset,
            "CAMCOL",
        )

        field = self._required_int_metadata(
            dataset,
            "FIELD",
        )

        primary = self._primary_hdu(
            dataset,
        )

        filter_value: Any = self._header_value(
            primary.header,
            "FILTER",
        )

        if filter_value is None:
            filter_value = self._find_metadata_value(
                dataset,
                "FILTER",
            )

        if filter_value is None:
            raise ObservationError(
                "SDSS dataset is missing required metadata 'FILTER'."
            )

        band = str(
            filter_value,
        ).strip().lower()

        if band not in {
            "u",
            "g",
            "r",
            "i",
            "z",
        }:
            raise ObservationError(
                "Invalid SDSS filter/band: "
                f"{band!r}."
            )

        self._logger.info(
            "Acquiring native SDSS PSF at "
            "(row=%.6f, col=%.6f).",
            y,
            x,
        )

        self._logger.debug(
            "SDSS PSF identifiers: "
            "run=%d, rerun=%d, camcol=%d, field=%d, band=%s.",
            run,
            rerun,
            camcol,
            field,
            band,
        )

        try:
            acquirer = SDSSPSFAcquirer(
                run=run,
                rerun=rerun,
                camcol=camcol,
                field=field,
                band=band,
                cache_dir=self._psf_cache_dir,
                data_release=self._psf_data_release,
                timeout=self._psf_timeout,
            )

            return acquirer.acquire(
                row=y,
                col=x,
            )

        except SDSSPSFError as exc:
            raise ObservationError(
                "Unable to acquire native SDSS PSF at "
                f"(row={y:.6f}, col={x:.6f})."
            ) from exc

        except Exception as exc:
            raise ObservationError(
                "Unexpected error while acquiring native SDSS PSF "
                f"for Run={run}, Rerun={rerun}, "
                f"Camcol={camcol}, Field={field}, Band={band}."
            ) from exc

    # ======================================================================
    # Survey-specific PSF -> domain PSF
    # ======================================================================

    @staticmethod
    def _convert_psf(
        psf: SDSSPSF,
    ) -> PSFData:
        """
        Convert an SDSSPSF into the canonical domain PSFData representation.

        The resulting PSFData contains no dependency on the SDSS-specific
        PSF class and can therefore be consumed by downstream processing.
        """

        if psf is None:
            raise ObservationError(
                "SDSS PSF conversion received None."
            )

        try:
            data = np.asarray(
                psf.data,
                dtype=np.float64,
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ObservationError(
                "SDSS PSF data could not be converted to float64."
            ) from exc

        if data.ndim != 2:
            raise ObservationError(
                "SDSS PSF must be two-dimensional; "
                f"received shape={data.shape}."
            )

        if data.size == 0:
            raise ObservationError(
                "SDSS PSF contains no pixels."
            )

        if not np.all(
            np.isfinite(data),
        ):
            raise ObservationError(
                "SDSS PSF contains non-finite values."
            )

        total_flux = float(
            np.sum(data),
        )

        if not np.isfinite(
            total_flux,
        ) or total_flux <= 0.0:
            raise ObservationError(
                "SDSS PSF has invalid total flux: "
                f"{total_flux!r}."
            )

        # The SDSSPSF returned by the acquisition layer is already
        # normalised. Do not renormalise here; preserve the acquired
        # scientific product.
        data = data.copy()

        try:
            psf_x = float(
                psf.col,
            )

            psf_y = float(
                psf.row,
            )

            pixel_scale = float(
                psf.pixel_scale_arcsec,
            )

            source_path = str(
                psf.source_path,
            )

            band = str(
                psf.band,
            )

            run = int(
                psf.run,
            )

            rerun = int(
                psf.rerun,
            )

            camcol = int(
                psf.camcol,
            )

            field = int(
                psf.field,
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise ObservationError(
                "SDSS PSF contains invalid provenance or position "
                "metadata."
            ) from exc

        if not np.isfinite(
            psf_x,
        ):
            raise ObservationError(
                "SDSS PSF column coordinate is not finite."
            )

        if not np.isfinite(
            psf_y,
        ):
            raise ObservationError(
                "SDSS PSF row coordinate is not finite."
            )

        if not np.isfinite(
            pixel_scale,
        ) or pixel_scale <= 0.0:
            raise ObservationError(
                "SDSS PSF pixel scale must be positive and finite."
            )

        return PSFData(
            data=data,
            model_type="SDSS_psField_KL",
            normalised=True,
            x=psf_x,
            y=psf_y,
            pixel_scale_arcsec=pixel_scale,
            source_path=source_path,
            metadata={
                "survey": "SDSS",
                "band": band,
                "run": run,
                "rerun": rerun,
                "camcol": camcol,
                "field": field,
                "reconstruction": (
                    "position-dependent psField"
                ),
            },
        )