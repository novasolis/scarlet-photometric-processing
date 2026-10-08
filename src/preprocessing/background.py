"""
Background subtraction for astronomical observations.

This module estimates and subtracts the astronomical sky background from
survey-independent Observation objects.

Large astronomical images are processed using Photutils' Background2D
algorithm, while small image cutouts, such as TESS Target Pixel Files,
use a sigma-clipped global background estimate.

Both approaches produce the same survey-independent background and
background-subtracted image products.

The original acquired image is never modified.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, cast

import numpy as np
from astropy.nddata import NDData
from astropy.stats import SigmaClip, sigma_clipped_stats
from numpy.typing import NDArray
from photutils.background import Background2D, MedianBackground

from core.exceptions import PreprocessingError
from core.models import PreprocessingConfig
from domain.models import ImageData, Observation
from preprocessing.preprocessing import Preprocessor


class BackgroundSubtractor(Preprocessor):
    """
    Estimate and subtract the astronomical image background.

    Two background-estimation strategies are used:

    * ``Background2D`` for sufficiently large images.
    * A sigma-clipped global median for small image cutouts.

    The resulting background and background-subtracted images are stored
    as named Observation image products.

    The background-subtracted image becomes the active image so that
    downstream source detection operates on a background-corrected
    dataset.

    Parameters
    ----------
    config
        Immutable background-preprocessing configuration.

    logger
        Optional logger used for diagnostic and processing messages.
    """

    #
    # Images smaller than this in either dimension are treated as
    # compact cutouts and use a global background estimate.
    #
    _MIN_BACKGROUND2D_SIZE = 100

    _BACKGROUND_IMAGE_NAME = "background"

    _BACKGROUND_REMOVED_IMAGE_NAME = "background_removed"

    def __init__(
        self,
        config: PreprocessingConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the background subtractor.

        Parameters
        ----------
        config
            Background preprocessing configuration.

        logger
            Optional logger.
        """

        super().__init__(
            logger,
        )

        self._config = config

        self._validate_configuration()

    def process(
        self,
        observation: Observation,
    ) -> Observation:
        """
        Estimate and subtract the image background.

        Parameters
        ----------
        observation
            Observation whose active image will be processed.

        Returns
        -------
        Observation
            Updated observation containing:

            * the estimated background image;
            * the background-subtracted image; and
            * the updated preprocessing state.

        Raises
        ------
        PreprocessingError
            If the observation is invalid, background estimation fails,
            or a derived image cannot be created.
        """

        self._validate_observation(
            observation,
        )

        if observation.processing_state.background_removed:
            raise PreprocessingError(
                "Background subtraction has already been completed "
                "for this observation."
            )

        image = observation.active.image

        self._logger.info(
            "Starting astronomical background subtraction."
        )

        image_data = self._image_array(
            image,
        )

        self._logger.info(
            "Input image shape: %s.",
            image_data.shape,
        )

        try:
            valid_pixel_count = int(
                np.count_nonzero(
                    self._valid_pixel_mask(
                        image,
                    )
                )
            )

            self._logger.info(
                "Valid pixels available for background estimation: "
                "%d of %d.",
                valid_pixel_count,
                image_data.size,
            )

            if valid_pixel_count == 0:
                raise PreprocessingError(
                    "Active image contains no valid pixels for "
                    "background estimation."
                )

            background = self._estimate_background(
                image,
            )

            self._validate_background(
                background,
                image_data.shape,
            )

            background_image = self._create_background_image(
                image,
                background,
            )

            corrected_image = self._create_corrected_image(
                image,
                background,
            )

            observation.add_image(
                self._BACKGROUND_IMAGE_NAME,
                ImageData(
                    background_image,
                ),
            )

            observation.add_image(
                self._BACKGROUND_REMOVED_IMAGE_NAME,
                ImageData(
                    corrected_image,
                ),
                make_active=True,
            )

            observation.processing_state.background_removed = True

            self._logger.info(
                "Background subtraction completed successfully."
            )

            self._logger.info(
                "Active image is now '%s'.",
                self._BACKGROUND_REMOVED_IMAGE_NAME,
            )

            return observation

        except PreprocessingError:
            raise

        except Exception as exc:
            self._logger.exception(
                "Background subtraction failed."
            )

            raise PreprocessingError(
                "Background subtraction failed."
            ) from exc

    def _estimate_background(
        self,
        image: NDData,
    ) -> NDArray[np.floating]:
        """
        Estimate the image background.

        Large astronomical CCD images are modelled using Photutils'
        Background2D algorithm. Small image cutouts use a sigma-clipped
        global median because their limited spatial extent does not
        provide sufficient information for a robust two-dimensional
        background model.

        Parameters
        ----------
        image
            Active image.

        Returns
        -------
        NDArray[np.floating]
            Estimated background image with the same shape as the
            input image.
        """

        image_data = self._image_array(
            image,
        )

        rows, columns = image_data.shape

        mask = self._combined_mask(
            image,
        )

        if (
            rows >= self._MIN_BACKGROUND2D_SIZE
            and columns >= self._MIN_BACKGROUND2D_SIZE
        ):
            self._logger.info(
                "Estimating background using Background2D."
            )

            return self._estimate_background_2d(
                image_data=image_data,
                mask=mask,
            )

        self._logger.info(
            "Estimating background using sigma-clipped global median."
        )

        return self._estimate_global_background(
            image_data=image_data,
            mask=mask,
        )

    def _estimate_background_2d(
        self,
        *,
        image_data: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        """
        Estimate a spatially varying background using Background2D.

        Parameters
        ----------
        image_data
            Two-dimensional image data.

        mask
            Boolean mask identifying pixels that must be excluded from
            background estimation.

        Returns
        -------
        NDArray[np.floating]
            Two-dimensional background model.

        Raises
        ------
        PreprocessingError
            If Background2D cannot produce a valid background model.
        """

        sigma_clip = SigmaClip(
            sigma=float(
                self._config.sigma_clip,
            ),
        )

        #
        # The installed Photutils typing currently exposes a narrower
        # static type for sigma_clip than the runtime API accepts.
        # Keep the cast local to this third-party API boundary.
        #
        background = Background2D(
            image_data,
            box_size=(
                self._config.background_box_size,
                self._config.background_box_size,
            ),
            filter_size=(
                self._config.background_filter_size,
                self._config.background_filter_size,
            ),
            mask=mask,
            exclude_percentile=float(
                self._config.exclude_percentile,
            ),
            sigma_clip=cast(
                Any,
                sigma_clip,
            ),
            bkg_estimator=MedianBackground(),
        )

        return np.asarray(
            background.background,
            dtype=np.float64,
        )

    def _estimate_global_background(
        self,
        *,
        image_data: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        """
        Estimate a spatially constant sigma-clipped background.

        Parameters
        ----------
        image_data
            Two-dimensional image data.

        mask
            Boolean mask identifying invalid or excluded pixels.

        Returns
        -------
        NDArray[np.floating]
            Constant background image matching the input shape.

        Raises
        ------
        PreprocessingError
            If a finite background estimate cannot be obtained.
        """

        _, median, standard_deviation = sigma_clipped_stats(
            image_data,
            mask=mask,
            sigma=float(
                self._config.sigma_clip,
            ),
        )

        median = float(
            median,
        )

        standard_deviation = float(
            standard_deviation,
        )

        if not np.isfinite(median):
            raise PreprocessingError(
                "Sigma-clipped background estimate is not finite."
            )

        if not np.isfinite(standard_deviation):
            raise PreprocessingError(
                "Sigma-clipped background standard deviation is not finite."
            )

        self._logger.debug(
            "Global background median: %.6f.",
            median,
        )

        self._logger.debug(
            "Global background standard deviation: %.6f.",
            standard_deviation,
        )

        return np.full(
            image_data.shape,
            fill_value=median,
            dtype=np.float64,
        )

    def _create_background_image(
        self,
        image: NDData,
        background: NDArray[np.floating],
    ) -> NDData:
        """
        Create an NDData image containing the estimated background.

        The background product preserves the original image's spatial
        metadata, mask, WCS, unit and PSF where applicable.

        The original image uncertainty is intentionally not copied
        because no uncertainty model for the estimated background has
        been calculated.
        """

        return NDData(
            data=np.array(
                background,
                dtype=np.float64,
                copy=True,
            ),
            mask=self._copy_mask(
                image.mask,
            ),
            wcs=image.wcs,
            meta=self._derived_metadata(
                image,
                product=self._BACKGROUND_IMAGE_NAME,
            ),
            unit=image.unit,
            copy=False,
            psf=self._copy_psf(
                image.psf,
            ),
        )

    def _create_corrected_image(
        self,
        image: NDData,
        background: NDArray[np.floating],
    ) -> NDData:
        """
        Create the background-subtracted image.

        The original image data is never modified.

        The original uncertainty is retained as the input-image
        uncertainty. It is not augmented with uncertainty from the
        estimated background because the current preprocessing model
        does not calculate background-model uncertainty.
        """

        image_data = self._image_array(
            image,
        )

        corrected = np.subtract(
            image_data,
            background,
            dtype=np.float64,
        )

        return NDData(
            data=corrected,
            uncertainty=image.uncertainty,
            mask=self._copy_mask(
                image.mask,
            ),
            wcs=image.wcs,
            meta=self._derived_metadata(
                image,
                product=self._BACKGROUND_REMOVED_IMAGE_NAME,
            ),
            unit=image.unit,
            copy=False,
            psf=self._copy_psf(
                image.psf,
            ),
        )

    @staticmethod
    def _image_array(
        image: NDData,
    ) -> NDArray[np.floating]:
        """
        Return image data as a floating-point NumPy array.

        Parameters
        ----------
        image
            NDData image.

        Returns
        -------
        NDArray[np.floating]
            Two-dimensional floating-point image array.
        """

        return np.asarray(
            image.data,
            dtype=np.float64,
        )

    @staticmethod
    def _valid_pixel_mask(
        image: NDData,
    ) -> NDArray[np.bool_]:
        """
        Return a mask identifying pixels usable for background estimation.

        A pixel is valid only when:

        * it is not masked by the input image; and
        * its numerical value is finite.
        """

        image_data = np.asarray(
            image.data,
        )

        mask = BackgroundSubtractor._combined_mask(
            image,
        )

        return (
            ~mask
            & np.isfinite(
                image_data,
            )
        )

    @staticmethod
    def _combined_mask(
        image: NDData,
    ) -> NDArray[np.bool_]:
        """
        Construct the complete background-estimation mask.

        The resulting mask includes:

        * the original NDData mask; and
        * NaN/infinite image pixels.
        """

        image_data = np.asarray(
            image.data,
        )

        combined = ~np.isfinite(
            image_data,
        )

        if image.mask is not None:
            image_mask = np.asarray(
                image.mask,
                dtype=bool,
            )

            if image_mask.shape != image_data.shape:
                raise PreprocessingError(
                    "Image mask shape does not match image data shape."
                )

            combined |= image_mask

        return combined

    @staticmethod
    def _copy_mask(
        mask: Any,
    ) -> Any:
        """
        Return a defensive copy of an image mask when present.
        """

        if mask is None:
            return None

        return np.array(
            mask,
            copy=True,
        )

    @staticmethod
    def _copy_psf(
        psf: Any,
    ) -> Any:
        """
        Return the PSF unchanged when present.

        PSF objects may be Astropy model instances rather than NumPy
        arrays, so they must not be coerced through ``np.array``.
        """

        return psf

    @staticmethod
    def _derived_metadata(
        image: NDData,
        *,
        product: str,
    ) -> dict[str, Any]:
        """
        Create metadata for a derived image product.
        """

        metadata: dict[str, Any] = {}

        if image.meta is not None:
            metadata.update(
                dict(
                    image.meta,
                )
            )

        metadata.update(
            {
                "processing_product": product,
                "background_created_utc": (
                    datetime.now(
                        timezone.utc,
                    ).isoformat()
                ),
            }
        )

        return metadata

    def _validate_observation(
        self,
        observation: Observation,
    ) -> None:
        """
        Validate the supplied Observation.

        Raises
        ------
        PreprocessingError
            If the observation does not contain a usable active image.
        """

        if observation is None:
            raise PreprocessingError(
                "Observation cannot be None."
            )

        if observation.active is None:
            raise PreprocessingError(
                "Observation has no active image."
            )

        if observation.active.image is None:
            raise PreprocessingError(
                "Observation contains no active image data."
            )

        image = observation.active.image

        if image.data is None:
            raise PreprocessingError(
                "Active image contains no image data."
            )

        image_data = np.asarray(
            image.data,
        )

        if image_data.ndim != 2:
            raise PreprocessingError(
                "Background subtraction requires a "
                "two-dimensional image."
            )

        if image_data.size == 0:
            raise PreprocessingError(
                "Background subtraction cannot operate on "
                "an empty image."
            )

        if image.mask is not None:
            mask = np.asarray(
                image.mask,
            )

            if mask.shape != image_data.shape:
                raise PreprocessingError(
                    "Image mask shape does not match image data shape."
                )

    def _validate_background(
        self,
        background: NDArray[np.floating],
        expected_shape: tuple[int, ...],
    ) -> None:
        """
        Validate the generated background model.

        Raises
        ------
        PreprocessingError
            If the background has an incorrect shape or contains
            non-finite values.
        """

        background_array = np.asarray(
            background,
        )

        if background_array.shape != expected_shape:
            raise PreprocessingError(
                "Estimated background shape does not match "
                "the input image shape."
            )

        if not np.all(
            np.isfinite(
                background_array,
            )
        ):
            raise PreprocessingError(
                "Estimated background contains non-finite values."
            )

    def _validate_configuration(
        self,
    ) -> None:
        """
        Validate preprocessing configuration.

        Raises
        ------
        ValueError
            If any background-estimation parameter is invalid.
        """

        box_size = int(
            self._config.background_box_size,
        )

        filter_size = int(
            self._config.background_filter_size,
        )

        sigma_clip = float(
            self._config.sigma_clip,
        )

        exclude_percentile = float(
            self._config.exclude_percentile,
        )

        if box_size <= 0:
            raise ValueError(
                "Background box size must be greater than zero."
            )

        if filter_size <= 0:
            raise ValueError(
                "Background filter size must be greater than zero."
            )

        if sigma_clip <= 0.0 or not np.isfinite(
            sigma_clip,
        ):
            raise ValueError(
                "Sigma-clip threshold must be a finite value "
                "greater than zero."
            )

        if (
            exclude_percentile < 0.0
            or exclude_percentile > 100.0
            or not np.isfinite(
                exclude_percentile,
            )
        ):
            raise ValueError(
                "Background exclude percentile must be between "
                "0 and 100."
            )