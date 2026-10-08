"""
SCARLET input adaptation for the Photometric Analysis Pipeline.

This module defines the boundary between the project's survey-independent
Observation domain model and the SCARLET-specific processing layer.

The adapter prepares and validates the scientific inputs required by the
SCARLET preprocessing stage without importing SCARLET itself.

Responsibilities
----------------
* Select the image channels configured for SCARLET.
* Convert Astropy NDData images into finite 2-D floating-point arrays.
* Construct the channel-first image cube expected by SCARLET.
* Validate channel dimensions and image contents.
* Construct robust per-channel inverse-variance weights.
* Propagate the observation-level domain PSF when available.
* Validate the PSF when a PSF-dependent SCARLET workflow requires one.
* Validate target sky coordinates for TARGET initialisation.
* Validate source centres for FIELD initialisation.
* Preserve the source ImageData objects required for downstream provenance.

The adapter does not:
* perform background subtraction;
* alter astronomical flux values;
* clip or replace invalid pixels;
* perform source detection;
* perform source modelling;
* construct a SCARLET Frame;
* construct a SCARLET Observation;
* run SCARLET optimisation.

Those responsibilities remain with the appropriate pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from core.exceptions import PreprocessingError
from domain.models import ImageData, Observation, PSFData


# ---------------------------------------------------------------------------
# Public type aliases.
# ---------------------------------------------------------------------------

SourceCenter = tuple[float, float]


# ---------------------------------------------------------------------------
# SCARLET input contract.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ScarletInput:
    """
    Validated input representation for SCARLET processing.

    The input supports two mutually exclusive source-initialisation modes.

    TARGET mode
        ``target_sky_coord`` is populated and ``source_centers`` is ``None``.

    FIELD mode
        ``target_sky_coord`` is ``None`` and ``source_centers`` contains one
        or more image-space source centres represented as ``(y, x)`` pairs.

    Parameters
    ----------
    image_cube
        Channel-first floating-point image cube with shape
        ``(channels, height, width)``.

    weights
        Channel-first inverse-variance weight cube with the same shape as
        ``image_cube``.

    channel_names
        Ordered names corresponding to the first axis of ``image_cube``.

    target_sky_coord
        Target position as an ``(RA, Dec)`` pair in degrees for TARGET
        initialisation. ``None`` for FIELD initialisation.

    wcs
        Observation WCS, or ``None`` when no WCS is available.

    channel_images
        Original domain-level ImageData objects corresponding to
        ``channel_names``.

    psf
        Optional survey-independent observation-level PSF.

        This is intentionally represented by the domain ``PSFData`` type
        rather than a survey-specific PSF class. This keeps the adapter
        independent of SDSS and allows the same contract to be used by
        TESS or another survey.

    source_centers
        Source centres for FIELD initialisation as ``(y, x)`` image
        coordinates.
    """

    image_cube: np.ndarray
    weights: np.ndarray
    channel_names: tuple[str, ...]
    target_sky_coord: tuple[float, float] | None
    wcs: WCS | None
    channel_images: tuple[ImageData, ...]
    psf: PSFData | None = None
    source_centers: tuple[SourceCenter, ...] | None = None

    def __post_init__(self) -> None:
        """Validate invariants of the immutable SCARLET input contract."""

        if not isinstance(self.image_cube, np.ndarray):
            raise ValueError(
                "ScarletInput.image_cube must be a NumPy array."
            )

        if self.image_cube.ndim != 3:
            raise ValueError(
                "ScarletInput.image_cube must be a 3-D channel-first array."
            )

        if not isinstance(self.weights, np.ndarray):
            raise ValueError(
                "ScarletInput.weights must be a NumPy array."
            )

        if self.weights.shape != self.image_cube.shape:
            raise ValueError(
                "ScarletInput.weights must have the same shape as "
                "ScarletInput.image_cube."
            )

        if self.image_cube.shape[0] != len(self.channel_names):
            raise ValueError(
                "ScarletInput channel count does not match channel_names."
            )

        if len(self.channel_images) != len(self.channel_names):
            raise ValueError(
                "ScarletInput channel count does not match channel_images."
            )

        target_supplied = self.target_sky_coord is not None
        centers_supplied = self.source_centers is not None

        if target_supplied == centers_supplied:
            raise ValueError(
                "ScarletInput must contain exactly one source initialisation "
                "representation: target_sky_coord or source_centers."
            )

        if target_supplied:
            self._validate_target_sky_coord()

        if centers_supplied:
            self._validate_source_centers()

        if self.psf is not None:
            self._validate_psf()

    # ------------------------------------------------------------------
    # PSF validation.
    # ------------------------------------------------------------------

    def _validate_psf(self) -> None:
        """Validate the observation-level PSF."""

        if not isinstance(self.psf, PSFData):
            raise ValueError(
                "ScarletInput.psf must be a PSFData instance or None."
            )

        data = np.asarray(
            self.psf.data,
            dtype=np.float64,
        )

        if data.ndim != 2:
            raise ValueError(
                "ScarletInput PSF must be two-dimensional."
            )

        if data.size == 0:
            raise ValueError(
                "ScarletInput PSF must not be empty."
            )

        if not np.all(np.isfinite(data)):
            raise ValueError(
                "ScarletInput PSF contains non-finite values."
            )

        if not np.any(data != 0):
            raise ValueError(
                "ScarletInput PSF must contain non-zero values."
            )

        if self.psf.normalised:
            total = float(np.sum(data))

            if not np.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "ScarletInput normalised PSF must have a finite, "
                    "positive total flux."
                )

    # ------------------------------------------------------------------
    # Coordinate validation.
    # ------------------------------------------------------------------

    def _validate_target_sky_coord(self) -> None:
        """Validate the TARGET-mode sky coordinate."""

        assert self.target_sky_coord is not None

        if len(self.target_sky_coord) != 2:
            raise ValueError(
                "ScarletInput.target_sky_coord must contain RA and Dec."
            )

        right_ascension, declination = self.target_sky_coord

        if not np.isfinite(right_ascension):
            raise ValueError(
                "ScarletInput target right ascension is not finite."
            )

        if not np.isfinite(declination):
            raise ValueError(
                "ScarletInput target declination is not finite."
            )

        if not 0.0 <= right_ascension <= 360.0:
            raise ValueError(
                "ScarletInput target right ascension must be between "
                "0 and 360 degrees."
            )

        if not -90.0 <= declination <= 90.0:
            raise ValueError(
                "ScarletInput target declination must be between "
                "-90 and 90 degrees."
            )

    def _validate_source_centers(self) -> None:
        """Validate the FIELD-mode source-centre catalogue."""

        assert self.source_centers is not None

        if not self.source_centers:
            raise ValueError(
                "ScarletInput.source_centers cannot be empty."
            )

        height = self.image_cube.shape[1]
        width = self.image_cube.shape[2]

        for index, center in enumerate(self.source_centers):

            if len(center) != 2:
                raise ValueError(
                    "SCARLET source center "
                    f"{index} must contain (y, x)."
                )

            y, x = center

            if not np.isfinite(y) or not np.isfinite(x):
                raise ValueError(
                    "SCARLET source center "
                    f"{index} contains a non-finite coordinate."
                )

            if not 0.0 <= y < height:
                raise ValueError(
                    "SCARLET source center "
                    f"{index} has y={y}, outside image bounds "
                    f"[0, {height})."
                )

            if not 0.0 <= x < width:
                raise ValueError(
                    "SCARLET source center "
                    f"{index} has x={x}, outside image bounds "
                    f"[0, {width})."
                )


# ---------------------------------------------------------------------------
# Adapter.
# ---------------------------------------------------------------------------

class ScarletInputAdapter:
    """
    Adapt a domain Observation into validated SCARLET input.

    Parameters
    ----------
    channel_names
        Optional ordered sequence of image names to use as SCARLET
        channels. If omitted, the Observation's active image is used as
        a single channel.

    default_channel_name
        Name assigned to the single active-image channel when
        ``channel_names`` is not explicitly supplied.

    Notes
    -----
    This class deliberately does not import SCARLET.

    The adapter is responsible for constructing the common input contract.
    Actual SCARLET Frame, Observation, source and Blend construction remains
    in the SCARLET preprocessing layer.
    """

    DEFAULT_CHANNEL_NAME = "science"

    def __init__(
        self,
        channel_names: Sequence[str] | None = None,
        *,
        default_channel_name: str = DEFAULT_CHANNEL_NAME,
    ) -> None:

        if channel_names is not None:
            channel_names = tuple(channel_names)

            if not channel_names:
                raise ValueError(
                    "SCARLET channel_names cannot be empty."
                )

            if any(
                not isinstance(name, str) or not name.strip()
                for name in channel_names
            ):
                raise ValueError(
                    "SCARLET channel_names must contain non-empty strings."
                )

            if len(set(channel_names)) != len(channel_names):
                raise ValueError(
                    "SCARLET channel_names must be unique."
                )

        if (
            not isinstance(default_channel_name, str)
            or not default_channel_name.strip()
        ):
            raise ValueError(
                "SCARLET default_channel_name must be a non-empty string."
            )

        self._channel_names = channel_names
        self._default_channel_name = default_channel_name

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def prepare(
        self,
        observation: Observation,
        *,
        source_centers: Sequence[SourceCenter] | None = None,
        require_psf: bool = False,
    ) -> ScarletInput:
        """
        Validate and adapt an Observation for SCARLET.

        Parameters
        ----------
        observation
            Survey-independent domain observation.

        source_centers
            Optional sequence of detected source centres in ``(y, x)``
            image coordinates.

            When omitted, TARGET mode is used with the Observation's
            target sky coordinates.

            When supplied, FIELD mode is used.

        require_psf
            Require an observation-level PSF.

            This should be set to ``True`` for PSF-dependent production
            experiments such as the SDSS SCARLET workflow.

            The default remains ``False`` so the common adapter does not
            unnecessarily constrain surveys whose PSF architecture has
            not yet been established.

        Returns
        -------
        ScarletInput
            Validated SCARLET-compatible input representation.

        Raises
        ------
        PreprocessingError
            If the observation does not satisfy the SCARLET input contract.
        """

        self._validate_observation(
            observation,
        )

        if not observation.processing_state.background_removed:
            raise PreprocessingError(
                "SCARLET processing requires the common "
                "background-subtracted image."
            )

        if require_psf and observation.psf is None:
            raise PreprocessingError(
                "This SCARLET workflow requires an observation-level PSF, "
                "but no PSF is available."
            )

        channel_images = self._get_channel_images(
            observation,
        )

        image_cube = self._build_image_cube(
            channel_images,
        )

        channel_names = self._get_channel_names(
            len(channel_images),
        )

        if source_centers is None:
            target_sky_coord = self._target_sky_coordinate(
                observation,
            )

            normalized_source_centers = None

        else:
            target_sky_coord = None

            normalized_source_centers = self._source_centers(
                source_centers,
                image_shape=image_cube.shape[1:],
            )

        weights = self._build_weights(
            image_cube,
            channel_images,
        )

        return ScarletInput(
            image_cube=image_cube,
            weights=weights,
            channel_names=channel_names,
            target_sky_coord=target_sky_coord,
            wcs=observation.metadata.wcs,
            channel_images=tuple(channel_images),
            psf=observation.psf,
            source_centers=normalized_source_centers,
        )

    # ------------------------------------------------------------------
    # Channel selection.
    # ------------------------------------------------------------------

    def _get_channel_images(
        self,
        observation: Observation,
    ) -> list[ImageData]:
        """
        Return the domain images selected as SCARLET channels.

        If no explicit channels were configured, the active image is used
        as a single-channel SCARLET input.
        """

        if self._channel_names is None:
            return [observation.active]

        images: list[ImageData] = []

        for name in self._channel_names:

            if name not in observation.images:
                raise PreprocessingError(
                    f"SCARLET channel image '{name}' was not found "
                    "in the observation."
                )

            images.append(
                observation.images[name],
            )

        return images

    def _get_channel_names(
        self,
        count: int,
    ) -> tuple[str, ...]:
        """
        Return the ordered names corresponding to the SCARLET channels.
        """

        if self._channel_names is not None:

            if len(self._channel_names) != count:
                raise PreprocessingError(
                    "SCARLET channel configuration is inconsistent."
                )

            return self._channel_names

        if count != 1:
            raise PreprocessingError(
                "SCARLET channel configuration is inconsistent."
            )

        return (
            self._default_channel_name,
        )

    # ------------------------------------------------------------------
    # Image conversion.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_image_cube(
        images: Sequence[ImageData],
    ) -> np.ndarray:
        """
        Convert domain image channels into a channel-first image cube.

        Returns
        -------
        numpy.ndarray
            Floating-point array with shape ``(C, Y, X)``.
        """

        if not images:
            raise PreprocessingError(
                "SCARLET requires at least one image channel."
            )

        arrays = [
            ScarletInputAdapter._image_array(
                image.image,
            )
            for image in images
        ]

        reference_shape = arrays[0].shape

        for index, array in enumerate(arrays):

            if array.shape != reference_shape:
                raise PreprocessingError(
                    "SCARLET channel image "
                    f"{index} has shape {array.shape}; "
                    f"expected {reference_shape}."
                )

        return np.stack(
            arrays,
            axis=0,
        )

    @staticmethod
    def _image_array(
        image: NDData,
    ) -> np.ndarray:
        """
        Return an image as a finite 2-D float64 array.

        No pixel values are altered.

        Invalid values are rejected rather than silently replaced or
        clipped.
        """

        if image is None:
            raise PreprocessingError(
                "SCARLET channel image cannot be None."
            )

        if image.data is None:
            raise PreprocessingError(
                "SCARLET channel image contains no image data."
            )

        try:
            data = np.asarray(
                image.data,
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise PreprocessingError(
                "SCARLET channel image could not be converted to "
                "a floating-point array."
            ) from exc

        if data.ndim != 2:
            raise PreprocessingError(
                "SCARLET channel images must be 2-D."
            )

        if data.size == 0:
            raise PreprocessingError(
                "SCARLET cannot process an empty image."
            )

        if not np.all(np.isfinite(data)):
            raise PreprocessingError(
                "SCARLET input image contains non-finite values."
            )

        return data

    # ------------------------------------------------------------------
    # TARGET-mode coordinates.
    # ------------------------------------------------------------------

    @staticmethod
    def _target_sky_coordinate(
        observation: Observation,
    ) -> tuple[float, float]:
        """
        Return the target position as an ``(RA, Dec)`` pair in degrees.
        """

        metadata = observation.metadata

        if metadata.right_ascension is None:
            raise PreprocessingError(
                "SCARLET requires a target right ascension."
            )

        if metadata.declination is None:
            raise PreprocessingError(
                "SCARLET requires a target declination."
            )

        try:
            right_ascension = float(
                metadata.right_ascension,
            )

            declination = float(
                metadata.declination,
            )

        except (TypeError, ValueError, OverflowError) as exc:
            raise PreprocessingError(
                "SCARLET target coordinates must be numeric."
            ) from exc

        if not np.isfinite(right_ascension):
            raise PreprocessingError(
                "Target right ascension is not finite."
            )

        if not np.isfinite(declination):
            raise PreprocessingError(
                "Target declination is not finite."
            )

        if not 0.0 <= right_ascension <= 360.0:
            raise PreprocessingError(
                "Target right ascension must be between "
                "0 and 360 degrees."
            )

        if not -90.0 <= declination <= 90.0:
            raise PreprocessingError(
                "Target declination must be between "
                "-90 and 90 degrees."
            )

        return (
            right_ascension,
            declination,
        )

    # ------------------------------------------------------------------
    # FIELD-mode source centres.
    # ------------------------------------------------------------------

    @staticmethod
    def _source_centers(
        source_centers: Sequence[SourceCenter],
        *,
        image_shape: tuple[int, int],
    ) -> tuple[SourceCenter, ...]:
        """
        Validate and normalise FIELD-mode source centres.

        Coordinates are not transformed or rounded.

        SCARLET receives the validated image coordinates exactly as supplied
        by the upstream source-detection stage.
        """

        if len(image_shape) != 2:
            raise PreprocessingError(
                "SCARLET source-centre validation requires a 2-D image."
            )

        if not source_centers:
            raise PreprocessingError(
                "SCARLET FIELD processing requires at least "
                "one source centre."
            )

        height, width = image_shape

        normalised: list[SourceCenter] = []

        for index, center in enumerate(source_centers):

            try:
                if len(center) != 2:
                    raise PreprocessingError(
                        "SCARLET source center "
                        f"{index} must contain (y, x)."
                    )

            except TypeError as exc:
                raise PreprocessingError(
                    "SCARLET source center "
                    f"{index} must be a two-element sequence."
                ) from exc

            try:
                y = float(center[0])
                x = float(center[1])

            except (TypeError, ValueError, OverflowError) as exc:
                raise PreprocessingError(
                    "SCARLET source center "
                    f"{index} coordinates must be numeric."
                ) from exc

            if not np.isfinite(y) or not np.isfinite(x):
                raise PreprocessingError(
                    "SCARLET source center "
                    f"{index} contains a non-finite coordinate."
                )

            if not 0.0 <= y < height:
                raise PreprocessingError(
                    "SCARLET source center "
                    f"{index} has y={y}, outside image bounds "
                    f"[0, {height})."
                )

            if not 0.0 <= x < width:
                raise PreprocessingError(
                    "SCARLET source center "
                    f"{index} has x={x}, outside image bounds "
                    f"[0, {width})."
                )

            normalised.append(
                (y, x),
            )

        return tuple(normalised)

    # ------------------------------------------------------------------
    # Weight construction.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_weights(
        image_cube: np.ndarray,
        images: Sequence[ImageData],
    ) -> np.ndarray:
        """
        Construct channel-specific inverse-variance weights.

        The current domain model does not expose reliable per-pixel
        uncertainties. A robust global background RMS is therefore
        estimated independently for each channel using sigma-clipped
        statistics.

        Masked pixels receive zero weight.
        """

        if image_cube.ndim != 3:
            raise PreprocessingError(
                "SCARLET weight construction requires a 3-D image cube."
            )

        if image_cube.shape[0] != len(images):
            raise PreprocessingError(
                "SCARLET image and channel metadata are inconsistent."
            )

        weights = np.zeros_like(
            image_cube,
            dtype=np.float64,
        )

        for index, image_data in enumerate(image_cube):

            valid = np.isfinite(
                image_data,
            )

            mask = images[index].image.mask

            if mask is not None:

                try:
                    mask = np.asarray(
                        mask,
                        dtype=bool,
                    )

                except (TypeError, ValueError) as exc:
                    raise PreprocessingError(
                        "SCARLET image mask could not be converted "
                        "to a boolean array."
                    ) from exc

                if mask.shape != image_data.shape:
                    raise PreprocessingError(
                        "Image mask shape does not match image data shape."
                    )

                valid &= ~mask

            if not np.any(valid):
                raise PreprocessingError(
                    f"SCARLET channel {index} contains no valid pixels."
                )

            try:
                _, _, std = sigma_clipped_stats(
                    image_data[valid],
                    sigma=3.0,
                )

            except Exception as exc:
                raise PreprocessingError(
                    "Unable to estimate the background RMS for "
                    f"SCARLET channel {index}."
                ) from exc

            sigma = float(std)

            if not np.isfinite(sigma) or sigma <= 0.0:
                raise PreprocessingError(
                    "Unable to determine a positive background RMS "
                    f"for SCARLET channel {index}."
                )

            weights[index, valid] = 1.0 / (
                sigma * sigma
            )

        if not np.all(np.isfinite(weights)):
            raise PreprocessingError(
                "SCARLET weights contain non-finite values."
            )

        return weights

    # ------------------------------------------------------------------
    # Observation validation.
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_observation(
        observation: Observation,
    ) -> None:
        """
        Validate the minimum domain state required by the adapter.
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

        if observation.active.image.data is None:
            raise PreprocessingError(
                "Active image contains no image data."
            )


__all__ = [
    "SourceCenter",
    "ScarletInput",
    "ScarletInputAdapter",
]