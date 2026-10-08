"""
SCARLET-based astronomical image preprocessing.

This module integrates the SCARLET source-modelling algorithm with the
project's survey-independent Observation domain model.

Two SCARLET source-initialisation modes are supported:

TARGET
    A single target is initialized from its sky coordinate using
    ``scarlet.ExtendedSource``.

FIELD
    Multiple detected sources are supplied by the photometry pipeline in
    ``(y, x)`` image coordinates. Because SCARLET's WCS-aware source
    initialization expects sky coordinates, the preprocessor converts those
    pixel centres to ``(RA, Dec)`` at the SCARLET boundary before calling
    ``scarlet.initialization.init_all_sources``.

The SCARLET input boundary is handled by ``ScarletInputAdapter``.
This preprocessor is responsible for executing SCARLET and storing its
derived products back on the Observation.

PSF handling
------------
The Observation may contain a survey-specific PSF represented through the
survey-independent ``PSFData`` domain model.

When present, that PSF is converted into SCARLET's ``ImagePSF`` and supplied
to ``scarlet.Observation``. This is the correct SCARLET architecture for
survey observations whose native PSF is known.

For SDSS, the PSF is reconstructed from the native ``psField`` product at
the target image position by the acquisition/builder layers before this
preprocessor is invoked.

TARGET mode preserves the existing downstream behaviour: the reconstructed
target channel becomes the active image for classical aperture photometry.

FIELD mode deliberately does not make the full-scene SCARLET model active.
The fitted model is rendered through the matched SCARLET Observation so the
reconstruction and residual are in the same observational space as the input
image. Individual source products use the same rendering. The existing
active image is preserved because a full-scene model is not an isolated
target image.
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
import logging
from typing import Any, Sequence

import numpy as np
from astropy.nddata import NDData

from core.exceptions import PreprocessingError
from domain.models import ImageData, Observation
from preprocessing.preprocessing import Preprocessor
from preprocessing.scarlet_adapter import (
    ScarletInput,
    ScarletInputAdapter,
)
from utils.logger import get_logger


class ScarletPreprocessor(Preprocessor):
    """
    SCARLET-based source-separation preprocessor.

    The preprocessor delegates domain-to-SCARLET input preparation to
    ``ScarletInputAdapter`` and then executes the SCARLET source-modelling
    workflow.

    Two source-initialisation modes are supported.

    TARGET mode
        The Observation target coordinates are used to construct a single
        ``ExtendedSource``.

    FIELD mode
        Explicit ``(y, x)`` source centres are supplied to ``process`` in
        the project's canonical image-coordinate convention. The centres are
        converted to SCARLET sky coordinates immediately before source
        initialization; the canonical pixel coordinates are preserved for
        downstream photometry and provenance.

    In single-channel mode, the active Observation image is used.

    In multi-channel mode, ``channel_names`` identifies image products stored
    in ``Observation.images``. SCARLET models all supplied channels jointly,
    while ``photometry_channel`` determines which reconstructed channel
    becomes active for TARGET processing.

    Parameters
    ----------
    max_iter
        Maximum number of SCARLET optimisation iterations.

    e_rel
        Relative convergence tolerance passed to ``Blend.fit``.

    channel_names
        Optional ordered sequence of image names to use as SCARLET channels.
        If omitted, the active image is used as a single channel.

    photometry_channel
        Name of the channel whose reconstructed target image becomes the
        active image in TARGET mode. Required when more than one channel
        is used.

    field_max_components
        Maximum number of components requested by SCARLET's robust field
        source initialisation.

    field_min_snr
        Minimum signal-to-noise ratio used by SCARLET field source
        initialisation.

    field_thresh
        Detection threshold passed to SCARLET field source initialisation.

    field_fallback
        Whether SCARLET may fall back to simpler source models when the
        requested model cannot be initialized.

    field_max_sources
        Retained only for backward-compatible configuration. The
        authoritative FIELD source population is supplied explicitly by the
        workflow through ``source_centers`` (normally the selected
        ``LocalField``). This preprocessor does not re-select or truncate
        those sources. ``None`` is the default so SCARLET does not expose a
        second source-count limit.

    logger
        Optional logger. If omitted, the project logger is used.
    """

    _TARGET_IMAGE_NAME = "scarlet_target"
    _MODEL_IMAGE_NAME = "scarlet_model"
    _RESIDUAL_IMAGE_NAME = "scarlet_residual"

    _FIELD_MODEL_IMAGE_NAME = "scarlet_field_model"
    _FIELD_RESIDUAL_IMAGE_NAME = "scarlet_field_residual"
    _FIELD_INTRINSIC_MODEL_IMAGE_NAME = "scarlet_field_model_intrinsic"
    _FIELD_SOURCE_PREFIX = "scarlet_source_"

    def __init__(
        self,
        max_iter: int = 100,
        e_rel: float = 1.0e-4,
        channel_names: Sequence[str] | None = None,
        photometry_channel: str | None = None,
        *,
        field_max_components: int = 2,
        field_min_snr: float = 50.0,
        field_thresh: float = 1.0,
        field_fallback: bool = True,
        field_max_sources: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            logger=logger or get_logger(__name__),
        )

        if max_iter <= 0:
            raise ValueError(
                "SCARLET max_iter must be greater than zero."
            )

        if e_rel <= 0:
            raise ValueError(
                "SCARLET e_rel must be greater than zero."
            )

        if field_max_components <= 0:
            raise ValueError(
                "SCARLET field_max_components must be greater than zero."
            )

        if field_min_snr <= 0:
            raise ValueError(
                "SCARLET field_min_snr must be greater than zero."
            )

        if field_thresh <= 0:
            raise ValueError(
                "SCARLET field_thresh must be greater than zero."
            )

        if field_max_sources is not None and field_max_sources <= 0:
            raise ValueError(
                "SCARLET field_max_sources must be greater than zero "
                "when specified."
            )

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
            channel_names is not None
            and len(channel_names) > 1
            and photometry_channel is None
        ):
            raise ValueError(
                "photometry_channel is required for multi-channel "
                "SCARLET processing."
            )

        if (
            photometry_channel is not None
            and channel_names is not None
            and photometry_channel not in channel_names
        ):
            raise ValueError(
                "photometry_channel must be one of channel_names."
            )

        self._max_iter = max_iter
        self._e_rel = e_rel

        self._channel_names = (
            tuple(channel_names)
            if channel_names is not None
            else None
        )

        self._photometry_channel = photometry_channel

        self._field_max_components = field_max_components
        self._field_min_snr = field_min_snr
        self._field_thresh = field_thresh
        self._field_fallback = field_fallback
        self._field_max_sources = field_max_sources

        self._adapter = ScarletInputAdapter(
            channel_names=self._channel_names,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        observation: Observation,
        *,
        source_centers: Sequence[tuple[float, float]] | None = None,
    ) -> Observation:
        """
        Apply SCARLET source modelling to an Observation.

        Parameters
        ----------
        observation
            Survey-independent domain Observation.

        source_centers
            Optional FIELD-mode source centres in the project's canonical
            ``(y, x)`` image-coordinate convention. They are converted to
            SCARLET sky coordinates at the SCARLET boundary.

            If omitted, TARGET mode is selected and the target coordinates
            stored in the Observation are used.

            If supplied, FIELD mode is selected and SCARLET initializes
            multiple sources using ``init_all_sources``.

        Returns
        -------
        Observation
            The supplied Observation with SCARLET-derived products stored
            in ``Observation.images``.

        Raises
        ------
        PreprocessingError
            If SCARLET processing fails or produces invalid output.
        """
        self._validate_observation(observation)

        if not observation.processing_state.background_removed:
            raise PreprocessingError(
                "SCARLET processing requires the common "
                "background-subtracted image."
            )

        if observation.processing_state.scarlet_applied:
            raise PreprocessingError(
                "SCARLET processing has already been completed "
                "for this observation."
            )

        mode = (
            "FIELD"
            if source_centers is not None
            else "TARGET"
        )

        self._logger.info(
            "Starting SCARLET %s processing for target '%s'.",
            mode,
            observation.metadata.target_name,
        )

        try:
            scarlet_input = self._adapter.prepare(
                observation,
                source_centers=source_centers,
            )

            self._logger.info(
                "SCARLET input cube shape: %s.",
                scarlet_input.image_cube.shape,
            )

            scarlet = self._load_scarlet()

            scarlet_observation = self._build_scarlet_observation(
                scarlet=scarlet,
                scarlet_input=scarlet_input,
                observation=observation,
            )

            model_frame = self._build_model_frame(
                scarlet=scarlet,
                scarlet_input=scarlet_input,
            )

            scarlet_observation.match(
                model_frame,
            )

            if scarlet_input.source_centers is not None:
                return self._process_field(
                    observation=observation,
                    scarlet=scarlet,
                    scarlet_input=scarlet_input,
                    scarlet_observation=scarlet_observation,
                    model_frame=model_frame,
                )

            return self._process_target(
                observation=observation,
                scarlet=scarlet,
                scarlet_input=scarlet_input,
                scarlet_observation=scarlet_observation,
                model_frame=model_frame,
            )

        except PreprocessingError:
            raise

        except Exception as exc:
            self._logger.exception(
                "SCARLET processing failed."
            )

            raise PreprocessingError(
                "SCARLET processing failed."
            ) from exc

    # ------------------------------------------------------------------
    # TARGET processing
    # ------------------------------------------------------------------

    def _process_target(
        self,
        *,
        observation: Observation,
        scarlet: Any,
        scarlet_input: ScarletInput,
        scarlet_observation: Any,
        model_frame: Any,
    ) -> Observation:
        """
        Execute the single-target SCARLET workflow.
        """
        if scarlet_input.target_sky_coord is None:
            raise PreprocessingError(
                "SCARLET TARGET processing requires a target sky coordinate."
            )

        source: Any = scarlet.ExtendedSource(
            model_frame,
            scarlet_input.target_sky_coord,
            scarlet_observation,
            K=1,
            thresh=1.0,
        )

        sources = [source]

        blend: Any = scarlet.Blend(
            sources,
            scarlet_observation,
        )

        self._logger.info(
            "Fitting SCARLET target model "
            "(channels=%s, max_iter=%d, e_rel=%g).",
            scarlet_input.channel_names,
            self._max_iter,
            self._e_rel,
        )

        iterations, log_likelihood = self._fit_blend(
            blend,
        )

        self._logger.info(
            "SCARLET target fit completed "
            "(iterations=%s, log_likelihood=%s).",
            iterations,
            log_likelihood,
        )

        target_model = source.get_model(
            frame=model_frame,
        )

        full_model = blend.get_model(
            frame=model_frame,
        )

        expected_channels = len(
            scarlet_input.channel_names,
        )

        target_cube = self._as_model_cube(
            target_model,
            expected_channels=expected_channels,
        )

        intrinsic_model_cube = self._as_model_cube(
            full_model,
            expected_channels=expected_channels,
        )

        expected_shape = scarlet_input.image_cube.shape

        target_rendered_cube = self._render_observation_cube(
            scarlet_observation,
            target_cube,
            expected_shape=expected_shape,
            description="SCARLET target rendered cube",
        )
        rendered_model_cube = self._render_observation_cube(
            scarlet_observation,
            intrinsic_model_cube,
            expected_shape=expected_shape,
            description="SCARLET model rendered cube",
        )

        residual_cube = scarlet_input.image_cube - rendered_model_cube

        self._validate_cube(
            target_rendered_cube, expected_shape,
            "SCARLET target rendered cube",
        )
        self._validate_cube(
            rendered_model_cube, expected_shape,
            "SCARLET model rendered cube",
        )
        self._validate_cube(
            residual_cube, expected_shape,
            "SCARLET residual cube",
        )

        photometry_channel = self._get_photometry_channel(
            scarlet_input.channel_names,
        )

        photometry_index = scarlet_input.channel_names.index(
            photometry_channel,
        )

        target_image = target_rendered_cube[
            photometry_index
        ]

        model_image = rendered_model_cube[
            photometry_index
        ]

        residual_image = residual_cube[
            photometry_index
        ]

        source_image = scarlet_input.channel_images[
            photometry_index
        ].image

        target_nddata = self._create_derived_image(
            source_image,
            target_image,
            product=self._TARGET_IMAGE_NAME,
            channel=photometry_channel,
            extra_metadata={
                "scarlet_representation": "observation_space_psf_rendered",
            },
        )

        model_nddata = self._create_derived_image(
            source_image,
            model_image,
            product=self._MODEL_IMAGE_NAME,
            channel=photometry_channel,
            extra_metadata={
                "scarlet_representation": "observation_space_psf_rendered",
            },
        )

        residual_nddata = self._create_derived_image(
            source_image,
            residual_image,
            product=self._RESIDUAL_IMAGE_NAME,
            channel=photometry_channel,
            extra_metadata={
                "scarlet_representation": "observation_space_residual",
                "scarlet_residual_definition": "observed_minus_rendered_model",
            },
        )

        observation.add_image(
            self._TARGET_IMAGE_NAME,
            ImageData(target_nddata),
            make_active=True,
        )

        observation.add_image(
            self._MODEL_IMAGE_NAME,
            ImageData(model_nddata),
        )

        observation.add_image(
            self._RESIDUAL_IMAGE_NAME,
            ImageData(residual_nddata),
        )

        observation.processing_state.scarlet_applied = True

        self._logger.info(
            "SCARLET TARGET processing completed successfully."
        )

        self._logger.info(
            "Active image is SCARLET target channel '%s'.",
            photometry_channel,
        )

        return observation

    # ------------------------------------------------------------------
    # FIELD processing
    # ------------------------------------------------------------------

    def _process_field(
        self,
        *,
        observation: Observation,
        scarlet: Any,
        scarlet_input: ScarletInput,
        scarlet_observation: Any,
        model_frame: Any,
    ) -> Observation:
        """
        Execute multi-source FIELD SCARLET processing.

        The implementation uses SCARLET's robust field initialisation
        strategy through ``init_all_sources``. The supplied source centres
        are authoritative: they are consumed exactly as provided by the
        workflow's LocalField selection, with no additional source filtering
        or truncation inside SCARLET. Canonical image-coordinate centres are
        converted to sky coordinates before the SCARLET call.

        Individual source models are retained as separate Observation
        products after rendering into observation space. The full scene
        reconstruction and residual are also retained. An additional
        intrinsic model-frame field product is stored for diagnostics.

        The active image is intentionally not changed because a full-scene
        model is not equivalent to an isolated target image.
        """
        if scarlet_input.source_centers is None:
            raise PreprocessingError(
                "SCARLET FIELD processing requires source centres."
            )

        field_started = perf_counter()

        all_pixel_centers = tuple(
            (float(center[0]), float(center[1]))
            for center in scarlet_input.source_centers
        )
        detected_count = len(all_pixel_centers)

        if detected_count == 0:
            raise PreprocessingError(
                "SCARLET FIELD processing requires at least one source centre."
            )

        # ``source_centers`` is authoritative. The workflow has already
        # performed complete-catalogue detection and target-centred
        # LocalField selection. SCARLET must therefore consume exactly that
        # selected population, in exactly the order supplied, without a
        # second source-selection policy.
        #
        # ``field_max_sources`` is intentionally retained on the constructor
        # for backward compatibility, but it is no longer allowed to change
        # the explicit FIELD source list. This keeps source selection in one
        # architectural location: LocalFieldSelector.
        selected_indices = tuple(range(detected_count))
        pixel_centers = all_pixel_centers

        self._logger.info(
            "SCARLET FIELD source scope received from workflow: "
            "detected_input=%d, selected=%d, configured_legacy_max_sources=%s. "
            "No secondary source selection will be applied.",
            detected_count,
            len(pixel_centers),
            self._field_max_sources,
        )

        conversion_started = perf_counter()
        sky_centers = self._convert_field_centers_to_sky(
            scarlet_observation=scarlet_observation,
            pixel_centers=pixel_centers,
        )
        self._logger.info(
            "SCARLET FIELD coordinate conversion completed: sources=%d, "
            "elapsed=%.3fs.",
            len(pixel_centers),
            perf_counter() - conversion_started,
        )

        self._logger.info(
            "Initializing %d SCARLET field sources "
            "(detected=%d, max_sources=%s, max_components=%d, "
            "min_snr=%g, thresh=%g).",
            len(pixel_centers),
            detected_count,
            self._field_max_sources,
            self._field_max_components,
            self._field_min_snr,
            self._field_thresh,
        )

        try:
            # SCARLET's WCS-aware initialization API expects centres in sky
            # coordinates even though the project's source catalogue uses
            # image coordinates. Keep set_spectra=False here so an empty
            # source list produces a controlled error instead of allowing
            # SCARLET's internal np.stack([]) failure to escape. The official
            # API supports setting spectra as a separate operation.
            sources, skipped = (
                scarlet.initialization.init_all_sources(
                    model_frame,
                    sky_centers,
                    scarlet_observation,
                    max_components=self._field_max_components,
                    min_snr=self._field_min_snr,
                    thresh=self._field_thresh,
                    fallback=self._field_fallback,
                    silent=True,
                    set_spectra=False,
                )
            )

        except Exception as exc:
            raise PreprocessingError(
                "SCARLET FIELD source initialization failed."
            ) from exc

        skipped = list(skipped)

        if not sources:
            raise PreprocessingError(
                "SCARLET FIELD source initialization produced "
                "no usable sources."
            )

        try:
            scarlet.initialization.set_spectra_to_match(
                sources,
                scarlet_observation,
            )
        except Exception as exc:
            raise PreprocessingError(
                "SCARLET FIELD source spectrum initialization failed."
            ) from exc

        if skipped:
            self._logger.warning(
                "SCARLET skipped %d of %d requested field sources: %s.",
                len(skipped),
                len(scarlet_input.source_centers),
                skipped,
            )
        else:
            self._logger.info(
                "SCARLET initialized all %d selected field sources.",
                len(pixel_centers),
            )

        blend: Any = scarlet.Blend(
            sources,
            scarlet_observation,
        )

        self._logger.info(
            "Fitting SCARLET field model "
            "(sources=%d, channels=%s, max_iter=%d, e_rel=%g).",
            len(sources),
            scarlet_input.channel_names,
            self._max_iter,
            self._e_rel,
        )

        iterations, log_likelihood = self._fit_blend(
            blend,
        )

        self._logger.info(
            "SCARLET field fit completed "
            "(sources=%d, iterations=%s, log_likelihood=%s).",
            len(sources),
            iterations,
            log_likelihood,
        )

        full_model = blend.get_model(
            frame=model_frame,
        )

        expected_channels = len(
            scarlet_input.channel_names,
        )

        intrinsic_model_cube = self._as_model_cube(
            full_model,
            expected_channels=expected_channels,
        )

        expected_shape = scarlet_input.image_cube.shape

        rendered_model_cube = self._render_observation_cube(
            scarlet_observation,
            intrinsic_model_cube,
            expected_shape=expected_shape,
            description="SCARLET field rendered model cube",
        )

        self._logger.info(
            "SCARLET FIELD observation-space reconstruction generated "
            "(shape=%s, PSF-rendered=True).",
            rendered_model_cube.shape,
        )

        residual_cube = scarlet_input.image_cube - rendered_model_cube

        self._validate_cube(
            rendered_model_cube, expected_shape,
            "SCARLET field rendered model cube",
        )
        self._validate_cube(
            residual_cube, expected_shape,
            "SCARLET field residual cube",
        )

        photometry_channel = self._get_photometry_channel(
            scarlet_input.channel_names,
        )

        photometry_index = scarlet_input.channel_names.index(
            photometry_channel,
        )

        source_image = scarlet_input.channel_images[
            photometry_index
        ].image

        model_image = rendered_model_cube[
            photometry_index
        ]

        residual_image = residual_cube[
            photometry_index
        ]

        field_metadata = {
            "scarlet_mode": "FIELD",
            "scarlet_coordinate_system": "pixel_to_sky",
            "scarlet_detected_sources": detected_count,
            "scarlet_selected_sources": len(pixel_centers),
            "scarlet_requested_sources": len(pixel_centers),
            "scarlet_max_sources": self._field_max_sources,
            "scarlet_initialized_sources": len(sources),
            "scarlet_skipped_sources": len(skipped),
        }

        field_model_nddata = self._create_derived_image(
            source_image,
            model_image,
            product=self._FIELD_MODEL_IMAGE_NAME,
            channel=photometry_channel,
            extra_metadata={
                **field_metadata,
                "scarlet_representation": "observation_space_psf_rendered",
            },
        )

        intrinsic_model_nddata = self._create_derived_image(
            source_image,
            intrinsic_model_cube[photometry_index],
            product=self._FIELD_INTRINSIC_MODEL_IMAGE_NAME,
            channel=photometry_channel,
            extra_metadata={
                **field_metadata,
                "scarlet_representation": "model_frame_intrinsic",
            },
        )

        field_residual_nddata = self._create_derived_image(
            source_image,
            residual_image,
            product=self._FIELD_RESIDUAL_IMAGE_NAME,
            channel=photometry_channel,
            extra_metadata={
                **field_metadata,
                "scarlet_representation": "observation_space_residual",
                "scarlet_residual_definition": "observed_minus_rendered_model",
            },
        )

        observation.add_image(
            self._FIELD_MODEL_IMAGE_NAME,
            ImageData(field_model_nddata),
            #make_active=True
        )

        observation.add_image(
            self._FIELD_INTRINSIC_MODEL_IMAGE_NAME,
            ImageData(intrinsic_model_nddata),
        )

        observation.add_image(
            self._FIELD_RESIDUAL_IMAGE_NAME,
            ImageData(field_residual_nddata),
        )

        self._store_field_source_models(
            observation=observation,
            sources=sources,
            model_frame=model_frame,
            scarlet_observation=scarlet_observation,
            source_image=source_image,
            channel=photometry_channel,
            photometry_index=photometry_index,
            expected_channels=expected_channels,
            requested_centers=pixel_centers,
            requested_sky_centers=sky_centers,
            requested_source_indices=selected_indices,
            skipped=skipped,
        )

        observation.processing_state.scarlet_applied = True

        self._logger.info(
            "SCARLET FIELD processing completed successfully "
            "(detected=%d, selected=%d, initialized=%d, skipped=%d, "
            "elapsed=%.3fs). Active image was preserved as the "
            "pre-SCARLET image.",
            detected_count,
            len(pixel_centers),
            len(sources),
            len(skipped),
            perf_counter() - field_started,
        )

        return observation

    # ------------------------------------------------------------------
    # FIELD coordinate conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_field_centers_to_sky(
        *,
        scarlet_observation: Any,
        pixel_centers: Sequence[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Convert canonical FIELD ``(y, x)`` pixels to SCARLET sky coords.

        The project-level source catalogue deliberately remains in image
        coordinates. SCARLET's WCS-aware initialization, however, expects
        ``(RA, Dec)`` sky coordinates when an Observation has a WCS. This
        helper is the single explicit coordinate-system boundary.

        Parameters
        ----------
        scarlet_observation
            SCARLET Observation carrying the survey WCS.
        pixel_centers
            Source centres in the canonical ``(y, x)`` image convention.

        Returns
        -------
        list[tuple[float, float]]
            SCARLET sky coordinates as ``(RA_deg, Dec_deg)`` pairs.

        Raises
        ------
        PreprocessingError
            If the input centres or resulting sky coordinates are malformed
            or non-finite.
        """
        pixels = np.asarray(pixel_centers, dtype=np.float64)

        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise PreprocessingError(
                "SCARLET FIELD source centres must have shape (N, 2) "
                "in (y, x) pixel coordinates; "
                f"received shape={pixels.shape}."
            )

        if pixels.shape[0] == 0:
            raise PreprocessingError(
                "SCARLET FIELD processing requires at least one source "
                "centre."
            )

        if not np.all(np.isfinite(pixels)):
            raise PreprocessingError(
                "SCARLET FIELD source centres contain non-finite pixel "
                "coordinates."
            )

        try:
            sky = np.asarray(
                scarlet_observation.get_sky_coord(pixels),
                dtype=np.float64,
            )
        except Exception as exc:
            raise PreprocessingError(
                "SCARLET could not convert FIELD pixel coordinates to "
                "sky coordinates using the Observation WCS."
            ) from exc

        if sky.shape != pixels.shape:
            raise PreprocessingError(
                "SCARLET returned FIELD sky coordinates with unexpected "
                f"shape={sky.shape}; expected {pixels.shape}."
            )

        if not np.all(np.isfinite(sky)):
            raise PreprocessingError(
                "SCARLET FIELD pixel-to-sky conversion produced "
                "non-finite coordinates."
            )

        return [
            (float(coord[0]), float(coord[1]))
            for coord in sky
        ]

    # ------------------------------------------------------------------
    # FIELD source products
    # ------------------------------------------------------------------

    def _store_field_source_models(
        self,
        *,
        observation: Observation,
        sources: Sequence[Any],
        model_frame: Any,
        scarlet_observation: Any,
        source_image: NDData,
        channel: str,
        photometry_index: int,
        expected_channels: int,
        requested_centers: Sequence[tuple[float, float]],
        requested_sky_centers: Sequence[tuple[float, float]],
        requested_source_indices: Sequence[int],
        skipped: Sequence[int],
    ) -> None:
        """
        Store individual initialized source models.

        ``requested_source_indices`` maps each supplied source back to its
        index in the source-centre sequence received from the workflow. In the
        current architecture that sequence is the authoritative LocalField,
        not the complete detection catalogue. ``selected_index`` therefore
        indexes the LocalField source sequence, while ``source_index`` is
        retained for provenance and product naming. SCARLET returns only
        successfully initialized sources, so skipped indices are used to map
        successful source models back to their supplied positions.
        """
        if len(requested_sky_centers) != len(requested_centers):
            raise PreprocessingError(
                "SCARLET source-coordinate bookkeeping is inconsistent: "
                f"{len(requested_sky_centers)} sky centres for "
                f"{len(requested_centers)} pixel centres."
            )

        if len(requested_source_indices) != len(requested_centers):
            raise PreprocessingError(
                "SCARLET source-index bookkeeping is inconsistent: "
                f"{len(requested_source_indices)} indices for "
                f"{len(requested_centers)} selected centres."
            )

        skipped_set = set(skipped)

        successful_indices = [
            index
            for index in range(len(requested_centers))
            if index not in skipped_set
        ]

        if len(successful_indices) != len(sources):
            raise PreprocessingError(
                "SCARLET source initialization bookkeeping is "
                "inconsistent: "
                f"{len(sources)} sources returned for "
                f"{len(successful_indices)} successful indices."
            )

        for selected_index, source in zip(
            successful_indices,
            sources,
            strict=True,
        ):
            # ``selected_index`` indexes the capped/selected source arrays,
            # while ``source_index`` identifies the source in the original
            # full catalogue. Keep these coordinate/index spaces separate.
            source_index = int(requested_source_indices[selected_index])

            source_model = source.get_model(
                frame=model_frame,
            )

            source_cube = self._as_model_cube(
                source_model,
                expected_channels=expected_channels,
            )

            source_rendered_cube = self._render_observation_cube(
                scarlet_observation,
                source_cube,
                expected_shape=scarlet_observation.data.shape,
                description=f"SCARLET source {source_index} rendered cube",
            )

            source_image_data = source_rendered_cube[
                photometry_index
            ]

            self._validate_cube(
                source_cube,
                (
                    expected_channels,
                    source_image_data.shape[0],
                    source_image_data.shape[1],
                ),
                f"SCARLET source {source_index} model cube",
            )

            product_name = (
                f"{self._FIELD_SOURCE_PREFIX}"
                f"{source_index:04d}"
            )

            derived = self._create_derived_image(
                source_image,
                source_image_data,
                product=product_name,
                channel=channel,
                extra_metadata={
                    "scarlet_mode": "FIELD",
                    "scarlet_source_index": source_index,
                    # These arrays contain only the selected sources,
                    # so they must be indexed by ``selected_index``.
                    "scarlet_source_center_y": (
                        requested_centers[selected_index][0]
                    ),
                    "scarlet_source_center_x": (
                        requested_centers[selected_index][1]
                    ),
                    "scarlet_source_ra_deg": (
                        requested_sky_centers[selected_index][0]
                    ),
                    "scarlet_source_dec_deg": (
                        requested_sky_centers[selected_index][1]
                    ),
                    "scarlet_representation": "observation_space_psf_rendered",
                },
            )

            observation.add_image(
                product_name,
                ImageData(derived),
            )

    # ------------------------------------------------------------------
    # SCARLET construction
    # ------------------------------------------------------------------

    def _build_scarlet_observation(
        self,
        *,
        scarlet: Any,
        scarlet_input: ScarletInput,
        observation: Observation,
    ) -> Any:
        """
        Construct the SCARLET Observation.

        The canonical domain PSF is converted into SCARLET's ImagePSF.
        This preserves the distinction between:

            observation PSF
                = native/calibrated survey PSF

            model-frame PSF
                = internal reference PSF used for source modelling

        If the Observation has no PSF, SCARLET is allowed to construct an
        Observation without one for compatibility with existing non-PSF
        workflows. PSF-dependent production experiments should provide one.
        """
        psf_model = None

        if observation.psf is not None:
            psf_data = np.asarray(
                observation.psf.data,
                dtype=np.float64,
            )

            if psf_data.ndim != 2:
                raise PreprocessingError(
                    "Observation PSF must be two-dimensional; "
                    f"received shape={psf_data.shape}."
                )

            if psf_data.size == 0:
                raise PreprocessingError(
                    "Observation PSF contains no pixels."
                )

            if not np.all(
                np.isfinite(psf_data)
            ):
                raise PreprocessingError(
                    "Observation PSF contains non-finite values."
                )

            total_flux = float(
                np.sum(psf_data)
            )

            if not np.isfinite(total_flux) or total_flux <= 0.0:
                raise PreprocessingError(
                    "Observation PSF has invalid total flux: "
                    f"{total_flux!r}."
                )

            # ImagePSF normalises the image internally. Supplying a copy
            # prevents SCARLET from mutating the domain PSF array.
            psf_model = scarlet.ImagePSF(
                np.array(
                    psf_data,
                    dtype=np.float64,
                    copy=True,
                )
            )

            self._logger.info(
                "Using Observation PSF for SCARLET "
                "(model_type=%s, shape=%s, total_flux=%.12f).",
                observation.psf.model_type,
                psf_data.shape,
                total_flux,
            )

        else:
            self._logger.warning(
                "Observation contains no PSF. "
                "SCARLET Observation will use its default PSF behaviour."
            )

        return scarlet.Observation(
            data=scarlet_input.image_cube,
            channels=list(
                scarlet_input.channel_names,
            ),
            psf=psf_model,
            weights=scarlet_input.weights,
            wcs=scarlet_input.wcs,
        )

    def _build_model_frame(
        self,
        *,
        scarlet: Any,
        scarlet_input: ScarletInput,
    ) -> Any:
        """
        Construct the SCARLET model Frame.

        The model frame intentionally uses a narrower reference PSF than the
        observed survey PSF.

        The survey PSF belongs to the SCARLET Observation. SCARLET therefore
        models the underlying source at the model-frame resolution and
        renders/convolves it into the observed resolution through
        Observation.match().
        """
        model_psf = scarlet.GaussianPSF(
            sigma=0.8,
        )

        return scarlet.Frame(
            scarlet_input.image_cube.shape,
            channels=list(
                scarlet_input.channel_names,
            ),
            psf=model_psf,
            wcs=scarlet_input.wcs,
            dtype=scarlet_input.image_cube.dtype,
        )

    # ------------------------------------------------------------------
    # SCARLET fitting
    # ------------------------------------------------------------------

    def _fit_blend(
        self,
        blend: Any,
    ) -> tuple[Any, Any]:
        """
        Fit a SCARLET Blend and normalise the return contract.

        SCARLET 1.x returns ``(iterations, log_likelihood)`` from
        ``Blend.fit``. The values are retained for provenance/logging but
        are not required by the domain model.
        """
        try:
            result = blend.fit(
                self._max_iter,
                e_rel=self._e_rel,
            )

        except TypeError:
            # Compatibility with keyword-style invocations used by some
            # SCARLET-compatible implementations.
            result = blend.fit(
                max_iter=self._max_iter,
                e_rel=self._e_rel,
            )

        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0], result[1]

            if len(result) == 1:
                return result[0], None

        return result, None

    # ------------------------------------------------------------------
    # Model normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _as_model_cube(
        model: Any,
        *,
        expected_channels: int,
    ) -> np.ndarray:
        """
        Convert a SCARLET model into a channel-first 3-D cube.

        A single-channel 2-D model is promoted to a one-channel cube.
        """
        try:
            array = np.asarray(
                model,
                dtype=np.float64,
            )

        except (TypeError, ValueError, OverflowError) as exc:
            raise PreprocessingError(
                "SCARLET produced an invalid model."
            ) from exc

        if array.ndim == 2:
            array = array[
                np.newaxis,
                ...
            ]

        if array.ndim != 3:
            raise PreprocessingError(
                "SCARLET produced an unexpected model shape."
            )

        if array.shape[0] != expected_channels:
            raise PreprocessingError(
                "SCARLET produced "
                f"{array.shape[0]} channels; "
                f"expected {expected_channels}."
            )

        return array

    # ------------------------------------------------------------------
    # Observation-space rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_observation_cube(
        scarlet_observation: Any,
        model_cube: np.ndarray,
        *,
        expected_shape: tuple[int, ...],
        description: str,
    ) -> np.ndarray:
        """Render a model-frame cube into the SCARLET observation frame."""
        try:
            rendered = scarlet_observation.render(model_cube)
            rendered = np.asarray(rendered, dtype=np.float64)
        except Exception as exc:
            raise PreprocessingError(
                f"{description} could not be rendered into observation space."
            ) from exc

        if rendered.shape != expected_shape:
            raise PreprocessingError(
                f"{description} has shape {rendered.shape}; "
                f"expected {expected_shape}."
            )

        if not np.all(np.isfinite(rendered)):
            raise PreprocessingError(
                f"{description} contains non-finite values."
            )

        return rendered

    # ------------------------------------------------------------------
    # Channel handling
    # ------------------------------------------------------------------

    def _get_photometry_channel(
        self,
        channel_names: Sequence[str],
    ) -> str:
        """
        Return the channel whose reconstructed target becomes active.
        """
        if self._photometry_channel is not None:
            return self._photometry_channel

        if len(channel_names) == 1:
            return channel_names[0]

        raise PreprocessingError(
            "A photometry channel must be selected for "
            "multi-channel SCARLET processing."
        )

    # ------------------------------------------------------------------
    # Derived image creation
    # ------------------------------------------------------------------

    def _create_derived_image(
        self,
        source_image: NDData,
        data: np.ndarray,
        *,
        product: str,
        channel: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> NDData:
        """
        Create an NDData object for a SCARLET-derived image product.

        The source image's WCS, unit, mask and existing metadata are retained.
        """
        metadata: dict[str, Any] = {}

        if source_image.meta is not None:
            metadata.update(
                dict(source_image.meta)
            )

        metadata.update(
            {
                "processing_product": product,
                "scarlet_channel": channel,
                "scarlet_created_utc": datetime.now(
                    timezone.utc,
                ).isoformat(),
            }
        )

        if extra_metadata is not None:
            metadata.update(
                extra_metadata
            )

        mask = None

        if source_image.mask is not None:
            mask = np.array(
                source_image.mask,
                copy=True,
            )

        return NDData(
            data=np.asarray(
                data,
                dtype=np.float64,
            ),
            mask=mask,
            uncertainty=None,
            wcs=source_image.wcs,
            meta=metadata,
            unit=source_image.unit,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_cube(
        output: np.ndarray,
        expected_shape: tuple[int, ...],
        description: str,
    ) -> None:
        """
        Validate a SCARLET-derived image cube.
        """
        if output.shape != expected_shape:
            raise PreprocessingError(
                f"{description} has shape {output.shape}; "
                f"expected {expected_shape}."
            )

        if not np.all(
            np.isfinite(output)
        ):
            raise PreprocessingError(
                f"{description} contains non-finite values."
            )

    @staticmethod
    def _validate_observation(
        observation: Observation,
    ) -> None:
        """
        Validate the supplied Observation before processing.
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

    # ------------------------------------------------------------------
    # Lazy SCARLET import
    # ------------------------------------------------------------------

    def _load_scarlet(self) -> Any:
        """
        Import SCARLET lazily.

        Lazy import keeps SCARLET optional for workflows that do not require
        source separation.
        """
        try:
            import scarlet

        except Exception as exc:
            raise PreprocessingError(
                "SCARLET could not be imported. Ensure the compatible "
                "SCARLET implementation and its dependencies are "
                "installed."
            ) from exc

        return scarlet