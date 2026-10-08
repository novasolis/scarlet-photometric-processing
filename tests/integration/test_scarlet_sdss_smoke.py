"""
End-to-end SCARLET smoke test using a real SDSS observation.

This test verifies that a real SDSS image can pass through the project's
acquisition, domain-model construction, background subtraction, and actual
SCARLET preprocessing stages.

Unlike the unit tests in ``tests/preprocessing/test_scarlet.py``, this test
does not mock SCARLET and does not use a synthetic image.

The test is intentionally limited to the SCARLET preprocessing boundary.
Source detection, target-source resolution, aperture photometry, and workflow
evaluation are tested separately.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from acquisition.sdss import SDSSAcquirer
from config.experiment_config import create_experiment_config
from domain.enums import ExperimentType, SourceScope, Survey
from domain.sdss_builder import SDSSObservationBuilder
from preprocessing.background import BackgroundSubtractor
from preprocessing.scarlet import ScarletPreprocessor


LOGGER = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.scarlet
@pytest.mark.sdss
def test_scarlet_processes_real_sdss_observation() -> None:
    """
    Verify that SCARLET can process a real SDSS observation end to end.

    The test uses the project's configured M67 SDSS benchmark target and
    follows the production data path:

        ExperimentConfig
            ↓
        SDSSAcquirer
            ↓
        SDSSObservationBuilder
            ↓
        BackgroundSubtractor
            ↓
        ScarletPreprocessor
            ↓
        SCARLET target/model/residual products
    """

    # ------------------------------------------------------------------
    # 1. Build the standard SDSS experiment configuration.
    # ------------------------------------------------------------------
    config = create_experiment_config(
        survey=Survey.SDSS,
        experiment_type=ExperimentType.TARGET_SOURCE,
        source_scope=SourceScope.TARGET,
        target_name="m67_ybp_1303",
    )

    # ------------------------------------------------------------------
    # 2. Acquire a real SDSS observation.
    # ------------------------------------------------------------------
    acquirer = SDSSAcquirer(config)
    dataset = acquirer.acquire()

    assert dataset is not None
    assert len(dataset) > 0

    # ------------------------------------------------------------------
    # 3. Convert the SDSS FITS dataset into the project Observation
    #    domain model.
    # ------------------------------------------------------------------
    builder = SDSSObservationBuilder(config.target)
    observation = builder.build(dataset)

    assert observation is not None

    assert (
        observation.metadata.target_name
        == config.target.target_name
    )

    assert (
        observation.metadata.right_ascension
        == config.target.right_ascension
    )

    assert (
        observation.metadata.declination
        == config.target.declination
    )

    # ------------------------------------------------------------------
    # 4. Verify that the acquired observation contains usable image data.
    # ------------------------------------------------------------------
    raw_data = np.asarray(
        observation.active.image.data,
        dtype=np.float64,
    )

    assert raw_data.ndim == 2
    assert raw_data.size > 0
    assert np.isfinite(raw_data).any()

    # ------------------------------------------------------------------
    # 5. Apply the common background-subtraction stage.
    #
    # SCARLET requires the common background-subtracted observation as
    # its input.
    # ------------------------------------------------------------------
    background_subtractor = BackgroundSubtractor(
        config.preprocessing,
    )

    observation = background_subtractor.process(
        observation,
    )

    assert observation.processing_state.background_removed
    assert "background_removed" in observation.images

    background_removed_data = np.asarray(
        observation.active.image.data,
        dtype=np.float64,
    )

    assert background_removed_data.ndim == 2
    assert background_removed_data.shape == raw_data.shape
    assert np.isfinite(background_removed_data).any()

    # ------------------------------------------------------------------
    # 6. Validate the configured target against the observation WCS.
    #
    # This diagnostic intentionally occurs BEFORE SCARLET processing.
    # If SCARLET initialisation fails, these values are still available
    # to diagnose target/WCS/data alignment.
    # ------------------------------------------------------------------
    right_ascension = config.target.right_ascension
    declination = config.target.declination

    assert right_ascension is not None
    assert declination is not None

    target_coord = SkyCoord(
        float(right_ascension),
        float(declination),
        unit=u.deg,
        frame="icrs",
    )

    # The project metadata exposes an Astropy WCS at runtime.  Astropy's
    # WCS typing currently produces an incompatible overload diagnostic
    # with the installed Python/Astropy/Pylance combination, so isolate
    # that typing boundary here rather than weakening the rest of the test.
    wcs: Any = cast(Any, observation.metadata.wcs)

    x_raw, y_raw = wcs.world_to_pixel(target_coord)

    x = float(x_raw)
    y = float(y_raw)

    LOGGER.info(
        "SDSS target sky coordinate: RA=%s deg, Dec=%s deg.",
        right_ascension,
        declination,
    )

    LOGGER.info(
        "SDSS target pixel: x=%s, y=%s.",
        x,
        y,
    )

    # WCS must return finite pixel coordinates.
    assert np.isfinite(x)
    assert np.isfinite(y)

    target_x = int(round(x))
    target_y = int(round(y))

    LOGGER.info(
        "Rounded target pixel: x=%d, y=%d.",
        target_x,
        target_y,
    )

    # ------------------------------------------------------------------
    # 7. Check that the target falls inside the acquired image.
    # ------------------------------------------------------------------
    height, width = background_removed_data.shape

    assert 0 <= target_x < width, (
        f"Target x pixel {target_x} is outside image width {width}."
    )

    assert 0 <= target_y < height, (
        f"Target y pixel {target_y} is outside image height {height}."
    )

    target_value = float(
        background_removed_data[target_y, target_x]
    )

    LOGGER.info(
        "Target pixel value: %s.",
        target_value,
    )

    # ------------------------------------------------------------------
    # 8. Inspect a small neighbourhood around the configured target.
    #
    # NaNs are excluded from the statistics so the diagnostic itself
    # does not generate RuntimeWarnings or obscure the SCARLET failure.
    # ------------------------------------------------------------------
    window = np.asarray(
        background_removed_data[
            max(0, target_y - 5):min(height, target_y + 6),
            max(0, target_x - 5):min(width, target_x + 6),
        ],
        dtype=np.float64,
    )

    finite_window = window[np.isfinite(window)]

    assert finite_window.size > 0

    LOGGER.info(
        "Target neighbourhood: min=%s, max=%s, mean=%s, std=%s.",
        float(np.min(finite_window)),
        float(np.max(finite_window)),
        float(np.mean(finite_window)),
        float(np.std(finite_window)),
    )

    # ------------------------------------------------------------------
    # 9. Run the actual SCARLET integration.
    #
    # No mock is used here. ScarletPreprocessor imports the local
    # SCARLET source checkout and performs a real model fit.
    # ------------------------------------------------------------------
    preprocessor = ScarletPreprocessor()

    observation = preprocessor.process(
        observation,
    )

    # ------------------------------------------------------------------
    # 10. Verify that SCARLET produced all expected derived products.
    # ------------------------------------------------------------------
    assert "scarlet_target" in observation.images
    assert "scarlet_model" in observation.images
    assert "scarlet_residual" in observation.images

    target_data = np.asarray(
        observation.images["scarlet_target"].image.data,
        dtype=np.float64,
    )

    model_data = np.asarray(
        observation.images["scarlet_model"].image.data,
        dtype=np.float64,
    )

    residual_data = np.asarray(
        observation.images["scarlet_residual"].image.data,
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # 11. Verify valid image geometry.
    # ------------------------------------------------------------------
    assert target_data.ndim == 2
    assert model_data.ndim == 2
    assert residual_data.ndim == 2

    assert target_data.shape == raw_data.shape
    assert model_data.shape == raw_data.shape
    assert residual_data.shape == raw_data.shape

    # ------------------------------------------------------------------
    # 12. Verify numerically usable SCARLET products.
    # ------------------------------------------------------------------
    assert np.isfinite(target_data).any()
    assert np.isfinite(model_data).any()
    assert np.isfinite(residual_data).any()

    # The reconstructed target should contain actual signal rather than
    # being entirely zero.
    assert not np.allclose(
        target_data,
        0.0,
    )

    # ------------------------------------------------------------------
    # 13. Verify that the reconstructed target became the active image.
    # ------------------------------------------------------------------
    assert (
        observation.active
        is observation.images["scarlet_target"]
    )

    # ------------------------------------------------------------------
    # 14. Verify successful SCARLET processing state.
    # ------------------------------------------------------------------
    assert observation.processing_state.scarlet_applied