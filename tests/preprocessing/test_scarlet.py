"""
Unit tests for the SCARLET preprocessing stage.

These tests verify the contract enforced by ScarletPreprocessor without
depending on a real SCARLET optimisation.

Real end-to-end SCARLET execution against astronomical data is covered by
the integration tests under:

    tests/integration/test_scarlet_sdss_smoke.py

The unit suite therefore uses a lightweight fake SCARLET implementation.
This keeps the tests deterministic while exercising the complete project
boundary between ScarletPreprocessor and the SCARLET API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import NDData
from astropy.wcs import WCS

from core.exceptions import PreprocessingError
from domain.enums import ObservationSource, Survey
from domain.models import (
    ImageData,
    Observation,
    ObservationMetadata,
    ObservationProvenance,
    PSFData,
)
from preprocessing.scarlet import ScarletPreprocessor
from preprocessing.scarlet_adapter import ScarletInput


# ============================================================================
# Test data helpers
# ============================================================================


def _make_wcs() -> WCS:
    """Create a minimal celestial WCS suitable for unit tests."""
    wcs = WCS(naxis=2)

    wcs.wcs.crpix = [5.0, 5.0]
    wcs.wcs.cdelt = [-0.0001, 0.0001]
    wcs.wcs.crval = [100.0, -30.0]
    wcs.wcs.ctype = [
        "RA---TAN",
        "DEC--TAN",
    ]

    return wcs


def _make_psf(
    *,
    shape: tuple[int, int] = (5, 5),
) -> PSFData:
    """
    Create a deterministic, normalised PSFData object.

    The PSF is intentionally simple because these tests validate the
    SCARLET/domain boundary rather than the survey-specific PSF reconstruction.
    """
    psf = np.zeros(
        shape,
        dtype=np.float64,
    )

    centre_y = shape[0] // 2
    centre_x = shape[1] // 2

    psf[centre_y, centre_x] = 1.0

    return PSFData(
        data=psf,
        model_type="test_psf",
        normalised=True,
        x=5.0,
        y=5.0,
        pixel_scale_arcsec=0.396,
        source_path="test_psf.fits",
        metadata={
            "survey": "TEST",
        },
    )


def _make_image(
    *,
    value: float = 1.0,
    shape: tuple[int, int] = (10, 10),
    meta: dict[str, Any] | None = None,
    mask: np.ndarray | None = None,
    unit: u.UnitBase | None = None,
) -> ImageData:
    """
    Create a domain ImageData object containing a deterministic test image.

    Parameters
    ----------
    value
        Baseline pixel value.
    shape
        Two-dimensional image shape.
    meta
        Optional NDData metadata.
    mask
        Optional image mask.
    unit
        Optional physical unit attached to the NDData object.
    """
    data = np.full(
        shape,
        value,
        dtype=np.float64,
    )

    # Introduce a small spatially varying signal so that the image has
    # meaningful structure for fake SCARLET model tests.
    data[
        shape[0] // 2,
        shape[1] // 2,
    ] += 100.0

    nddata = NDData(
        data=data,
        wcs=_make_wcs(),
        meta=meta,
        mask=mask,
        unit=unit,
    )

    return ImageData(nddata)


def _make_observation(
    *,
    background_removed: bool = True,
    target_ra: float | None = 100.0,
    target_dec: float | None = -30.0,
    psf: PSFData | None = None,
) -> Observation:
    """Create a minimal valid Observation for SCARLET unit tests."""
    image = _make_image()

    observation = Observation(
        metadata=ObservationMetadata(
            target_name="Test Target",
            observation_time=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            observation_time_scale="utc",
            observation_time_format="datetime",
            right_ascension=target_ra,
            declination=target_dec,
            wcs=_make_wcs(),
        ),
        provenance=ObservationProvenance(
            source=ObservationSource.SPACE_BASED,
            survey=Survey.TESS,
            dataset_identifier="test-dataset",
            filename="test.fits",
            download_timestamp=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        ),
        original=image,
        active=image,
        images={
            "raw": image,
        },
        psf=(
            _make_psf()
            if psf is None
            else psf
        ),
    )

    observation.processing_state.background_removed = background_removed

    return observation


def _make_scarlet_input(
    observation: Observation,
    *,
    channel_names: tuple[str, ...] = ("science",),
) -> ScarletInput:
    """
    Construct a valid ScarletInput directly.

    The real adapter is tested independently in
    tests/preprocessing/test_scarlet_adapter.py.
    """
    channel_images: list[ImageData] = []

    for index, _channel_name in enumerate(channel_names):
        channel_images.append(
            _make_image(
                value=float(index + 1),
            )
        )

    image_cube = np.stack(
        [
            np.asarray(
                image.image.data,
                dtype=np.float64,
            )
            for image in channel_images
        ],
        axis=0,
    )

    weights = np.ones_like(
        image_cube,
        dtype=np.float64,
    )

    return ScarletInput(
        image_cube=image_cube,
        weights=weights,
        channel_names=channel_names,
        target_sky_coord=(
            float(
                observation.metadata.right_ascension
                if observation.metadata.right_ascension is not None
                else 100.0
            ),
            float(
                observation.metadata.declination
                if observation.metadata.declination is not None
                else -30.0
            ),
        ),
        wcs=observation.metadata.wcs,
        channel_images=tuple(channel_images),
    )


def _image_data(
    observation: Observation,
    name: str,
) -> ImageData:
    """
    Return a named ImageData object with explicit runtime validation.

    This helper also gives Pylance a concrete ImageData return type instead
    of repeatedly indexing a broadly typed mapping.
    """
    images: Any = cast(
        Any,
        observation.images,
    )

    image = images.get(name)

    assert image is not None
    assert isinstance(
        image,
        ImageData,
    )

    return image


def _image_array(
    observation: Observation,
    name: str,
) -> np.ndarray:
    """Return a named image as an explicit float64 NumPy array."""
    image_data = _image_data(
        observation,
        name,
    )

    data: Any = image_data.image.data

    assert data is not None

    return np.asarray(
        data,
        dtype=np.float64,
    )


def _metadata(
    observation: Observation,
    name: str,
) -> dict[str, Any]:
    """
    Return image metadata with an explicit non-None runtime check.
    """
    image = _image_data(
        observation,
        name,
    )

    metadata: Any = image.image.meta

    assert metadata is not None

    return dict(metadata)


# ============================================================================
# Fake SCARLET implementation
# ============================================================================
class _FakeScarletGaussianPSF:
    """Minimal fake for scarlet.GaussianPSF used by the model-frame tests."""

    def __init__(
        self,
        sigma: float,
        integrate: bool = True,
        boxsize: int | None = None,
    ) -> None:
        self.sigma = float(sigma)
        self.integrate = integrate
        self.boxsize = boxsize


class _FakeScarletImagePSF:
    """Minimal fake for scarlet.ImagePSF."""

    def __init__(
        self,
        image: np.ndarray,
    ) -> None:
        array = np.asarray(
            image,
            dtype=np.float64,
        )

        if array.ndim == 2:
            array = array[np.newaxis, ...]

        if array.ndim != 3:
            raise ValueError(
                "Fake ImagePSF requires a 2-D or 3-D PSF array."
            )

        total = float(np.sum(array))

        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(
                "Fake ImagePSF requires a finite, positive PSF."
            )

        self.image = array / total
        self.bbox = (
            (0, self.image.shape[-2]),
            (0, self.image.shape[-1]),
        )

    def get_model(
        self,
        *,
        offset: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return the normalised PSF model."""

        del offset

        return self.image.copy()

class _FakeScarletObservation:
    """Minimal fake for scarlet.Observation."""

    def __init__(
        self,
        *,
        data: np.ndarray,
        channels: list[str],
        weights: np.ndarray,
        wcs: WCS | None,
        psf: Any = None,
    ) -> None:
        self.images = data
        self.channels = channels
        self.weights = weights
        self.wcs = wcs
        self.psf = psf
        self.matched_frame: Any = None

    def match(
        self,
        frame: Any,
    ) -> None:
        self.matched_frame = frame


class _FakeScarletFrame:
    """Minimal fake for scarlet.Frame."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        channels: list[str],
        wcs: WCS | None,
        dtype: Any,
        psf: Any = None,
    ) -> None:
        self.shape = shape
        self.channels = channels
        self.wcs = wcs
        self.dtype = dtype
        self.psf = psf


class _FakeScarletSource:
    """Minimal fake for scarlet.ExtendedSource."""

    def __init__(
        self,
        frame: Any,
        target_sky_coord: tuple[float, float],
        observation: Any,
        *,
        K: int,
        thresh: float,
    ) -> None:
        self.frame = frame
        self.target_sky_coord = target_sky_coord
        self.observation = observation
        self.K = K
        self.thresh = thresh

        channels = len(
            frame.channels,
        )

        height = frame.shape[-2]
        width = frame.shape[-1]

        self.model = np.zeros(
            (
                channels,
                height,
                width,
            ),
            dtype=np.float64,
        )

        # Deterministic reconstructed target signal.
        for index in range(channels):
            self.model[
                index,
                height // 2,
                width // 2,
            ] = float(
                index + 10,
            )

    def get_model(
        self,
        *,
        frame: Any,
    ) -> np.ndarray:
        return self.model.copy()


class _FakeScarletBlend:
    """Minimal fake for scarlet.Blend."""

    def __init__(
        self,
        sources: list[Any],
        observation: Any,
    ) -> None:
        self.sources = sources
        self.observation = observation
        self.fit_calls: list[dict[str, Any]] = []

    def fit(
        self,
        *,
        max_iter: int,
        e_rel: float,
    ) -> None:
        self.fit_calls.append(
            {
                "max_iter": max_iter,
                "e_rel": e_rel,
            }
        )

    def get_model(
        self,
        *,
        frame: Any,
    ) -> np.ndarray:
        channels = len(
            frame.channels,
        )

        height = frame.shape[-2]
        width = frame.shape[-1]

        model = np.zeros(
            (
                channels,
                height,
                width,
            ),
            dtype=np.float64,
        )

        for index in range(channels):
            model[
                index,
                height // 2,
                width // 2,
            ] = float(
                index + 20,
            )

        return model


def _make_fake_scarlet() -> SimpleNamespace:
    """Return a fake SCARLET module compatible with ScarletPreprocessor."""
    return SimpleNamespace(
        ImagePSF=_FakeScarletImagePSF,
        GaussianPSF=_FakeScarletGaussianPSF,
        Observation=_FakeScarletObservation,
        Frame=_FakeScarletFrame,
        ExtendedSource=_FakeScarletSource,
        Blend=_FakeScarletBlend,
    )


# ============================================================================
# Constructor validation
# ============================================================================


@pytest.mark.parametrize(
    "max_iter",
    [
        0,
        -1,
    ],
)
def test_scarlet_rejects_non_positive_max_iter(
    max_iter: int,
) -> None:
    """SCARLET optimisation iterations must be positive."""
    with pytest.raises(
        ValueError,
        match="max_iter must be greater than zero",
    ):
        ScarletPreprocessor(
            max_iter=max_iter,
        )


@pytest.mark.parametrize(
    "e_rel",
    [
        0.0,
        -1.0,
    ],
)
def test_scarlet_rejects_non_positive_e_rel(
    e_rel: float,
) -> None:
    """SCARLET convergence tolerance must be positive."""
    with pytest.raises(
        ValueError,
        match="e_rel must be greater than zero",
    ):
        ScarletPreprocessor(
            e_rel=e_rel,
        )


def test_scarlet_rejects_empty_channel_names() -> None:
    """Explicit SCARLET channel configuration cannot be empty."""
    with pytest.raises(
        ValueError,
        match="channel_names cannot be empty",
    ):
        ScarletPreprocessor(
            channel_names=(),
        )


def test_scarlet_rejects_duplicate_channel_names() -> None:
    """SCARLET channels must be unique."""
    with pytest.raises(
        ValueError,
        match="channel_names must be unique",
    ):
        ScarletPreprocessor(
            channel_names=(
                "g",
                "g",
            ),
        )


def test_scarlet_requires_photometry_channel_for_multiple_channels() -> None:
    """Multi-channel processing requires an explicit photometry channel."""
    with pytest.raises(
        ValueError,
        match="photometry_channel is required",
    ):
        ScarletPreprocessor(
            channel_names=(
                "g",
                "r",
            ),
        )


def test_scarlet_rejects_unknown_photometry_channel() -> None:
    """The selected photometry channel must exist in channel_names."""
    with pytest.raises(
        ValueError,
        match="photometry_channel must be one of channel_names",
    ):
        ScarletPreprocessor(
            channel_names=(
                "g",
                "r",
            ),
            photometry_channel="i",
        )


def test_scarlet_accepts_single_channel_without_photometry_channel() -> None:
    """A single channel can implicitly serve as the photometry channel."""
    preprocessor = ScarletPreprocessor(
        channel_names=("science",),
    )

    assert preprocessor._channel_names == (
        "science",
    )

    assert preprocessor._photometry_channel is None


def _patch_adapter_prepare(
    monkeypatch: pytest.MonkeyPatch,
    preprocessor: ScarletPreprocessor,
    scarlet_input: ScarletInput,
) -> None:
    """Patch the adapter using the current ScarletPreprocessor contract."""

    def _prepare(
        _observation: Observation,
        *,
        source_centers: Any = None,
    ) -> ScarletInput:
        del source_centers
        return scarlet_input

    monkeypatch.setattr(
        preprocessor._adapter,
        "prepare",
        _prepare,
    )


# ============================================================================
# Processing preconditions
# ============================================================================


def test_scarlet_requires_background_subtraction() -> None:
    """SCARLET must not run before background subtraction."""
    observation = _make_observation(
        background_removed=False,
    )

    preprocessor = ScarletPreprocessor()

    with pytest.raises(
        PreprocessingError,
        match="background-subtracted",
    ):
        preprocessor.process(
            observation,
        )


def test_scarlet_rejects_already_processed_observation() -> None:
    """SCARLET must not silently execute twice."""
    observation = _make_observation()

    observation.processing_state.scarlet_applied = True

    preprocessor = ScarletPreprocessor()

    with pytest.raises(
        PreprocessingError,
        match="already been completed",
    ):
        preprocessor.process(
            observation,
        )


def test_scarlet_requires_right_ascension() -> None:
    """SCARLET requires a target right ascension."""
    observation = _make_observation(
        target_ra=None,
    )

    preprocessor = ScarletPreprocessor()

    with pytest.raises(
        PreprocessingError,
        match="right ascension",
    ):
        preprocessor._adapter.prepare(
            observation,
        )


def test_scarlet_requires_declination() -> None:
    """SCARLET requires a target declination."""
    observation = _make_observation(
        target_dec=None,
    )

    preprocessor = ScarletPreprocessor()

    with pytest.raises(
        PreprocessingError,
        match="declination",
    ):
        preprocessor._adapter.prepare(
            observation,
        )


def test_scarlet_rejects_non_finite_right_ascension() -> None:
    """SCARLET rejects non-finite target coordinates."""
    observation = _make_observation(
        target_ra=float("nan"),
    )

    preprocessor = ScarletPreprocessor()

    adapter = preprocessor._adapter

    with pytest.raises(
        PreprocessingError,
        match="not finite",
    ):
        adapter.prepare(
            observation,
        )


def test_scarlet_rejects_non_finite_declination() -> None:
    """SCARLET rejects non-finite target declination."""
    observation = _make_observation(
        target_dec=float("inf"),
    )

    preprocessor = ScarletPreprocessor()

    adapter = preprocessor._adapter

    with pytest.raises(
        PreprocessingError,
        match="not finite",
    ):
        adapter.prepare(
            observation,
        )


# ============================================================================
# Model cube conversion
# ============================================================================


def test_as_model_cube_promotes_two_dimensional_model() -> None:
    """A single-channel 2-D model becomes a one-channel cube."""
    model = np.ones(
        (10, 10),
        dtype=np.float64,
    )

    result = ScarletPreprocessor._as_model_cube(
        model,
        expected_channels=1,
    )

    assert result.shape == (
        1,
        10,
        10,
    )

    np.testing.assert_allclose(
        result,
        np.asarray(
            model,
            dtype=np.float64,
        )[np.newaxis, ...],
    )


def test_as_model_cube_accepts_three_dimensional_model() -> None:
    """A correctly shaped multi-channel model is retained."""
    model = np.arange(
        200,
        dtype=np.float64,
    ).reshape(
        2,
        10,
        10,
    )

    result = ScarletPreprocessor._as_model_cube(
        model,
        expected_channels=2,
    )

    assert result.shape == model.shape

    np.testing.assert_allclose(
        np.asarray(
            result,
            dtype=np.float64,
        ),
        model,
    )


def test_as_model_cube_rejects_invalid_dimension() -> None:
    """SCARLET models must be 2-D or 3-D."""
    model = np.ones(
        (10,),
        dtype=np.float64,
    )

    with pytest.raises(
        PreprocessingError,
        match="unexpected model shape",
    ):
        ScarletPreprocessor._as_model_cube(
            model,
            expected_channels=1,
        )


def test_as_model_cube_rejects_wrong_channel_count() -> None:
    """SCARLET model channel count must match the input."""
    model = np.ones(
        (
            2,
            10,
            10,
        ),
        dtype=np.float64,
    )

    with pytest.raises(
        PreprocessingError,
        match="expected 1",
    ):
        ScarletPreprocessor._as_model_cube(
            model,
            expected_channels=1,
        )


def test_as_model_cube_rejects_invalid_model_values() -> None:
    """Non-finite model values are rejected by cube validation."""
    model = np.ones(
        (
            1,
            10,
            10,
        ),
        dtype=np.float64,
    )

    model[
        0,
        5,
        5,
    ] = np.nan

    with pytest.raises(
        PreprocessingError,
        match="contains non-finite values",
    ):
        ScarletPreprocessor._validate_cube(
            model,
            model.shape,
            "SCARLET model cube",
        )


def test_validate_cube_rejects_wrong_shape() -> None:
    """Derived SCARLET products must preserve the expected geometry."""
    output = np.ones(
        (
            1,
            10,
            10,
        ),
        dtype=np.float64,
    )

    with pytest.raises(
        PreprocessingError,
        match="expected",
    ):
        ScarletPreprocessor._validate_cube(
            output,
            (
                1,
                8,
                8,
            ),
            "SCARLET target cube",
        )


# ============================================================================
# Photometry channel selection
# ============================================================================


def test_single_channel_automatically_selects_channel() -> None:
    """A single channel is implicitly selected for photometry."""
    preprocessor = ScarletPreprocessor()

    result = preprocessor._get_photometry_channel(
        ("science",),
    )

    assert result == "science"


def test_explicit_photometry_channel_is_selected() -> None:
    """An explicitly configured photometry channel is returned."""
    preprocessor = ScarletPreprocessor(
        channel_names=(
            "g",
            "r",
        ),
        photometry_channel="r",
    )

    result = preprocessor._get_photometry_channel(
        (
            "g",
            "r",
        ),
    )

    assert result == "r"


def test_multi_channel_processing_requires_photometry_selection() -> None:
    """Multiple channels cannot be processed without a selection."""
    preprocessor = ScarletPreprocessor(
        channel_names=(
            "g",
            "r",
        ),
        photometry_channel="r",
    )

    # Deliberately call the helper with multiple channels after removing
    # the configured selection. This is a controlled invalid runtime state.
    preprocessor._photometry_channel = None

    with pytest.raises(
        PreprocessingError,
        match="photometry channel must be selected",
    ):
        preprocessor._get_photometry_channel(
            (
                "g",
                "r",
            ),
        )


# ============================================================================
# SCARLET object construction
# ============================================================================


def test_build_scarlet_observation_passes_adapter_output() -> None:
    """SCARLET Observation receives adapter data, weights, WCS and PSF."""
    observation = _make_observation()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    fake_scarlet = _make_fake_scarlet()

    preprocessor = ScarletPreprocessor()

    result = preprocessor._build_scarlet_observation(
        scarlet=fake_scarlet,
        scarlet_input=scarlet_input,
        observation=observation,
    )

    assert isinstance(
        result,
        _FakeScarletObservation,
    )

    np.testing.assert_allclose(
        np.asarray(
            result.images,
            dtype=np.float64,
        ),
        np.asarray(
            scarlet_input.image_cube,
            dtype=np.float64,
        ),
    )

    np.testing.assert_allclose(
        np.asarray(
            result.weights,
            dtype=np.float64,
        ),
        np.asarray(
            scarlet_input.weights,
            dtype=np.float64,
        ),
    )

    assert result.channels == list(
        scarlet_input.channel_names,
    )

    assert result.wcs is scarlet_input.wcs

    # The test Observation contains a canonical PSF. The current production
    # preprocessor converts/passes that PSF to the SCARLET Observation.
    assert result.psf is not None


# ============================================================================
# Derived image construction
# ============================================================================


def test_create_derived_image_preserves_wcs() -> None:
    """Derived SCARLET images preserve the source WCS."""
    source = _make_image(
        meta={
            "TEST_KEY": "test-value",
        },
    )

    data = np.full(
        (
            10,
            10,
        ),
        5.0,
        dtype=np.float64,
    )

    preprocessor = ScarletPreprocessor()

    result = preprocessor._create_derived_image(
        source.image,
        data,
        product="scarlet_target",
        channel="science",
    )

    assert result.wcs is source.image.wcs


def test_create_derived_image_preserves_unit() -> None:
    """Derived SCARLET images preserve the source unit."""
    source = _make_image(
        unit=None,
    )

    data = np.ones(
        (
            10,
            10,
        ),
        dtype=np.float64,
    )

    preprocessor = ScarletPreprocessor()

    result = preprocessor._create_derived_image(
        source.image,
        data,
        product="scarlet_target",
        channel="science",
    )

    assert result.unit is source.image.unit


def test_create_derived_image_preserves_metadata() -> None:
    """Existing source metadata is copied into derived products."""
    source = _make_image(
        meta={
            "TEST_KEY": "test-value",
        },
    )

    data = np.ones(
        (
            10,
            10,
        ),
        dtype=np.float64,
    )

    preprocessor = ScarletPreprocessor()

    result = preprocessor._create_derived_image(
        source.image,
        data,
        product="scarlet_target",
        channel="science",
    )

    metadata: Any = result.meta

    assert metadata is not None

    assert metadata["TEST_KEY"] == "test-value"


def test_create_derived_image_adds_scarlet_metadata() -> None:
    """Derived products receive explicit SCARLET provenance metadata."""
    source = _make_image()

    data = np.ones(
        (
            10,
            10,
        ),
        dtype=np.float64,
    )

    preprocessor = ScarletPreprocessor()

    result = preprocessor._create_derived_image(
        source.image,
        data,
        product="scarlet_target",
        channel="science",
    )

    metadata: Any = result.meta

    assert metadata is not None

    assert metadata["processing_product"] == "scarlet_target"
    assert metadata["scarlet_channel"] == "science"

    assert "scarlet_created_utc" in metadata


def test_create_derived_image_returns_float64() -> None:
    """Derived SCARLET products are stored as float64 arrays."""
    source = _make_image()

    data = np.ones(
        (
            10,
            10,
        ),
        dtype=np.float32,
    )

    preprocessor = ScarletPreprocessor()

    result = preprocessor._create_derived_image(
        source.image,
        data,
        product="scarlet_target",
        channel="science",
    )

    assert result.data is not None

    result_data: Any = result.data

    assert np.asarray(
        result_data,
    ).dtype == np.float64


def test_create_derived_image_copies_mask() -> None:
    """A source image mask is copied rather than shared."""
    mask = np.zeros(
        (
            10,
            10,
        ),
        dtype=bool,
    )

    mask[
        0,
        0,
    ] = True

    source = _make_image(
        mask=mask,
    )

    data = np.ones(
        (
            10,
            10,
        ),
        dtype=np.float64,
    )

    preprocessor = ScarletPreprocessor()

    result = preprocessor._create_derived_image(
        source.image,
        data,
        product="scarlet_target",
        channel="science",
    )

    assert result.mask is not None

    result_mask = np.asarray(
        result.mask,
        dtype=bool,
    )

    np.testing.assert_array_equal(
        result_mask,
        mask,
    )

    # Confirm that the mask is not the same mutable object.
    assert result.mask is not source.image.mask


# ============================================================================
# Full process() using fake SCARLET
# ============================================================================


def test_scarlet_process_creates_expected_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful processing creates target, model and residual products."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    images: Any = result.images

    assert "scarlet_target" in images
    assert "scarlet_model" in images
    assert "scarlet_residual" in images


def test_scarlet_process_returns_same_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Processing mutates and returns the supplied Observation instance."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    assert result is observation


def test_scarlet_process_sets_processing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful SCARLET processing marks the Observation accordingly."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    assert result.processing_state.scarlet_applied is True


def test_scarlet_process_makes_target_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reconstructed target image becomes active for photometry."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    target = _image_data(
        result,
        "scarlet_target",
    )

    assert result.active is target


def test_scarlet_process_target_has_expected_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reconstructed target preserves the input image geometry."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    target = _image_array(
        result,
        "scarlet_target",
    )

    assert target.shape == (
        10,
        10,
    )

    assert target.ndim == 2


def test_scarlet_process_model_has_expected_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full reconstructed model preserves the input image geometry."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    model = _image_array(
        result,
        "scarlet_model",
    )

    assert model.shape == (
        10,
        10,
    )

    assert model.ndim == 2


def test_scarlet_process_residual_equals_input_minus_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The residual product is the input image minus the full model."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    target_input = np.asarray(
        scarlet_input.image_cube[0],
        dtype=np.float64,
    )

    model = _image_array(
        result,
        "scarlet_model",
    )

    residual = _image_array(
        result,
        "scarlet_residual",
    )

    expected = target_input - model

    np.testing.assert_allclose(
        np.asarray(
            residual,
            dtype=np.float64,
        ),
        np.asarray(
            expected,
            dtype=np.float64,
        ),
    )


def test_scarlet_process_products_are_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful SCARLET products contain only finite values."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    for name in (
        "scarlet_target",
        "scarlet_model",
        "scarlet_residual",
    ):
        data = _image_array(
            result,
            name,
        )

        assert np.all(
            np.isfinite(data),
        )


def test_scarlet_process_preserves_original_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCARLET processing does not replace the original raw image."""
    observation = _make_observation()

    original = np.asarray(
        observation.original.image.data,
        dtype=np.float64,
    ).copy()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    preprocessor.process(
        observation,
    )

    current = np.asarray(
        observation.original.image.data,
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        current,
        original,
    )


def test_scarlet_process_adds_correct_product_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each derived product records its SCARLET product and channel."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    expected_products = {
        "scarlet_target": "scarlet_target",
        "scarlet_model": "scarlet_model",
        "scarlet_residual": "scarlet_residual",
    }

    for image_name, product_name in expected_products.items():
        metadata = _metadata(
            result,
            image_name,
        )

        assert metadata["processing_product"] == product_name
        assert metadata["scarlet_channel"] == "science"
        assert "scarlet_created_utc" in metadata


def test_scarlet_process_passes_configured_fit_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured max_iter and e_rel reach Blend.fit()."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor(
        max_iter=37,
        e_rel=2.5e-5,
    )

    scarlet_input = _make_scarlet_input(
        observation,
    )

    fake_scarlet = _make_fake_scarlet()

    blend_instance: _FakeScarletBlend | None = None

    original_blend = fake_scarlet.Blend

    def _blend_factory(
        sources: list[Any],
        scarlet_observation: Any,
    ) -> _FakeScarletBlend:
        nonlocal blend_instance

        created_blend = original_blend(
            sources,
            scarlet_observation,
        )

        blend_instance = created_blend

        return created_blend

    fake_scarlet.Blend = _blend_factory

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: fake_scarlet,
    )

    preprocessor.process(
        observation,
    )

    assert blend_instance is not None

    assert blend_instance.fit_calls == [
        {
            "max_iter": 37,
            "e_rel": 2.5e-5,
        }
    ]


def test_scarlet_process_passes_target_coordinates_to_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The target sky coordinate from ScarletInput reaches ExtendedSource."""
    observation = _make_observation(
        target_ra=132.73609708,
        target_dec=11.83185478,
    )

    preprocessor = ScarletPreprocessor()

    scarlet_input = ScarletInput(
        image_cube=np.ones(
            (
                1,
                10,
                10,
            ),
            dtype=np.float64,
        ),
        weights=np.ones(
            (
                1,
                10,
                10,
            ),
            dtype=np.float64,
        ),
        channel_names=("science",),
        target_sky_coord=(
            132.73609708,
            11.83185478,
        ),
        wcs=observation.metadata.wcs,
        channel_images=(
            _make_image(),
        ),
    )

    fake_scarlet = _make_fake_scarlet()

    captured: dict[str, Any] = {}

    original_source = fake_scarlet.ExtendedSource

    def _source_factory(
        frame: Any,
        target_sky_coord: tuple[float, float],
        scarlet_observation: Any,
        *,
        K: int,
        thresh: float,
    ) -> Any:
        captured["target_sky_coord"] = target_sky_coord
        captured["K"] = K
        captured["thresh"] = thresh

        return original_source(
            frame,
            target_sky_coord,
            scarlet_observation,
            K=K,
            thresh=thresh,
        )

    fake_scarlet.ExtendedSource = _source_factory

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: fake_scarlet,
    )

    preprocessor.process(
        observation,
    )

    assert captured["target_sky_coord"] == (
        132.73609708,
        11.83185478,
    )

    assert captured["K"] == 1
    assert captured["thresh"] == 1.0


# ============================================================================
# Multi-channel processing
# ============================================================================


def test_scarlet_processes_selected_multi_channel_photometry_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured photometry channel becomes the active reconstruction."""
    observation = _make_observation()

    g_image = _make_image(
        value=1.0,
    )

    r_image = _make_image(
        value=2.0,
    )

    observation.images["g"] = g_image
    observation.images["r"] = r_image

    preprocessor = ScarletPreprocessor(
        channel_names=(
            "g",
            "r",
        ),
        photometry_channel="r",
    )

    scarlet_input = ScarletInput(
        image_cube=np.stack(
            [
                np.asarray(
                    g_image.image.data,
                    dtype=np.float64,
                ),
                np.asarray(
                    r_image.image.data,
                    dtype=np.float64,
                ),
            ],
            axis=0,
        ),
        weights=np.ones(
            (
                2,
                10,
                10,
            ),
            dtype=np.float64,
        ),
        channel_names=(
            "g",
            "r",
        ),
        target_sky_coord=(
            100.0,
            -30.0,
        ),
        wcs=observation.metadata.wcs,
        channel_images=(
            g_image,
            r_image,
        ),
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    active = result.active

    assert active is _image_data(
        result,
        "scarlet_target",
    )

    metadata = _metadata(
        result,
        "scarlet_target",
    )

    assert metadata["scarlet_channel"] == "r"


def test_scarlet_multi_channel_uses_selected_channel_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The reconstructed active target corresponds to the configured channel,
    not simply the first channel.
    """
    observation = _make_observation()

    g_image = _make_image(
        value=1.0,
    )

    r_image = _make_image(
        value=2.0,
    )

    observation.images["g"] = g_image
    observation.images["r"] = r_image

    preprocessor = ScarletPreprocessor(
        channel_names=(
            "g",
            "r",
        ),
        photometry_channel="r",
    )

    target_cube = np.zeros(
        (
            2,
            10,
            10,
        ),
        dtype=np.float64,
    )

    target_cube[
        0,
        5,
        5,
    ] = 10.0

    target_cube[
        1,
        5,
        5,
    ] = 20.0

    model_cube = np.zeros_like(
        target_cube,
    )

    scarlet_input = ScarletInput(
        image_cube=np.stack(
            [
                np.asarray(
                    g_image.image.data,
                    dtype=np.float64,
                ),
                np.asarray(
                    r_image.image.data,
                    dtype=np.float64,
                ),
            ],
            axis=0,
        ),
        weights=np.ones_like(
            target_cube,
        ),
        channel_names=(
            "g",
            "r",
        ),
        target_sky_coord=(
            100.0,
            -30.0,
        ),
        wcs=observation.metadata.wcs,
        channel_images=(
            g_image,
            r_image,
        ),
    )

    fake_scarlet = _make_fake_scarlet()

    class _TargetSource(_FakeScarletSource):
        def __init__(
            self,
            frame: Any,
            target_sky_coord: tuple[float, float],
            scarlet_observation: Any,
            *,
            K: int,
            thresh: float,
        ) -> None:
            super().__init__(
                frame,
                target_sky_coord,
                scarlet_observation,
                K=K,
                thresh=thresh,
            )

            self.model = target_cube.copy()

    class _Blend(_FakeScarletBlend):
        def get_model(
            self,
            *,
            frame: Any,
        ) -> np.ndarray:
            return model_cube.copy()

    fake_scarlet.ExtendedSource = _TargetSource
    fake_scarlet.Blend = _Blend

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: fake_scarlet,
    )

    result = preprocessor.process(
        observation,
    )

    active_target = _image_array(
        result,
        "scarlet_target",
    )

    expected_r_channel = target_cube[1]

    np.testing.assert_allclose(
        np.asarray(
            active_target,
            dtype=np.float64,
        ),
        np.asarray(
            expected_r_channel,
            dtype=np.float64,
        ),
    )


# ============================================================================
# Failure handling
# ============================================================================


def test_scarlet_wraps_unexpected_scarlet_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected SCARLET failures are wrapped in PreprocessingError."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    def _raise() -> Any:
        raise RuntimeError(
            "synthetic SCARLET failure",
        )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        _raise,
    )

    with pytest.raises(
        PreprocessingError,
        match="SCARLET processing failed",
    ):
        preprocessor.process(
            observation,
        )


def test_scarlet_preserves_preprocessing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing PreprocessingError instances are not double wrapped."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    expected = PreprocessingError(
        "synthetic preprocessing failure",
    )

    def _prepare(
        _observation: Observation,
        *,
        source_centers: Any = None,
    ) -> ScarletInput:
        del source_centers
        raise expected

    monkeypatch.setattr(
        preprocessor._adapter,
        "prepare",
        _prepare,
    )

    with pytest.raises(
        PreprocessingError,
    ) as exc_info:
        preprocessor.process(
            observation,
        )

    assert exc_info.value is expected


def test_scarlet_load_failure_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import failures are wrapped and preserve the original cause."""
    import builtins

    preprocessor = ScarletPreprocessor()

    original = RuntimeError(
        "import failure",
    )

    real_import = builtins.__import__

    def _raise_import(
        name: str,
        globals_: Any = None,
        locals_: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "scarlet":
            raise original

        return real_import(
            name,
            globals_,
            locals_,
            fromlist,
            level,
        )

    monkeypatch.setattr(
        builtins,
        "__import__",
        _raise_import,
    )

    with pytest.raises(
        PreprocessingError,
        match="SCARLET could not be imported",
    ) as exc_info:
        preprocessor._load_scarlet()

    assert exc_info.value.__cause__ is original


# ============================================================================
# Runtime-state validation
# ============================================================================


def test_validate_observation_rejects_none() -> None:
    """
    The runtime contract rejects None even though the public type annotation
    accepts Observation.

    cast() is intentional: this test deliberately creates an invalid runtime
    state in order to verify defensive validation.
    """
    invalid_observation = cast(
        Observation,
        None,
    )

    with pytest.raises(
        PreprocessingError,
        match="Observation cannot be None",
    ):
        ScarletPreprocessor._validate_observation(
            invalid_observation,
        )


def test_validate_observation_rejects_missing_active_image() -> None:
    """
    Deliberately construct an invalid Observation-like runtime state.

    Observation.active is not typed as optional, so assigning None directly
    would correctly trigger Pylance. cast() documents that this is intentional
    invalid-state testing.
    """
    observation = _make_observation()

    invalid_observation: Any = observation

    invalid_observation.active = None

    with pytest.raises(
        PreprocessingError,
        match="no active image",
    ):
        ScarletPreprocessor._validate_observation(
            cast(
                Observation,
                invalid_observation,
            ),
        )


def test_validate_observation_rejects_missing_active_image_data() -> None:
    """An active ImageData object without underlying image data is rejected."""
    observation = _make_observation()

    active: Any = observation.active

    active.image = None

    with pytest.raises(
        PreprocessingError,
        match="no active image data",
    ):
        ScarletPreprocessor._validate_observation(
            observation,
        )


def test_validate_observation_rejects_missing_active_data_array() -> None:
    """An active image object without data is rejected."""
    observation = _make_observation()

    active: Any = observation.active

    # NDData.data is read-only in the installed Astropy version.
    # Replace the image object with a deliberately invalid image-like object.
    active.image = SimpleNamespace(
        data=None,
    )

    with pytest.raises(
        PreprocessingError,
        match="Active image contains no image data",
    ):
        ScarletPreprocessor._validate_observation(
            observation,
        )


# ============================================================================
# SCARLET import boundary
# ============================================================================


def test_load_scarlet_returns_imported_module() -> None:
    """The loader returns the installed SCARLET module."""
    preprocessor = ScarletPreprocessor()

    module = preprocessor._load_scarlet()

    assert module is not None


# ============================================================================
# Regression tests for metadata and masks
# ============================================================================


def test_scarlet_target_metadata_is_not_none_after_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target metadata remains available after SCARLET processing."""
    observation = _make_observation()

    preprocessor = ScarletPreprocessor()

    scarlet_input = _make_scarlet_input(
        observation,
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    metadata = _metadata(
        result,
        "scarlet_target",
    )

    assert metadata["processing_product"] == "scarlet_target"


def test_scarlet_derived_mask_matches_source_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derived products retain the source channel's mask."""
    mask = np.zeros(
        (
            10,
            10,
        ),
        dtype=bool,
    )

    mask[
        2,
        3,
    ] = True

    source = _make_image(
        mask=mask,
    )

    observation = _make_observation()

    observation.active = source
    observation.images["raw"] = source

    preprocessor = ScarletPreprocessor()

    scarlet_input = ScarletInput(
        image_cube=np.asarray(
            source.image.data,
            dtype=np.float64,
        )[np.newaxis, ...],
        weights=np.ones(
            (
                1,
                10,
                10,
            ),
            dtype=np.float64,
        ),
        channel_names=("science",),
        target_sky_coord=(
            100.0,
            -30.0,
        ),
        wcs=observation.metadata.wcs,
        channel_images=(
            source,
        ),
    )

    _patch_adapter_prepare(
        monkeypatch,
        preprocessor,
        scarlet_input,
    )

    monkeypatch.setattr(
        preprocessor,
        "_load_scarlet",
        lambda: _make_fake_scarlet(),
    )

    result = preprocessor.process(
        observation,
    )

    derived = _image_data(
        result,
        "scarlet_target",
    )

    assert derived.image.mask is not None

    derived_mask = np.asarray(
        derived.image.mask,
        dtype=bool,
    )

    np.testing.assert_array_equal(
        derived_mask,
        mask,
    )