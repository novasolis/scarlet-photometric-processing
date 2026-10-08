"""
Unit tests for the SCARLET input adapter.

These tests exercise the domain-to-SCARLET input boundary without importing
or executing SCARLET itself.
"""

from datetime import datetime, timezone
from typing import cast

import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.wcs import WCS

from core.exceptions import PreprocessingError
from domain.enums import ObservationSource, Survey
from domain.models import (
    ImageData,
    Observation,
    ObservationMetadata,
    ObservationProvenance,
    ProcessingState,
)
from preprocessing.scarlet_adapter import (
    ScarletInput,
    ScarletInputAdapter,
)


@pytest.fixture
def observation_factory():
    """Return a factory for constructing controlled test Observations."""

    def _create(
        *,
        data=None,
        images=None,
        target_ra=132.73609708,
        target_dec=11.83185478,
        wcs=None,
        background_removed=True,
        mask=None,
    ):
        if data is None:
            rng = np.random.default_rng(42)
            data = rng.normal(
                loc=2.0,
                scale=0.1,
                size=(8, 10),
            ).astype(np.float32)

        active_nddata = NDData(
            data=np.asarray(data),
            mask=mask,
            wcs=wcs,
        )

        active = ImageData(active_nddata)

        metadata = ObservationMetadata(
            target_name="test-target",
            observation_time=datetime.now(timezone.utc),
            right_ascension=target_ra,
            declination=target_dec,
            wcs=wcs,
        )

        provenance = ObservationProvenance(
            source=ObservationSource.GROUND_BASED,
            survey=Survey.SDSS,
            dataset_identifier="test-dataset",
            filename="test.fits",
            download_timestamp=datetime.now(timezone.utc),
        )

        processing_state = ProcessingState(
            background_removed=background_removed,
        )

        observation = Observation(
            metadata=metadata,
            provenance=provenance,
            original=active,
            active=active,
            processing_state=processing_state,
        )

        if images:
            for name, image_data in images.items():
                observation.images[name] = ImageData(
                    NDData(
                        data=np.asarray(image_data),
                    )
                )

        return observation

    return _create


def test_prepare_single_channel_returns_valid_scarlet_input(
    observation_factory,
):
    """A valid active image produces a valid single-channel input."""
    observation = observation_factory()

    result = ScarletInputAdapter().prepare(observation)

    assert isinstance(result, ScarletInput)

    assert result.image_cube.shape == (1, 8, 10)
    assert result.image_cube.dtype == np.float64

    assert result.weights.shape == (1, 8, 10)
    assert result.weights.dtype == np.float64

    assert result.channel_names == ("science",)

    assert result.target_sky_coord == (
        132.73609708,
        11.83185478,
    )

    assert result.wcs is None
    assert result.channel_images == (observation.active,)


def test_prepare_preserves_image_values(
    observation_factory,
):
    """The adapter does not alter valid astronomical image values."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    observation = observation_factory(data=data)

    result = ScarletInputAdapter().prepare(observation)

    np.testing.assert_array_equal(
        result.image_cube[0],
        data,
    )


def test_prepare_promotes_input_to_float64(
    observation_factory,
):
    """Integer and lower-precision input is represented as float64."""
    data = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=np.int16,
    )

    observation = observation_factory(data=data)

    result = ScarletInputAdapter().prepare(observation)

    assert result.image_cube.dtype == np.float64


def test_prepare_uses_active_image_by_default(
    observation_factory,
):
    """Without explicit channels, the active image is selected."""
    observation = observation_factory()

    result = ScarletInputAdapter().prepare(observation)

    assert result.channel_names == ("science",)
    assert result.channel_images[0] is observation.active


def test_prepare_uses_explicit_channels(
    observation_factory,
):
    """Explicit channel names select the corresponding Observation images."""
    rng = np.random.default_rng(42)

    image_a = rng.normal(
        loc=2.0,
        scale=0.1,
        size=(8, 10),
    )

    image_b = rng.normal(
        loc=5.0,
        scale=0.1,
        size=(8, 10),
    )

    observation = observation_factory(
        images={
            "g": image_a,
            "r": image_b,
        },
    )

    adapter = ScarletInputAdapter(
        channel_names=("g", "r"),
    )

    result = adapter.prepare(observation)

    assert result.channel_names == ("g", "r")
    assert result.image_cube.shape == (2, 8, 10)

    np.testing.assert_array_equal(
        result.image_cube[0],
        image_a,
    )

    np.testing.assert_array_equal(
        result.image_cube[1],
        image_b,
    )


def test_prepare_preserves_explicit_channel_order(
    observation_factory,
):
    """Channel order is exactly the configured order."""
    rng = np.random.default_rng(42)
    images = {
        "g": rng.normal(2.0, 0.1, (20, 20)),
        "r": rng.normal(5.0, 0.1, (20, 20)),
        "i": rng.normal(8.0, 0.1, (20, 20)),
    }
    observation = observation_factory(
        
        images = images,
    )

    adapter = ScarletInputAdapter(
        channel_names=("i", "g", "r"),
    )

    result = adapter.prepare(observation)

    assert result.channel_names == ("i", "g", "r")

    np.testing.assert_array_equal(
        result.image_cube[0],
        images["i"],
    )

    np.testing.assert_array_equal(
        result.image_cube[1],
        images["g"],
    )

    np.testing.assert_array_equal(
        result.image_cube[2],
        images["r"],
    )


def test_prepare_rejects_missing_explicit_channel(
    observation_factory,
):
    """A configured channel must exist in Observation.images."""
    observation = observation_factory(
        images={
            "g": np.ones((4, 4)),
        },
    )

    adapter = ScarletInputAdapter(
        channel_names=("g", "r"),
    )

    with pytest.raises(
        PreprocessingError,
        match="SCARLET channel image 'r' was not found",
    ):
        adapter.prepare(observation)


def test_adapter_rejects_empty_channel_configuration():
    """An empty channel configuration is invalid."""
    with pytest.raises(
        ValueError,
        match="channel_names cannot be empty",
    ):
        ScarletInputAdapter(channel_names=())


def test_adapter_rejects_duplicate_channel_names():
    """SCARLET channel names must be unique."""
    with pytest.raises(
        ValueError,
        match="channel_names must be unique",
    ):
        ScarletInputAdapter(
            channel_names=("g", "g"),
        )


def test_prepare_rejects_empty_image(
    observation_factory,
):
    """An empty image cannot form a SCARLET channel."""
    observation = observation_factory(
        data=np.empty((0, 0)),
    )

    with pytest.raises(
        PreprocessingError,
        match="SCARLET cannot process an empty image",
    ):
        ScarletInputAdapter().prepare(observation)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (10,),
        (2, 3, 4),
        (1, 2, 3, 4),
    ],
)
def test_prepare_rejects_non_2d_images(
    observation_factory,
    shape,
):
    """SCARLET channel images must be exactly two-dimensional."""
    observation = observation_factory(
        data=np.ones(shape),
    )

    with pytest.raises(
        PreprocessingError,
        match="SCARLET channel images must be 2-D",
    ):
        ScarletInputAdapter().prepare(observation)


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_prepare_rejects_non_finite_image_values(
    observation_factory,
    bad_value,
):
    """NaN and infinite pixel values are rejected."""
    data = np.ones((4, 4))
    data[1, 1] = bad_value

    observation = observation_factory(data=data)

    with pytest.raises(
        PreprocessingError,
        match="SCARLET input image contains non-finite values",
    ):
        ScarletInputAdapter().prepare(observation)


def test_prepare_rejects_inconsistent_channel_shapes(
    observation_factory,
):
    """All SCARLET channels must have identical spatial dimensions."""
    observation = observation_factory(
        images={
            "g": np.ones((8, 10)),
            "r": np.ones((7, 10)),
        },
    )

    adapter = ScarletInputAdapter(
        channel_names=("g", "r"),
    )

    with pytest.raises(
        PreprocessingError,
        match=(
            r"channel image 1 has shape \(7, 10\); "
            r"expected \(8, 10\)"
        ),
    ):
        adapter.prepare(observation)


def test_prepare_accepts_missing_wcs(
    observation_factory,
):
    """
    WCS is optional at this boundary.

    The current SCARLET Frame implementation explicitly supports a None WCS.
    """
    observation = observation_factory(wcs=None)

    result = ScarletInputAdapter().prepare(observation)

    assert result.wcs is None


def test_prepare_preserves_wcs(
    observation_factory,
):
    """A supplied WCS is carried through unchanged."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [5.0, 5.0]
    wcs.wcs.crval = [132.73609708, 11.83185478]
    wcs.wcs.cdelt = [-0.0001, 0.0001]
    wcs.wcs.ctype = [
        "RA---TAN",
        "DEC--TAN",
    ]

    observation = observation_factory(wcs=wcs)

    result = ScarletInputAdapter().prepare(observation)

    assert result.wcs is wcs


def test_prepare_rejects_missing_right_ascension(
    observation_factory,
):
    """Target RA is required for SCARLET source initialisation."""
    observation = observation_factory(target_ra=None)

    with pytest.raises(
        PreprocessingError,
        match="SCARLET requires a target right ascension",
    ):
        ScarletInputAdapter().prepare(observation)


def test_prepare_rejects_missing_declination(
    observation_factory,
):
    """Target Dec is required for SCARLET source initialisation."""
    observation = observation_factory(target_dec=None)

    with pytest.raises(
        PreprocessingError,
        match="SCARLET requires a target declination",
    ):
        ScarletInputAdapter().prepare(observation)


@pytest.mark.parametrize(
    "ra, dec",
    [
        (np.nan, 11.0),
        (np.inf, 11.0),
        (-np.inf, 11.0),
        (132.0, np.nan),
        (132.0, np.inf),
        (132.0, -np.inf),
    ],
)
def test_prepare_rejects_non_finite_target_coordinates(
    observation_factory,
    ra,
    dec,
):
    """RA and Dec must both be finite."""
    observation = observation_factory(
        target_ra=ra,
        target_dec=dec,
    )

    with pytest.raises(
        PreprocessingError,
        match="not finite",
    ):
        ScarletInputAdapter().prepare(observation)


def test_prepare_requires_background_subtraction(
    observation_factory,
):
    """The adapter must not bypass the common background stage."""
    observation = observation_factory(
        background_removed=False,
    )

    with pytest.raises(
        PreprocessingError,
        match="background-subtracted image",
    ):
        ScarletInputAdapter().prepare(observation)


def test_prepare_rejects_none_observation():
    """None is not a valid domain observation."""
    with pytest.raises(
        PreprocessingError,
        match="Observation cannot be None",
    ):
        ScarletInputAdapter().prepare(
            cast(Observation, None)
        )


def test_prepare_rejects_missing_active_image(
    observation_factory,
):
    """An Observation without an active image is invalid."""
    observation = observation_factory()

    setattr(
        observation,
        "active",
        None,
    )

    with pytest.raises(
        PreprocessingError,
        match="Observation has no active image",
    ):
        ScarletInputAdapter().prepare(observation)


def test_prepare_generates_positive_finite_weights(
    observation_factory,
):
    """Valid channels produce finite positive inverse-variance weights."""
    rng = np.random.default_rng(42)

    data = rng.normal(
        loc=10.0,
        scale=2.0,
        size=(50, 50),
    )

    observation = observation_factory(data=data)

    result = ScarletInputAdapter().prepare(observation)

    assert result.weights.shape == result.image_cube.shape
    assert np.all(np.isfinite(result.weights))
    assert np.all(result.weights > 0)


def test_prepare_zeroes_masked_pixels_in_weights(
    observation_factory,
):
    """Masked pixels receive zero SCARLET weight."""
    rng = np.random.default_rng(42)

    data = rng.normal(
        loc=10.0,
        scale=2.0,
        size=(50, 50),
    )

    mask = np.zeros(
        data.shape,
        dtype=bool,
    )

    mask[2, 3] = True
    mask[7, 8] = True

    observation = observation_factory(
        data=data,
        mask=mask,
    )

    result = ScarletInputAdapter().prepare(observation)

    assert result.weights[0, 2, 3] == 0.0
    assert result.weights[0, 7, 8] == 0.0

    assert np.all(
        result.weights[0][~mask] > 0
    )


def test_prepare_rejects_mask_with_wrong_shape(
    observation_factory,
):
    """A mask must have the same geometry as its image."""
    data = np.ones((8, 10))
    mask = np.zeros(
        (7, 10),
        dtype=bool,
    )

    observation = observation_factory(
        data=data,
        mask=mask,
    )

    with pytest.raises(
        PreprocessingError,
        match="Image mask shape does not match image data shape",
    ):
        ScarletInputAdapter().prepare(observation)


def test_prepare_rejects_channel_with_no_valid_pixels(
    observation_factory,
):
    """A completely masked channel cannot be weighted."""
    data = np.ones((8, 10))

    mask = np.ones(
        (8, 10),
        dtype=bool,
    )

    observation = observation_factory(
        data=data,
        mask=mask,
    )

    with pytest.raises(
        PreprocessingError,
        match="contains no valid pixels",
    ):
        ScarletInputAdapter().prepare(observation)


def test_prepare_rejects_zero_background_rms(
    observation_factory,
):
    """A zero-variance channel cannot produce a valid inverse variance."""
    data = np.full(
        (20, 20),
        5.0,
    )

    observation = observation_factory(data=data)

    with pytest.raises(
        PreprocessingError,
        match="positive background RMS",
    ):
        ScarletInputAdapter().prepare(observation)


def test_scarlet_input_is_immutable(
    observation_factory,
):
    """ScarletInput is an immutable value object."""
    observation = observation_factory()

    result = ScarletInputAdapter().prepare(observation)

    with pytest.raises(AttributeError):
        setattr(
            result,
            "channel_names",
            ("changed",),
        )


def test_scarlet_input_rejects_shape_mismatch():
    """ScarletInput enforces its core structural invariants."""
    with pytest.raises(
        ValueError,
        match="weights must have the same shape",
    ):
        ScarletInput(
            image_cube=np.ones((1, 4, 4)),
            weights=np.ones((1, 3, 4)),
            channel_names=("science",),
            target_sky_coord=(132.0, 11.0),
            wcs=None,
            channel_images=(),
        )