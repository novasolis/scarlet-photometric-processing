"""
Experiment configuration registry.

This module defines the reusable, reproducible configuration used by the
Photometric Analysis Pipeline.

The configuration layer describes:

    - scientific experiment type;
    - survey;
    - astronomical field/target coordinate;
    - acquisition preferences;
    - image preprocessing;
    - source detection;
    - aperture photometry;
    - evaluation;
    - runtime behaviour;
    - optional cadence-level time-series execution; and
    - optional differential-photometry reference-star configuration.

The experiment architecture distinguishes between:

    FIELD_ANALYSIS
        Uses a configured field coordinate to acquire an astronomical field,
        perform preprocessing, source detection and crowding analysis, and
        stop before target-centred photometry.

    TARGET_PHOTOMETRY
        Uses a specifically selected astronomical target.

    DIFFERENTIAL_PHOTOMETRY
        Uses a specifically selected astronomical target together with an
        explicitly configured reference-star set.

During the field-reconnaissance phase, the field registries contain the
candidate astronomical fields to be analysed. Target registries contain
the specifically selected astronomical sources that will be used for
target-centred photometry experiments.

This module contains configuration and factory logic only.

The active experiment suite itself belongs in ``experiments.py`` and must
not be defined here.

Acquisition strategy remains a responsibility of the survey-specific
acquisition backend.
"""

from __future__ import annotations

from core.models import (
    AcquisitionConfig,
    DetectionConfig,
    DifferentialPhotometryConfig,
    EvaluationConfig,
    ExperimentConfig,
    PhotometryConfig,
    PreprocessingConfig,
    ReferenceStarConfig,
    RuntimeConfig,
    TargetConfig,
    TimeSeriesConfig,
)

from domain.enums import (
    ExperimentType,
    ProcessingPipeline,
    Survey,
)


# ============================================================================
# TESS FIELD REGISTRY
# ============================================================================

"""
TESS field registry.

These coordinates identify the astronomical field centres used during
FIELD_ANALYSIS. The same sky position may also correspond to a named
astrophysical object, but the ``_field`` identifier denotes its role as an
acquisition/reconnaissance coordinate, not the photometric target role.

The purpose of these fields is to:

    1. acquire a representative astronomical field;
    2. perform the existing preprocessing/background-removal stage;
    3. detect all usable sources;
    4. calculate source-neighbour/crowding information; and
    5. identify suitable candidate targets for later target-centred
       experiments.

The eventual photometric target will be selected from the FIELD_ANALYSIS
results and subsequently added to the target registry.
"""

TESS_FIELD_TARGETS: dict[
    str,
    TargetConfig,
] = {
    # ------------------------------------------------------------------------
    # HD 209458 field (verified)
    # ------------------------------------------------------------------------

    "hd_209458_field": TargetConfig(
        target_name="HD 209458",
        right_ascension=330.79504167,
        declination=18.88424167,
    ),
    "toi_2155_field": TargetConfig(
        target_name="TOI-2155",
        right_ascension=5.80025000,
        declination=84.45283889,
    ),
    "wasp12_field": TargetConfig(
        target_name="WASP-12",
        right_ascension=97.63662500,
        declination=29.67226667,
    ),

    # ------------------------------------------------------------------------
    # WASP-33 field (verified)
    # ------------------------------------------------------------------------
    
    "wasp33_field": TargetConfig(
        target_name="WASP-33",
        right_ascension=36.71275000,
        declination=37.55044444,
    ),

    "gj9827_field": TargetConfig(
        target_name="GJ 9827",
        right_ascension=351.77179167,
        declination=-1.28534444,
    ),

    # ------------------------------------------------------------------------
    # LHS 3844 field
    # ------------------------------------------------------------------------

    "lhs_3844_field": TargetConfig(
        target_name="TIC 410153553",
        right_ascension=340.49210,
        declination=-69.16900,
    ),
    "ngc_3532_field": TargetConfig(
        target_name="NGC 3532",
        right_ascension=166.39900000,
        declination=-58.69600000,
    ),

    "hd_191939_field": TargetConfig(
        target_name="HD 191939",
        right_ascension=302.02562500,
        declination=66.85030278,
    ),
}


# ============================================================================
# TESS TARGET REGISTRIES
# ============================================================================

"""
TESS target-centred photometry registry.

These entries identify the actual astronomical objects whose photometry will
be measured. The identifier suffix ``_photometry`` deliberately distinguishes
them from FIELD_ANALYSIS identifiers ending in ``_field``.
"""

TESS_PHOTOMETRY_TARGETS: dict[
    str,
    TargetConfig,
] = {

    # HD 209458 — a hot Jupiter famous for being one of the first transiting
    # exoplanets whose atmosphere was studied in detail.
    "hd_209458_photometry": TargetConfig(
        target_name="HD 209458",
        right_ascension=330.79504167,
        declination=18.88424167,
    ),
    # WASP-12 — a highly irradiated hot Jupiter orbiting extremely close to its star.
    "wasp12_photometry": TargetConfig(
        target_name="WASP-12",
        right_ascension=97.63662500,
        declination=29.67226667,
    ),

    # WASP-33 — a hot Jupiter orbiting a rapidly rotating A-type star.
    "wasp33_photometry": TargetConfig(
        target_name="WASP-33",
        right_ascension=36.71275000,
        declination=37.55044444,
    ),

    #GJ 9827 is a star in the constellation of Pisces a nearby multi-planet system with three known transiting worlds.
    "gj9827_photometry": TargetConfig(
        target_name="GJ 9827",
        right_ascension=351.77179167,
        declination=-1.28534444,
    ),

    # TOI-1820 — a short-period giant planet system used to calibrate TESS photometric deblending factors against nearby contamination.
    "toi_1820_photometry": TargetConfig(
        target_name="TOI-1820",
        right_ascension=187.68672083,
        declination=27.45200167,
    ),
    
    # "toi_1000_photometry": TargetConfig(
    #     target_name="TOI-1000",
    #     right_ascension=112.35770833,
    #     declination=-12.69596111,
    # ),

    "hd_191939_photometry": TargetConfig(
        target_name="HD 191939",
        right_ascension=302.02562500,
        declination=66.85030278,
    ),

    "toi_2155_photometry": TargetConfig(
        target_name="TOI-2155",
        right_ascension=5.80025000,
        declination=84.45283889,
    ),
    
}


# ============================================================================
# TESS DIFFERENTIAL PHOTOMETRY
# ============================================================================

"""
TESS targets selected for the differential-photometry experiment.

The target definitions themselves remain exclusively in
``TESS_PHOTOMETRY_TARGETS`` above. This registry only selects which existing
TESS photometry targets participate in the differential-photometry study.
"""

DIFFERENTIAL_PHOTOMETRY_TARGETS: tuple[str, ...] = (
    "hd_209458_photometry",
    "toi_2155_photometry",
    "gj9827_photometry",
)


"""
Target-specific differential-photometry reference-star selections.

These are intentionally empty during the initial setup. Reference stars will
be selected independently for each target following FIELD_ANALYSIS.

Reference stars do not need to be the same across targets.
"""

DIFFERENTIAL_PHOTOMETRY_REFERENCE_STARS: dict[
    str,
    tuple[ReferenceStarConfig, ...],
] = {
    "hd_209458_photometry": (
        ReferenceStarConfig(
            identifier="TIC 420814467",
            right_ascension=330.72922558,
            declination=18.77336046,
        ),
        ReferenceStarConfig(
            identifier="TIC 372975014",
            right_ascension=330.86801183,
            declination=18.86493500,
        ),
    ),
    "toi_2155_photometry": (
        ReferenceStarConfig(
            identifier="TIC 461591578",
            right_ascension=5.47222495,
            declination=84.33648279,
        ),
        ReferenceStarConfig(
            identifier="TIC 461593426",
            right_ascension=7.09442341,
            declination=84.45291873,
        ),
    ),
    "gj9827_photometry": (
        ReferenceStarConfig(
            identifier="TIC 301287598",
            right_ascension=351.66859999,
            declination=-1.35191039,
        ),
        ReferenceStarConfig(
            identifier="TIC 301289529",
            right_ascension=351.73757053,
            declination=-1.23185834,
        ),
    ),
}


# ============================================================================
# SDSS FIELD REGISTRY
# ============================================================================

"""
SDSS field-centre registry.

These coordinates identify fields to be acquired and investigated during
FIELD_ANALYSIS.

They are NOT designated photometric targets.

The field-analysis stage will identify detected sources and their crowding
characteristics. Suitable individual sources can subsequently be promoted
to the SDSS target registry.
"""

SDSS_FIELD_TARGETS: dict[
    str,
    TargetConfig,
] = {
    # ------------------------------------------------------------------------
    # M67 / NGC 2682
    # ------------------------------------------------------------------------

    "m67": TargetConfig(
        target_name="M67",
        right_ascension=132.82500000,
        declination=11.81700000,
    ),

    # ------------------------------------------------------------------------
    # NGC 6791
    # ------------------------------------------------------------------------

    "ngc6791": TargetConfig(
        target_name="NGC 6791",
        right_ascension=290.22083333,
        declination=37.77805556,
    ),

    # ------------------------------------------------------------------------
    # M13 / NGC 6205
    # ------------------------------------------------------------------------

    "ngc2420": TargetConfig(
        target_name="NGC 2420",
        right_ascension=114.6020833333,
        declination=21.5750000000,
    ),

    # ------------------------------------------------------------------------
    # M92 / NGC 6341
    # ------------------------------------------------------------------------

    "m92": TargetConfig(
        target_name="M92",
        right_ascension=259.28041667,
        declination=43.13583333,
    ),

    # ------------------------------------------------------------------------
    # M3 / NGC 5272
    # ------------------------------------------------------------------------

    "ngc2158": TargetConfig(
        target_name="NGC 2158",
        right_ascension=91.86208333,
        declination=24.09888889,
    ),
}


# ============================================================================
# SDSS TARGET REGISTRIES
# ============================================================================

"""
SDSS target-centred benchmark registry.

Intentionally empty during the field-reconnaissance phase.

Individual sources identified from FIELD_ANALYSIS will be added here once
the scientific target-selection stage has been completed.
"""

SDSS_PHOTOMETRY_TARGETS: dict[
    str,
    TargetConfig,
] = {

    "sdss_m67_src262": TargetConfig(
        target_name="SDSS M67 Source 262",
        right_ascension=132.82294548,
        declination=11.75579447,
    ),

    "sdss_m67_src118" : TargetConfig(
        target_name="SDSS M67 Source 118",
        right_ascension=132.76448090,
        declination=11.75032944,
    ),

    "m67_sand978": TargetConfig(
        target_name="M67 SAND 978",
        right_ascension=132.8228475,
        declination=11.7563036,
    ),

    "sdss_ngc6791_src2029" : TargetConfig(
        target_name="SDSS NGC 6791 Source 2029",
        right_ascension=290.203657,
        declination=37.728022,
    ),

    "sdss_ngc2420_src240": TargetConfig(
        target_name="SDSS NGC 2420 Source 240",
        right_ascension=114.60043016,
        declination=21.58017349,
    ),

    "sdss_m92_src2627": TargetConfig(
        target_name="SDSS M92 Source 2627",
        right_ascension=259.23082673,
        declination=43.16980267,
    ),

    "sdss_ngc2158_src2662": TargetConfig(
        target_name="SDSS NGC 2158 Source 2662",
        right_ascension=91.86912343,
        declination=24.08872289,
    ),
    #Non-crowded
    "sdss_ngc2158_src1317": TargetConfig(
        target_name="SDSS NGC 2158 Source 1317",
        right_ascension=91.82770025,
        declination=24.06746794,
    ),
}


"""
SDSS differential-photometry reference registry.

Reserved for future use. Differential photometry is currently staged only
for TESS because the present experiment requires time-series execution.
"""

SDSS_DIFFERENTIAL_REFERENCES: dict[
    str,
    tuple[str, ...],
] = {}


# ============================================================================
# Default field identifiers
# ============================================================================

"""
Defaults are field identifiers rather than photometric targets.

They are used only when an experiment type permits a default field.

FIELD_ANALYSIS still requires an explicit ``target_name`` so that the field
being analysed is always unambiguous.
"""

DEFAULT_TESS_FIELD = "hd_209458_field"
DEFAULT_SDSS_FIELD = "m67"


# ============================================================================
# Acquisition configurations
# ============================================================================

"""
Standard TESS Target Pixel File acquisition.

This remains the normal target-centred acquisition configuration.

The resulting observation contains the spatial information required by the
target-centred workflow.
"""

TESS_ACQUISITION_CONFIG = AcquisitionConfig(
    sector=None,
    author="SPOC",
    cadence_seconds=120,
    cutout_size=25,
)


"""
Extended-spatial-context TESS acquisition.

This is used by FIELD_ANALYSIS.

The larger TESSCut field provides the spatial context required for:

    - multi-source detection;
    - complete-field source identification;
    - crowding analysis;
    - candidate target selection.

No target-centred photometry is implied by this configuration.
"""

TESS_LOCAL_FIELD_ACQUISITION_CONFIG = AcquisitionConfig(
    sector=None,
    author=None,
    cadence_seconds=None,
    cutout_size=50,
)


"""
Backward-compatible acquisition alias.

Retained temporarily for source-level compatibility while the remaining
architecture is migrated away from the old field terminology.
"""

TESS_FIELD_ACQUISITION_CONFIG = TESS_LOCAL_FIELD_ACQUISITION_CONFIG


"""
SDSS acquisition.

SDSS does not use the TESS-style cutout mechanism.

``cutout_size=None`` therefore preserves the native/full image supplied by
the SDSS acquisition path.
"""

SDSS_ACQUISITION_CONFIG = AcquisitionConfig(
    sector=None,
    author=None,
    cadence_seconds=None,
    cutout_size=None,
)


# ============================================================================
# Preprocessing configuration
# ============================================================================

"""
Common preprocessing configuration.

The existing preprocessing values are retained.

For TESS, the established 32-pixel background mesh remains in place.
Background treatment remains part of the normal workflow; FIELD_ANALYSIS
does not bypass preprocessing.

The FIELD_ANALYSIS experiment type only controls where the workflow stops.
"""

DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig(
    background_box_size=32,
    background_filter_size=3,
    sigma_clip=3.0,
    exclude_percentile=10.0,
)


# ============================================================================
# Source-detection configurations
# ============================================================================

"""
Generic conservative detection configuration.
"""

DEFAULT_DETECTION_CONFIG = DetectionConfig(
    fwhm=3.0,
    detection_threshold_sigma=5.0,
)


"""
TESS source detection.

TESS has a broad PSF relative to its pixel scale.

The established configuration is retained:

    FWHM = 2.5 pixels
    threshold = 3 sigma

The lower threshold is useful during FIELD_ANALYSIS because neighbouring
sources are scientifically relevant to the crowding analysis.
"""

TESS_DETECTION_CONFIG = DetectionConfig(
    fwhm=2.5,
    detection_threshold_sigma=3.0,
)


"""
SDSS source detection.

The established conservative configuration is retained.
"""

SDSS_DETECTION_CONFIG = DetectionConfig(
    fwhm=3.0,
    detection_threshold_sigma=5.0,
)


# ============================================================================
# Photometry configurations
# ============================================================================

"""
TESS aperture photometry.

TESS pixels are large, so the source aperture remains compact.
"""

TESS_PHOTOMETRY_CONFIG = PhotometryConfig(
    aperture_radius=2.0,
    annulus_inner_radius=3.0,
    annulus_outer_radius=4.0,
    minimum_aperture_coverage=0.90,
    minimum_annulus_coverage=0.70,
    minimum_annulus_pixels=20,
    reference_star_count=10,
)


"""
SDSS aperture photometry.

The larger geometry reflects the finer spatial sampling of SDSS.
"""

SDSS_PHOTOMETRY_CONFIG = PhotometryConfig(
    aperture_radius=4.0,
    annulus_inner_radius=6.0,
    annulus_outer_radius=8.0,
    minimum_aperture_coverage=0.90,
    minimum_annulus_coverage=0.70,
    minimum_annulus_pixels=20,
    reference_star_count=10,
)


# ============================================================================
# Evaluation configuration
# ============================================================================

"""
Default evaluation configuration.

matching_tolerance
    Maximum positional difference used when matching detections.

crowding_radius
    Maximum source separation used when assessing local crowding.
"""

DEFAULT_EVALUATION_CONFIG = EvaluationConfig(
    matching_tolerance=5.0,
    crowding_radius=10.0,
)


# ============================================================================
# Runtime configuration
# ============================================================================

"""
Default runtime configuration.

Figures are generated for diagnostic and experimental analysis but are not
automatically displayed by the configuration layer.
"""

DEFAULT_RUNTIME_CONFIG = RuntimeConfig(
    save_results=True,
    generate_figures=True,
    show_figures=False,
    overwrite_existing=True,
    logging_level="INFO",
)


# ============================================================================
# Factory
# ============================================================================

def create_experiment_config(
    *,
    survey: Survey,
    experiment_type: ExperimentType,
    pipeline: ProcessingPipeline = ProcessingPipeline.BASELINE,
    target_name: str | None = None,
    acquisition: AcquisitionConfig | None = None,
    time_series: TimeSeriesConfig | None = None,
    differential_photometry: DifferentialPhotometryConfig | None = None,
) -> ExperimentConfig:
    """
    Create a complete experiment configuration.

    FIELD_ANALYSIS
        ``target_name`` identifies a field entry ending in ``_field``. The workflow performs
        acquisition, preprocessing, source detection and crowding analysis,
        then terminates before target-centred photometry.

    TARGET_PHOTOMETRY
        ``target_name`` must identify an entry in the appropriate target
        registry (for TESS, identifiers ending in ``_photometry``).

    DIFFERENTIAL_PHOTOMETRY
        ``target_name`` must identify an entry in the appropriate
        differential-photometry target registry and a differential
        configuration must be supplied.

    Parameters
    ----------
    survey
        Astronomical survey providing the observation.

    experiment_type
        Scientific experiment being performed.

    pipeline
        Photometric processing pipeline.

    target_name
        Field or target identifier.

        Required for FIELD_ANALYSIS.

        For target-centred experiments, the identifier must exist in the
        appropriate target registry.

    acquisition
        Optional experiment-specific acquisition configuration.

    time_series
        Optional cadence-level time-series configuration.

    differential_photometry
        Differential-photometry configuration. Required only for
        DIFFERENTIAL_PHOTOMETRY experiments.

    Returns
    -------
    ExperimentConfig
        Complete experiment configuration.

    Raises
    ------
    ValueError
        If configuration requirements are not satisfied.
    """

    # ------------------------------------------------------------------------
    # Validate experiment type
    # ------------------------------------------------------------------------

    if not isinstance(experiment_type, ExperimentType):
        raise ValueError(
            f"Unsupported experiment type: {experiment_type!r}.",
        )

    # ------------------------------------------------------------------------
    # Validate processing pipeline
    # ------------------------------------------------------------------------

    if not isinstance(pipeline, ProcessingPipeline):
        raise ValueError(
            f"Unsupported processing pipeline: {pipeline!r}.",
        )

    # ------------------------------------------------------------------------
    # FIELD_ANALYSIS requires an explicit field
    # ------------------------------------------------------------------------

    if (
        experiment_type is ExperimentType.FIELD_ANALYSIS
        and target_name is None
    ):
        raise ValueError(
            (
                "FIELD_ANALYSIS experiments require target_name because "
                "target_name identifies the field centre used for acquisition."
            ),
        )

    # ------------------------------------------------------------------------
    # FIELD_ANALYSIS is single-observation reconnaissance
    # ------------------------------------------------------------------------

    if (
        experiment_type is ExperimentType.FIELD_ANALYSIS
        and time_series is not None
        and time_series.enabled
    ):
        raise ValueError(
            "FIELD_ANALYSIS experiments do not support time-series execution.",
        )

    # ------------------------------------------------------------------------
    # Validate differential-photometry configuration
    # ------------------------------------------------------------------------

    if experiment_type is ExperimentType.DIFFERENTIAL_PHOTOMETRY:

        if differential_photometry is None:
            raise ValueError(
                (
                    "DIFFERENTIAL_PHOTOMETRY experiments require "
                    "a differential_photometry configuration."
                ),
            )

    elif differential_photometry is not None:

        raise ValueError(
            (
                "differential_photometry configuration is only valid for "
                "DIFFERENTIAL_PHOTOMETRY experiments."
            ),
        )

    # ------------------------------------------------------------------------
    # Select survey-specific configuration
    # ------------------------------------------------------------------------

    match survey:

        case Survey.TESS:

            detection = TESS_DETECTION_CONFIG
            photometry = TESS_PHOTOMETRY_CONFIG

            field_registry = TESS_FIELD_TARGETS
            target_registry = TESS_PHOTOMETRY_TARGETS

            default_field = DEFAULT_TESS_FIELD

            default_acquisition = (
                TESS_LOCAL_FIELD_ACQUISITION_CONFIG
                if experiment_type is ExperimentType.FIELD_ANALYSIS
                else TESS_ACQUISITION_CONFIG
            )

        case Survey.SDSS:

            detection = SDSS_DETECTION_CONFIG
            photometry = SDSS_PHOTOMETRY_CONFIG

            field_registry = SDSS_FIELD_TARGETS
            target_registry = SDSS_PHOTOMETRY_TARGETS

            default_field = DEFAULT_SDSS_FIELD
            default_acquisition = SDSS_ACQUISITION_CONFIG

        case _:

            raise ValueError(
                f"Unsupported survey: {survey!r}.",
            )

    # ------------------------------------------------------------------------
    # Select the appropriate coordinate registry
    # ------------------------------------------------------------------------

    if experiment_type is ExperimentType.FIELD_ANALYSIS:

        registry = field_registry

        if target_name is None:
            raise ValueError(
                "FIELD_ANALYSIS requires an explicit field target_name.",
            )

        selected_name = target_name

    elif experiment_type is ExperimentType.TARGET_PHOTOMETRY:

        registry = target_registry

        if target_name is None:
            raise ValueError(
                (
                    f"{survey.value} TARGET_PHOTOMETRY requires an explicit "
                    "target_name. No default photometric target is currently "
                    "configured."
                ),
            )

        selected_name = target_name

    elif experiment_type is ExperimentType.DIFFERENTIAL_PHOTOMETRY:

        # Differential photometry operates on an existing photometric target.
        # The target coordinates therefore come from the same target registry
        # used by TARGET_PHOTOMETRY. Reference-star selection is supplied
        # separately through DifferentialPhotometryConfig.
        registry = target_registry

        if target_name is None:
            raise ValueError(
                (
                    f"{survey.value} DIFFERENTIAL_PHOTOMETRY requires an "
                    "explicit target_name."
                ),
            )

        selected_name = target_name

    else:

        raise ValueError(
            f"Unsupported experiment type: {experiment_type!r}.",
        )

    # ------------------------------------------------------------------------
    # Resolve field/target coordinates
    # ------------------------------------------------------------------------

    try:

        target = registry[selected_name]

    except KeyError as exc:

        available = ", ".join(
            sorted(registry),
        )

        if available:
            message = (
                f"Unknown {survey.value} {experiment_type.value} identifier "
                f"'{selected_name}'. Available entries: {available}."
            )
        else:
            message = (
                f"No {survey.value} {experiment_type.value} targets are "
                "currently configured. Run FIELD_ANALYSIS first and populate "
                "the target registry with a selected source."
            )

        raise ValueError(message) from exc

    # ------------------------------------------------------------------------
    # Validate coordinates
    # ------------------------------------------------------------------------

    if (
        target.right_ascension is None
        or target.declination is None
    ):
        raise ValueError(
            (
                f"Configuration entry '{target.target_name}' requires "
                "right ascension and declination."
            ),
        )

    # ------------------------------------------------------------------------
    # Resolve acquisition configuration
    # ------------------------------------------------------------------------

    effective_acquisition = (
        acquisition
        if acquisition is not None
        else default_acquisition
    )

    # ------------------------------------------------------------------------
    # Normalize time-series configuration
    # ------------------------------------------------------------------------

    effective_time_series = (
        time_series
        if time_series is not None
        else TimeSeriesConfig()
    )

    # ------------------------------------------------------------------------
    # Validate time-series compatibility
    # ------------------------------------------------------------------------

    if effective_time_series.enabled:

        if survey is not Survey.TESS:
            raise ValueError(
                (
                    "Time-series execution is currently supported only "
                    "for TESS experiments."
                ),
            )

    # ------------------------------------------------------------------------
    # Construct final configuration
    # ------------------------------------------------------------------------

    return ExperimentConfig(
        experiment_type=experiment_type,
        survey=survey,
        target=target,
        acquisition=effective_acquisition,
        preprocessing=DEFAULT_PREPROCESSING_CONFIG,
        detection=detection,
        photometry=photometry,
        evaluation=DEFAULT_EVALUATION_CONFIG,
        runtime=DEFAULT_RUNTIME_CONFIG,
        time_series=effective_time_series,
        differential_photometry=differential_photometry,
        pipeline=pipeline,
    )