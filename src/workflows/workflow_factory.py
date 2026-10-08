"""
Workflow factory.

Constructs fully configured photometric analysis workflows from an
experiment configuration.

This module serves as the application's composition root by wiring
survey-specific acquisition and Observation builders together with
survey-independent processing, source detection, SCARLET preprocessing,
target resolution, LocalField selection, photometry, differential
photometry and evaluation components.

Execution ordering belongs to ``PhotometricWorkflow``. This factory is
responsible only for constructing and injecting the appropriate
components.
"""

from __future__ import annotations

from typing import Any

from acquisition.base import BaseAcquirer
from acquisition.sdss import SDSSAcquirer
from acquisition.tess import TESSAcquirer

from core.models import ExperimentConfig

from detection.dao_star_finder import DAOStarFinderDetector

from domain.builders import ObservationBuilder
from domain.enums import ExperimentType, ProcessingPipeline, Survey
from domain.sdss_builder import SDSSObservationBuilder
from domain.tess_builder import TESSObservationBuilder

from evaluation.photometry import PhotometryEvaluator

from photometry.aperture import AperturePhotometer
from photometry.differential import DifferentialPhotometer

from preprocessing.background import BackgroundSubtractor
from preprocessing.preprocessing import Preprocessor
from preprocessing.scarlet import ScarletPreprocessor

from target.local_field import LocalFieldSelector
from target.reference_resolver import ReferenceSourceResolver
from target.resolver import TargetSourceResolver
from core.models import PhotometryConfig

from utils.logger import get_logger

from workflows.photometric_workflow import PhotometricWorkflow


def create_workflow(
    config: ExperimentConfig,
) -> PhotometricWorkflow:
    """
    Create a fully configured photometric analysis workflow.

    Survey-specific acquisition and Observation construction are selected
    from the experiment configuration. All subsequent processing operates
    on the common Observation domain model.

    The workflow owns execution ordering. The factory therefore injects
    the processing stages separately:

        BASELINE
            BackgroundSubtractor
                ↓
            SourceDetector

        SCARLET
            BackgroundSubtractor
                ↓
            SourceDetector
                ↓
            LocalFieldSelector
                ↓
            ScarletPreprocessor

    SCARLET is target-centred. The detected source catalogue is available
    to the workflow so that it can construct the target's LocalField
    before invoking SCARLET.

    Differential photometry is the only experiment type requiring the
    reference-source resolver and differential photometer.

    Parameters
    ----------
    config
        Experiment configuration.

    Returns
    -------
    PhotometricWorkflow
        Fully configured workflow.

    Raises
    ------
    ValueError
        If the configured survey or processing pipeline is unsupported.
    """

    logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Validate configuration boundary.
    # ------------------------------------------------------------------

    if not isinstance(
        config,
        ExperimentConfig,
    ):
        raise TypeError(
            "create_workflow requires an ExperimentConfig instance.",
        )

    # ------------------------------------------------------------------
    # Survey-specific acquisition and Observation construction.
    # ------------------------------------------------------------------
    #
    # Only acquisition and Observation construction depend on the
    # survey. Everything downstream operates on the common Observation
    # domain model.

    acquirer: BaseAcquirer[Any]
    builder: ObservationBuilder[Any]

    match config.survey:

        case Survey.TESS:
            acquirer = TESSAcquirer(
                config=config,
            )

            builder = TESSObservationBuilder(
                target=config.target,
            )

        case Survey.SDSS:
            acquirer = SDSSAcquirer(
                config=config,
            )

            builder = SDSSObservationBuilder(
                target=config.target,
            )

        case _:
            raise ValueError(
                f"Unsupported survey: {config.survey.value}",
            )

    # ------------------------------------------------------------------
    # Primary preprocessing.
    # ------------------------------------------------------------------
    #
    # Background subtraction is deliberately injected independently
    # from SCARLET.
    #
    # PhotometricWorkflow must execute this stage before source
    # detection so that DAOStarFinder operates on the same
    # background-subtracted image used by the subsequent processing
    # stages.

    preprocessor: Preprocessor = BackgroundSubtractor(
        config.preprocessing,
    )

    # ------------------------------------------------------------------
    # Optional SCARLET preprocessing.
    # ------------------------------------------------------------------
    #
    # SCARLET is intentionally NOT wrapped in SequentialPreprocessor.
    #
    # The workflow must first run source detection. The resulting
    # catalogue is then available for construction of the target's
    # LocalField and for SCARLET source modelling.
    #
    # Keeping ScarletPreprocessor separate preserves the required
    # execution ordering without making SCARLET part of the ordinary
    # preprocessing chain.

    scarlet_preprocessor: ScarletPreprocessor | None = None

    match config.pipeline:

        case ProcessingPipeline.BASELINE:
            scarlet_preprocessor = None

        case ProcessingPipeline.SCARLET:
            scarlet_preprocessor = ScarletPreprocessor(
                logger=logger,
            )

        case _:
            raise ValueError(
                f"Unsupported processing pipeline: "
                f"{config.pipeline.value}",
            )

    # ------------------------------------------------------------------
    # Source detection.
    # ------------------------------------------------------------------
    #
    # The detector is survey-independent and is intentionally injected
    # separately from preprocessing.
    #
    # PhotometricWorkflow executes it after the primary preprocessing
    # stage and before LocalField construction / SCARLET.

    detector = DAOStarFinderDetector(
        config=config.detection,
    )

    # ------------------------------------------------------------------
    # Target-source resolution.
    # ------------------------------------------------------------------
    #
    # Every experiment is now target-centred. Target resolution is
    # therefore part of the common workflow rather than something
    # selected according to a source-scope concept.

    target_resolver = TargetSourceResolver(
        matching_tolerance=config.evaluation.matching_tolerance,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Local-field selection.
    # ------------------------------------------------------------------
    #
    # LocalFieldSelector is a workflow dependency and is therefore
    # constructed at the composition root rather than inside
    # PhotometricWorkflow.
    #
    # The current scientific selection policy is:
    #
    #     1 target + up to 29 nearest neighbours = 30sources
    #
    # The selector is used to construct the target-centred LocalField
    # from the detected source catalogue. The same LocalField is then
    # available to Baseline and SCARLET processing.

    local_field_selector = LocalFieldSelector(
        max_sources=config.photometry.local_field_max_sources,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Differential-photometry reference-source resolution.
    # ------------------------------------------------------------------
    #
    # Differential photometry is the only experiment type that requires
    # an explicit reference-star population.
    #
    # This replaces the former SourceScope.TARGET_AND_REFERENCE check.

    reference_resolver: ReferenceSourceResolver | None = None
    differential_photometer: DifferentialPhotometer | None = None

    if (
        config.experiment_type
        is ExperimentType.DIFFERENTIAL_PHOTOMETRY
    ):
        reference_resolver = ReferenceSourceResolver(
            matching_tolerance=config.evaluation.matching_tolerance,
            logger=logger,
        )

        differential_photometer = DifferentialPhotometer()

    # ------------------------------------------------------------------
    # Aperture photometry.
    # ------------------------------------------------------------------

    photometer = AperturePhotometer(
        config=config.photometry,
    )

    # ------------------------------------------------------------------
    # Evaluation.
    # ------------------------------------------------------------------

    evaluator = PhotometryEvaluator(
        config=config.evaluation,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Compose the complete workflow.
    # ------------------------------------------------------------------
    #
    # ``preprocessor`` is ONLY the primary preprocessing stage
    # (currently BackgroundSubtractor).
    #
    # ``scarlet_preprocessor`` is injected separately so that
    # PhotometricWorkflow can execute:
    #
    #     preprocessing
    #         ↓
    #     detection
    #         ↓
    #     target resolution
    #         ↓
    #     LocalField construction
    #         ↓
    #     Baseline / SCARLET
    #
    # rather than treating a field as a separate experiment type.

    return PhotometricWorkflow(
        config=config,
        acquirer=acquirer,
        builder=builder,
        preprocessor=preprocessor,
        detector=detector,
        photometer=photometer,
        evaluator=evaluator,
        target_resolver=target_resolver,
        local_field_selector=local_field_selector,
        reference_resolver=reference_resolver,
        logger=logger,
        differential_photometer=differential_photometer,
        scarlet_preprocessor=scarlet_preprocessor,
    )
