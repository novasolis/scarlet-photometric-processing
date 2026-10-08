"""
Photometric analysis workflow.

Coordinates execution of the photometric analysis pipeline.

The workflow is intentionally survey-independent. Survey-specific
acquisition and Observation construction are delegated to injected
components, allowing all subsequent processing stages to operate on
the common Observation domain model.

Two production execution modes are supported:

    run()
        Execute the existing single-Observation workflow.

    run_time_series()
        Execute the same scientific processing chain over a sequence
        of survey-provided Observations, such as TESS cadences.

The workflow deliberately keeps preprocessing and source detection
separate from target resolution, reference resolution, photometry and
evaluation. This allows both single-image and time-series execution to
reuse the same scientific processing stages without executing any stage
twice.

Time-series target-resolution behaviour
---------------------------------------
For target-centred time-series experiments, the target must be resolved in each
cadence before target-centred photometry can be performed.

If the target resolver reports that the target can no longer be matched
within its configured tolerance, the time series is considered to have
reached the end of its usable target-resolved interval.

In that specific case:

    * all successfully processed preceding cadences are retained;
    * the current and subsequent cadences are not processed;
    * the time-series workflow completes successfully with the partial
      sequence;
    * the stopping cadence and resolver failure are logged.

This behaviour is restricted to time-series execution and does not alter
single-observation processing.

Reference-source resolution
----------------------------
For differential-photometry experiments, configured differential-
photometry reference stars are resolved independently of target
resolution.

Reference resolution is performed after source detection and before
aperture photometry.

Reference-resolution failures are not treated as target-resolution
termination conditions. They remain fatal workflow errors.

Other processing failures remain fatal. In particular, failures in
preprocessing, detection, photometry, evaluation, invalid Observation
construction, or unexpected resolver errors still raise
PhotometricAnalysisError.

Logging policy
--------------
INFO
    High-level workflow milestones, periodic time-series progress and
    final summaries.

DEBUG
    Detailed acquisition, preprocessing, detection, target-resolution,
    reference-resolution, photometry and evaluation diagnostics.

WARNING / ERROR / EXCEPTION
    Unexpected or degraded conditions requiring attention.

During time-series execution, noisy lower-level component loggers and
third-party SCARLET optimisation loggers are temporarily restricted to
WARNING+ so that cadence processing does not flood the application
terminal. The workflow itself emits concise SCARLET lifecycle diagnostics
through the dedicated time-series logger. Original component logging
levels are restored when the time-series run completes or fails.

A diagnostic execution path is also provided for visual inspection of
intermediate Observation products before photometry. This diagnostic
path does not alter the behaviour of the production workflows.
"""

from __future__ import annotations

import logging

import numpy as np
from dataclasses import is_dataclass, replace
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Callable, cast

from acquisition.base import BaseAcquirer
from core.exceptions import ObservationError, PhotometricAnalysisError
from core.models import ExperimentConfig,PhotometryConfig
from detection.detection import SourceDetector
from domain.builders import ObservationBuilder
from domain.enums import ExperimentType, ProcessingPipeline
from domain.models import (
    DetectedSource,
    DetectionResult,
    Observation,
    ReferenceSourceMatch,
    TargetSourceMatch,
)
from evaluation.base import Evaluator
from photometry.photometry import Photometer
from photometry.differential import DifferentialPhotometer
from preprocessing.preprocessing import Preprocessor
from preprocessing.scarlet import ScarletPreprocessor
from target.local_field import LocalField, LocalFieldSelector
from target.reference_resolver import ReferenceSourceResolver
from target.resolver import TargetSourceResolver
from utils.logger import get_logger
from workflows.models import FieldAnalysisResult, WorkflowResult



class PhotometricWorkflow:
    """
    Coordinate execution of the photometric analysis workflow.

    Parameters
    ----------
    config
        Experiment configuration.

    acquirer
        Survey-specific acquisition implementation.

    builder
        Survey-specific Observation construction implementation.

        ``run()`` requires the standard ``build()`` method.

        ``run_time_series()`` additionally requires the injected builder
        to expose a callable ``build_sequence()`` method.

    preprocessor
        Survey-independent image preprocessing implementation.

    detector
        Survey-independent source detection implementation.

    photometer
        Survey-independent photometry implementation.

    evaluator
        Survey-independent evaluation implementation.

    target_resolver
        Optional target-source resolver.

        Required for every target-centred experiment.

    reference_resolver
        Optional reference-source resolver.

        Required for differential-photometry experiments when differential-
        photometry references are configured.

    differential_photometer
        Optional differential-photometry implementation.

    scarlet_preprocessor
        Optional SCARLET preprocessing stage. When configured, source
        detection is completed before SCARLET and detected source centres
        are supplied for target-centred experiments. target-centred experiments retain
        target-coordinate SCARLET initialisation.

    logger
        Optional logger. If omitted, the project logger is used.
    """

    # ------------------------------------------------------------------
    # Time-series logging configuration
    # ------------------------------------------------------------------

    # Maximum number of periodic INFO progress messages for a time series.
    # The first and final cadence are always reported separately.
    _MAX_PROGRESS_LOG_MESSAGES = 10

    # Lower-level loggers whose INFO output is too verbose for production
    # time-series execution.
    #
    # These are suppressed only while run_time_series() is executing.
    # Their original levels are restored afterward.
    #
    # NOTE:
    #   preprocessing.scarlet is intentionally NOT listed here. The
    #   workflow owns the production-facing SCARLET lifecycle logging,
    #   while detailed preprocessor diagnostics remain available for
    #   diagnostic/debug runs.
    _TIME_SERIES_SUPPRESSED_LOGGER_NAMES = (
        "workflows.workflow_factory",
        "preprocessing.preprocessing",
        "detection.detection",
        "photometry.aperture",
        "photometry.differential",
        "evaluation",
        "evaluation.photometry",
        # Third-party SCARLET optimisation logs.
        "proxmin",
        "scarlet.blend",
    )

    def __init__(
        self,
        config: ExperimentConfig,
        acquirer: BaseAcquirer[Any],
        builder: ObservationBuilder[Any],
        preprocessor: Preprocessor,
        detector: SourceDetector,
        photometer: Photometer,
        evaluator: Evaluator,
        target_resolver: TargetSourceResolver | None = None,
        reference_resolver: ReferenceSourceResolver | None = None,
        logger: logging.Logger | None = None,
        differential_photometer: DifferentialPhotometer | None = None,
        scarlet_preprocessor: ScarletPreprocessor | None = None,
        local_field_selector: LocalFieldSelector | None = None,
    ) -> None:
        """
        Initialise the photometric workflow.
        """

        self._config = config
        self._acquirer = acquirer
        self._builder = builder
        self._preprocessor = preprocessor
        self._detector = detector
        self._photometer = photometer
        self._evaluator = evaluator
        self._target_resolver = target_resolver
        self._reference_resolver = reference_resolver
        self._differential_photometer = differential_photometer
        self._scarlet_preprocessor = scarlet_preprocessor

        # Preserve the injected workflow logger for normal workflow
        # operations and diagnostics.
        self._logger = logger or get_logger(__name__)

        # The local-field selector is an explicit workflow dependency, but
        # a default is retained during the architecture migration so that
        # existing workflow construction remains valid. The production
        # experiment default is 10 total sources, including the target.
        self._local_field_selector = (
            local_field_selector
            or LocalFieldSelector(
                max_sources=config.photometry.local_field_max_sources,
                logger=self._logger,
            )
        )

        # Dedicated logger for concise time-series progress messages.
        #
        # This must be separate from self._logger because the injected
        # logger may be "workflows.workflow_factory", which is one of
        # the verbose loggers temporarily suppressed during TESS
        # time-series processing.
        self._time_series_logger = get_logger(
            "workflows.photometric_workflow.timeseries",
        )

    # ==================================================================
    # Single-observation production workflow
    # ==================================================================

    def run(
        self,
    ) -> WorkflowResult | FieldAnalysisResult:
        """
        Execute the complete single-observation workflow.

        Processing order
        ----------------
        TARGET_PHOTOMETRY / DIFFERENTIAL_PHOTOMETRY
            acquisition
                ↓
            Observation construction
                ↓
            image preprocessing
                ↓
            source detection
                ↓
            target/reference resolution
                ↓
            aperture photometry
                ↓
            optional differential photometry
                ↓
            photometric evaluation

        FIELD_ANALYSIS
            acquisition
                ↓
            Observation construction
                ↓
            image preprocessing
                ↓
            source detection
                ↓
            field-wide neighbour/crowding analysis
                ↓
            return

        Returns
        -------
        WorkflowResult | FieldAnalysisResult
            ``WorkflowResult`` for the existing target-centred
            experiments, or ``FieldAnalysisResult`` for ``FIELD_ANALYSIS``.

        Raises
        ------
        PhotometricAnalysisError
            If workflow execution fails.

        Notes
        -----
        Single-observation execution intentionally retains the existing
        fail-fast behaviour. A target or configured reference that
        cannot be resolved is a genuine workflow failure in this mode
        because there is no subsequent cadence sequence to continue with.
        """

        self._log_header()

        try:
            self._logger.info(
                "Executing single-observation workflow.",
            )

            observation = self._acquire_and_build_observation()

            observation = self._prepare_observation(
                observation,
            )

            if self._config.experiment_type is ExperimentType.FIELD_ANALYSIS:
                result = self._finalize_field_analysis(
                    observation,
                )
            else:
                result = self._finalize_observation(
                    observation,
                )

            self._logger.info(
                "Photometric workflow completed successfully.",
            )

            return result

        except PhotometricAnalysisError:
            raise

        except Exception as exc:
            self._logger.exception(
                "Photometric analysis failed.",
            )

            raise PhotometricAnalysisError(
                (
                    "Photometric workflow failed for "
                    f"'{self._config.target.target_name}'."
                ),
            ) from exc

    # ==================================================================
    # Time-series production workflow
    # ==================================================================

    def run_time_series(
        self,
        *,
        cadence_indices: Sequence[int] | None = None,
    ) -> tuple[WorkflowResult, ...]:
        """
        Execute the workflow over a sequence of observations.

        The dataset is acquired exactly once. The injected Observation
        builder must expose a callable ``build_sequence()`` method that
        converts the acquired dataset into one Observation per selected
        cadence.

        Each cadence follows exactly the same processing chain as the
        single-observation workflow.

        For target-centred experiments, if target resolution is lost at a
        particular cadence, processing stops at that point and all
        successfully completed preceding cadences are returned.

        Parameters
        ----------
        cadence_indices
            Optional zero-based cadence indices passed to the sequence
            builder.

        Returns
        -------
        tuple[WorkflowResult, ...]
            Successfully completed WorkflowResult objects in builder
            sequence order.

        Raises
        ------
        PhotometricAnalysisError
            If sequence construction is unsupported, no Observations
            are produced, or a non-recoverable cadence cannot be
            processed.

        Notes
        -----
        A target-resolution failure is deliberately treated differently
        from other cadence failures.

        The resolver's configured matching tolerance remains unchanged.
        The workflow does not accept a distant source as the target and
        does not alter resolver behaviour.

        Instead, when the resolver reports that the target can no longer
        be located within the configured tolerance, the current cadence
        is treated as the first unusable target-resolved cadence and the
        sequence terminates cleanly.

        Reference-resolution failures are deliberately NOT treated as
        target-resolution termination conditions. A configured reference
        that cannot be resolved represents a failure of the differential-
        photometry experiment and therefore remains fatal.

        Detailed component-level INFO logging is temporarily suppressed
        during cadence processing. WARNING, ERROR and EXCEPTION records
        remain visible.
        """

        self._log_header()

        try:
            self._time_series_logger.info(
                "Executing TESS time-series workflow.",
            )

            self._time_series_logger.info(
                "Acquiring time-series dataset once.",
            )

            dataset = self._acquirer.acquire()

            build_sequence = getattr(
                self._builder,
                "build_sequence",
                None,
            )

            if not callable(
                build_sequence,
            ):
                raise PhotometricAnalysisError(
                    (
                        "Time-series workflow requires the injected "
                        "Observation builder to provide a callable "
                        "'build_sequence()' method."
                    ),
                )

            self._time_series_logger.info(
                "Building Observation sequence.",
            )

            try:
                observations_raw: Any = build_sequence(
                    dataset,
                    cadence_indices=cadence_indices,
                )

            except PhotometricAnalysisError:
                raise

            except TypeError as exc:
                raise PhotometricAnalysisError(
                    (
                        "The configured Observation builder exposes "
                        "'build_sequence()', but it does not accept the "
                        "expected 'cadence_indices' argument."
                    ),
                ) from exc

            except Exception as exc:
                self._time_series_logger.exception(
                    "TESS Observation sequence construction failed.",
                )

                raise PhotometricAnalysisError(
                    "Failed to construct the TESS Observation sequence.",
                ) from exc

            if observations_raw is None:
                raise PhotometricAnalysisError(
                    (
                        "Observation builder returned None for the "
                        "time-series sequence."
                    ),
                )

            observation_sequence = tuple(
                observations_raw,
            )

            if not observation_sequence:
                raise PhotometricAnalysisError(
                    (
                        "Observation builder returned an empty "
                        "time-series sequence."
                    ),
                )

            total_cadences = len(
                observation_sequence,
            )

            self._time_series_logger.info(
                "Observation sequence contains %d cadence(s).",
                total_cadences,
            )

            if self._config.pipeline is ProcessingPipeline.SCARLET:
                self._time_series_logger.info(
                    (
                        "SCARLET preprocessing enabled for TESS time series; "
                        "third-party optimiser logs are suppressed."
                    ),
                )

            progress_interval = (
                self._calculate_progress_interval(
                    total_cadences,
                )
            )

            self._time_series_logger.debug(
                (
                    "Time-series progress interval: "
                    "%d cadence(s)."
                ),
                progress_interval,
            )

            results: list[WorkflowResult] = []

            stopped_by_target_resolution = False
            stopping_position: int | None = None
            stopping_observation_time: Any = None
            stopping_error: Exception | None = None

            # ----------------------------------------------------------
            # Suppress verbose component INFO logging only while the
            # cadence-level processing loop is active.
            #
            # This includes the injected workflows.workflow_factory
            # logger, which may also be self._logger.
            # ----------------------------------------------------------
            with self._time_series_logging_context():

                for position, observation in enumerate(
                    observation_sequence,
                    start=1,
                ):
                    if not isinstance(
                        observation,
                        Observation,
                    ):
                        raise PhotometricAnalysisError(
                            (
                                "Observation builder returned an invalid "
                                "Observation at time-series position "
                                f"{position}."
                            ),
                        )

                    observation_time = (
                        observation.metadata.observation_time
                    )

                    self._time_series_logger.debug(
                        (
                            "Processing cadence %d/%d "
                            "(observation_time=%s)."
                        ),
                        position,
                        total_cadences,
                        observation_time,
                    )

                    try:
                        observation = self._prepare_observation(
                            observation,
                        )

                        result = self._finalize_observation(
                            observation,
                        )

                    except PhotometricAnalysisError as exc:
                        # --------------------------------------------------
                        # Target resolution is the one expected condition
                        # under which a TARGET time series may legitimately
                        # terminate before the end of the supplied
                        # observation sequence.
                        #
                        # _resolve_target_source() deliberately preserves
                        # ObservationError in its cause chain so that this
                        # condition can be identified here.
                        #
                        # Reference-resolution failures are wrapped without
                        # an ObservationError cause and therefore do not
                        # enter this branch.
                        # --------------------------------------------------
                        if self._contains_observation_error(exc):
                            stopped_by_target_resolution = True
                            stopping_position = position
                            stopping_observation_time = observation_time
                            stopping_error = exc

                            self._time_series_logger.warning(
                                (
                                    "Target resolution lost at TESS "
                                    "cadence %d/%d "
                                    "(observation_time=%s). "
                                    "Stopping target-resolved time-series "
                                    "processing and retaining %d "
                                    "successfully processed cadence(s)."
                                ),
                                position,
                                total_cadences,
                                observation_time,
                                len(results),
                            )

                            self._time_series_logger.debug(
                                (
                                    "Target-resolution termination cause "
                                    "for cadence %d/%d."
                                ),
                                position,
                                total_cadences,
                            )

                            break

                        # --------------------------------------------------
                        # Any PhotometricAnalysisError not caused by target
                        # resolution remains fatal.
                        # --------------------------------------------------
                        raise

                    except Exception as exc:
                        # Deliberately use the dedicated logger so the
                        # error remains visible even if the injected
                        # workflow logger is temporarily set to WARNING.
                        self._time_series_logger.exception(
                            (
                                "Processing failed for TESS cadence "
                                "%d of %d."
                            ),
                            position,
                            total_cadences,
                        )

                        raise PhotometricAnalysisError(
                            (
                                "Failed to process TESS cadence "
                                f"{position} of {total_cadences}."
                            ),
                        ) from exc

                    results.append(
                        result,
                    )

                    if (
                        position == 1
                        or position == total_cadences
                        or position % progress_interval == 0
                    ):
                        self._time_series_logger.info(
                            (
                                "TESS time-series progress: "
                                "%d/%d cadences completed."
                            ),
                            position,
                            total_cadences,
                        )

                    self._time_series_logger.debug(
                        (
                            "Completed cadence %d/%d successfully."
                        ),
                        position,
                        total_cadences,
                    )

            if not results:
                if stopped_by_target_resolution:
                    raise PhotometricAnalysisError(
                        (
                            "Target could not be resolved in the first "
                            "TESS cadence; no target-resolved "
                            "time-series results were produced."
                        ),
                    ) from stopping_error

                raise PhotometricAnalysisError(
                    "No time-series workflow results were produced.",
                )

            if stopped_by_target_resolution:
                self._time_series_logger.info(
                    (
                        "TESS target-resolved time series terminated "
                        "at cadence %d/%d. "
                        "Successfully retained %d cadence(s)."
                    ),
                    stopping_position,
                    total_cadences,
                    len(results),
                )

                self._time_series_logger.info(
                    (
                        "Target-resolution stopping time: %s."
                    ),
                    stopping_observation_time,
                )

            self._log_time_series_completion(
                results,
                total_cadences,
                stopped_by_target_resolution=(
                    stopped_by_target_resolution
                ),
                stopping_position=stopping_position,
            )

            return tuple(
                results,
            )

        except PhotometricAnalysisError:
            raise

        except Exception as exc:
            self._time_series_logger.exception(
                "TESS time-series photometric analysis failed.",
            )

            raise PhotometricAnalysisError(
                (
                    "TESS time-series workflow failed for "
                    f"'{self._config.target.target_name}'."
                ),
            ) from exc

    # ==================================================================
    # Diagnostic workflow
    # ==================================================================

    def run_until_detection(
        self,
    ) -> Observation:
        """
        Execute the workflow through source detection only.

        Intended for diagnostic and visual inspection.

        The returned Observation contains the acquired image,
        preprocessing products and source-detection result.
        """

        self._log_header()

        try:
            self._logger.info(
                "Executing diagnostic workflow through source detection.",
            )

            observation = self._acquire_and_build_observation()

            observation = self._prepare_observation(
                observation,
            )

            return observation

        except PhotometricAnalysisError:
            raise

        except Exception as exc:
            self._logger.exception(
                "Diagnostic observation preparation failed.",
            )

            raise PhotometricAnalysisError(
                (
                    "Diagnostic workflow failed for "
                    f"'{self._config.target.target_name}'."
                ),
            ) from exc

    # ==================================================================
    # Acquisition and domain construction
    # ==================================================================

    def _acquire_and_build_observation(
        self,
    ) -> Observation:
        """
        Acquire one dataset and construct one Observation.

        No preprocessing, detection, photometry or evaluation occurs
        here.
        """

        self._logger.debug(
            "Acquiring observation dataset.",
        )

        try:
            dataset = self._acquirer.acquire()

        except PhotometricAnalysisError:
            raise

        except Exception as exc:
            self._logger.exception(
                "Observation acquisition failed.",
            )

            raise PhotometricAnalysisError(
                (
                    "Failed to acquire observation data for "
                    f"'{self._config.target.target_name}'."
                ),
            ) from exc

        self._logger.debug(
            "Observation dataset acquired successfully.",
        )

        self._logger.debug(
            "Building Observation domain model.",
        )

        try:
            observation = self._builder.build(
                dataset,
            )

        except PhotometricAnalysisError:
            raise

        except Exception as exc:
            self._logger.exception(
                "Observation domain construction failed.",
            )

            raise PhotometricAnalysisError(
                "Failed to construct the Observation domain model.",
            ) from exc

        if not isinstance(
            observation,
            Observation,
        ):
            raise PhotometricAnalysisError(
                "Observation builder returned an invalid Observation.",
            )

        self._logger.debug(
            (
                "Observation domain model constructed successfully "
                "(observation_time=%s)."
            ),
            observation.metadata.observation_time,
        )

        return observation

    # ==================================================================
    # Shared preparation
    # ==================================================================

    def _prepare_observation(
        self,
        observation: Observation,
    ) -> Observation:
        """
        Prepare an observation through primary preprocessing and detection.

        Processing order
        ----------------
        primary preprocessing
            ↓
        source detection

        Target resolution and LocalField construction deliberately occur
        after this method. SCARLET is also invoked later, after the target
        has been resolved and its target-centred LocalField has been
        constructed. This ensures Baseline and SCARLET operate on the same
        experiment context.

        Notes
        -----
        ``self._preprocessor`` is the primary preprocessing stage.
        ``self._detector`` then produces the complete detected-source
        catalogue used by target resolution and LocalField selection.
        """

        if not isinstance(
            observation,
            Observation,
        ):
            raise PhotometricAnalysisError(
                (
                    "Workflow received an invalid Observation "
                    "for preparation."
                ),
            )

        try:
            self._logger.debug(
                "Running primary image preprocessing.",
            )

            observation = self._preprocessor.process(
                observation,
            )

            self._logger.debug(
                "Primary image preprocessing completed.",
            )

            self._logger.debug(
                "Running source detection.",
            )

            observation = self._detector.detect(
                observation,
            )

            self._logger.debug(
                "Source detection completed.",
            )

            if observation.detections is not None:
                self._logger.debug(
                    "Detected %d source(s).",
                    len(
                        observation.detections.sources,
                    ),
                )
            else:
                self._logger.debug(
                    "Source detection produced no detection result.",
                )

        except PhotometricAnalysisError:
            raise

        except Exception as exc:
            self._logger.exception(
                "Observation preparation failed.",
            )

            raise PhotometricAnalysisError(
                (
                    "Failed during preprocessing or source detection "
                    f"for '{self._config.target.target_name}'."
                ),
            ) from exc

        return observation

    # ==================================================================
    # Shared finalisation
    # ==================================================================

    def _finalize_field_analysis(
        self,
        observation: Observation,
    ) -> FieldAnalysisResult:
        """
        Finalise a field-centred analysis after normal observation
        preparation and source detection.

        FIELD_ANALYSIS deliberately does not resolve the configured
        target coordinates to a detected source. ``target_name`` is
        retained by the experiment configuration as the field/target
        coordinate identifier used during acquisition.

        The existing preparation path is preserved, including primary
        preprocessing/background handling and source detection. After
        detection, the existing neighbour-count algorithm is applied
        to the complete detected-source catalogue and the workflow
        returns immediately.

        No target resolution, LocalField selection, reference-source
        resolution, SCARLET processing, aperture photometry,
        differential photometry or photometric evaluation is performed.
        """

        if not isinstance(
            observation,
            Observation,
        ):
            raise PhotometricAnalysisError(
                (
                    "Workflow received an invalid Observation "
                    "for field analysis."
                ),
            )

        detections = observation.detections

        if detections is None:
            raise PhotometricAnalysisError(
                (
                    "FIELD_ANALYSIS requires a completed "
                    "source-detection result."
                ),
            )

        try:
            calculate_neighbour_counts = getattr(
                self._evaluator,
                "calculate_neighbour_counts",
                None,
            )

            if not callable(calculate_neighbour_counts):
                raise PhotometricAnalysisError(
                    (
                        "FIELD_ANALYSIS requires the configured "
                        "evaluator to provide "
                        "'calculate_neighbour_counts()'."
                    ),
                )

            calculate = cast(
                Callable[
                    [tuple[DetectedSource, ...]],
                    dict[str, int],
                ],
                calculate_neighbour_counts,
            )

            neighbour_counts = calculate(
                detections.sources,
            )

            self._logger.info(
                (
                    "FIELD_ANALYSIS completed successfully: "
                    "%d detected source(s), neighbour counts calculated "
                    "for the complete field."
                ),
                len(detections.sources),
            )

            return FieldAnalysisResult(
                observation=observation,
                neighbour_counts=neighbour_counts,
            )

        except PhotometricAnalysisError:
            raise

        except Exception as exc:
            self._logger.exception(
                "FIELD_ANALYSIS crowding analysis failed.",
            )

            raise PhotometricAnalysisError(
                (
                    "Failed during FIELD_ANALYSIS crowding analysis "
                    f"for '{self._config.target.target_name}'."
                ),
            ) from exc

    def _finalize_observation(
        self,
        observation: Observation,
    ) -> WorkflowResult:
        """
        Finalise a prepared observation as a target-centred experiment.

        Processing order
        ----------------
        target-source resolution
            ↓
        LocalField selection
            ↓
        reference-source resolution (differential experiments only)
            ↓
        LocalField analysis view creation
            ↓
        optional SCARLET processing using LocalField source centres
            ↓
        aperture photometry
            ↓
        optional differential photometry
            ↓
        photometric evaluation

        Every completed experiment therefore has the same core analytical
        context: a resolved target and its target-centred LocalField. The
        complete detector catalogue is used only for target/reference
        association and LocalField selection. From the LocalField analysis
        view onward, SCARLET (when configured), aperture photometry,
        differential photometry and evaluation operate only on the selected
        target + LocalField sources. Baseline and SCARLET therefore share
        the same source population without executing any analytical stage
        twice.
        """

        if not isinstance(
            observation,
            Observation,
        ):
            raise PhotometricAnalysisError(
                (
                    "Workflow received an invalid Observation "
                    "for finalisation."
                ),
            )

        try:
            # --------------------------------------------------------------
            # Every experiment is target-centred.
            # --------------------------------------------------------------
            target_match = self._resolve_target_source(
                observation,
            )

            # --------------------------------------------------------------
            # Construct the target-centred LocalField from the complete
            # detection catalogue. This is common to Baseline and SCARLET.
            # --------------------------------------------------------------
            detections = observation.detections

            if detections is None:
                raise PhotometricAnalysisError(
                    (
                        "LocalField selection requires a completed "
                        "source-detection result."
                    ),
                )

            self._logger.debug(
                (
                    "Selecting target-centred LocalField for target '%s' "
                    "with maximum size %d."
                ),
                target_match.target_name,
                self._local_field_selector.max_sources,
            )

            local_field = self._local_field_selector.select(
                detections=detections,
                target_match=target_match,
            )

            self._logger.debug(
                (
                    "LocalField selected: target='%s', sources=%d, "
                    "neighbours=%d, radius=%.3f pixels."
                ),
                local_field.target_source.identifier,
                local_field.selected_source_count,
                local_field.selected_neighbour_count,
                local_field.selection_radius_pixels,
            )

            # Reference resolution must still see the complete detection
            # catalogue. It is an association step, not a photometric
            # measurement step. Once target/local-field selection is
            # complete, all analytical processing below operates only on
            # the selected target + local-field sources.
            reference_matches = self._resolve_reference_sources(
                observation,
            )

            full_detection_result = observation.detections
            if full_detection_result is None:
                # This is already guarded above, but keeping the invariant
                # explicit here prevents a partially-scoped observation from
                # ever entering downstream processing.
                raise PhotometricAnalysisError(
                    "LocalField processing requires a completed source-detection result.",
                )

            analysis_observation: Observation
            restore_full_detections: Callable[
                [Observation, DetectionResult],
                None,
            ]

            (
                analysis_observation,
                restore_full_detections,
            ) = self._create_local_field_observation(
                observation=observation,
                local_field=local_field,
            )

            try:
                # ----------------------------------------------------------
                # From this point onward the workflow deliberately exposes
                # only the target + selected local-field sources. The full
                # detection catalogue remains available to the workflow
                # only as provenance and is restored on the final result.
                # ----------------------------------------------------------
                # ----------------------------------------------------------
                # From this point onward the analysis view contains only
                # the target + selected LocalField sources. SCARLET is run
                # exactly once, on that scoped observation, before aperture
                # photometry. There is deliberately no pre-scope SCARLET
                # execution.
                # ----------------------------------------------------------
                if self._config.pipeline is ProcessingPipeline.SCARLET:
                    if self._scarlet_preprocessor is None:
                        raise PhotometricAnalysisError(
                            (
                                "SCARLET processing is configured, but no "
                                "ScarletPreprocessor was provided to the "
                                "workflow."
                            ),
                        )

                    self._time_series_logger.debug(
                        (
                            "SCARLET preprocessing started for cadence "
                            "(target='%s', LocalField sources=%d)."
                        ),
                        target_match.target_name,
                        len(local_field.source_centers),
                    )

                    analysis_observation = self._scarlet_preprocessor.process(
                        analysis_observation,
                        source_centers=local_field.source_centers,
                    )

                    self._time_series_logger.debug(
                        "SCARLET preprocessing completed for cadence.",
                    )

                self._logger.debug(
                    "Running aperture photometry on %d LocalField source(s).",
                    local_field.selected_source_count,
                )

                if self._config.pipeline is ProcessingPipeline.SCARLET:
                    source_images = self._build_scarlet_source_images(
                        observation=analysis_observation,
                        local_field=local_field,
                    )

                    try:
                        analysis_observation = (
                            self._photometer.measure_source_images(
                                analysis_observation,
                                source_images,
                            )
                        )
                    except NotImplementedError as exc:
                        raise PhotometricAnalysisError(
                            (
                                "SCARLET processing requires the configured "
                                "photometer to implement source-specific "
                                "measurement. It must not fall back to the "
                                "blended active image."
                            ),
                        ) from exc
                else:
                    analysis_observation = self._photometer.measure(
                        analysis_observation,
                    )

                self._logger.debug(
                    "Aperture photometry completed for LocalField sources.",
                )

                differential_photometry = (
                    self._calculate_differential_photometry(
                        observation=analysis_observation,
                        target_match=target_match,
                        reference_matches=reference_matches,
                    )
                )

                self._logger.debug(
                    "Running photometric evaluation on LocalField sources.",
                )

                evaluation = self._evaluator.evaluate(
                    analysis_observation,
                )

                self._logger.debug(
                    "Photometric evaluation completed.",
                )

                self._log_evaluation_debug(
                    evaluation,
                )

            finally:
                # Restore the complete detection catalogue on the returned
                # observation so the full detection remains available for
                # diagnostics, visualisation and provenance. This does not
                # alter the fact that SCARLET, photometry, differential
                # photometry and evaluation all operated on the LocalField
                # population only.
                restore_full_detections(
                    analysis_observation,
                    full_detection_result,
                )

            observation = analysis_observation

            return WorkflowResult(
                observation=observation,
                evaluation=evaluation,
                target_match=target_match,
                local_field=local_field,
                reference_matches=reference_matches,
                differential_photometry=differential_photometry,
            )

        except PhotometricAnalysisError:
            raise

        except Exception as exc:
            self._logger.exception(
                "Observation finalisation failed.",
            )

            raise PhotometricAnalysisError(
                (
                    "Failed during target resolution, LocalField "
                    "selection, SCARLET processing, reference resolution, "
                    "photometry or evaluation for "
                    f"'{self._config.target.target_name}'."
                ),
            ) from exc

    def _create_local_field_observation(
        self,
        *,
        observation: Observation,
        local_field: LocalField,
    ) -> tuple[
        Observation,
        Callable[[Observation, DetectionResult], None],
    ]:
        """
        Create the downstream analysis view containing only LocalField sources.

        The complete detection catalogue is intentionally retained on the
        original observation. A shallow Observation copy is used for normal
        production execution so that downstream components can safely treat
        ``observation.detections`` as the analysis population without losing
        the authoritative full detection result. The returned analysis view
        is the only Observation passed to SCARLET, aperture photometry,
        differential photometry and evaluation.

        A small mutable fallback is retained for lightweight test doubles
        such as ``MagicMock(spec=Observation)`` used by the existing workflow
        tests.
        """

        detections = observation.detections
        if detections is None:
            raise PhotometricAnalysisError(
                "LocalField processing requires a completed source-detection result.",
            )

        scoped_detections = cast(
            DetectionResult,
            replace(
                detections,
                sources=local_field.sources,
            ),
        )

        if is_dataclass(observation):
            scoped_observation = cast(
                Observation,
                replace(
                    observation,
                    detections=scoped_detections,
                ),
            )

            def restore(
                current: Observation,
                full: DetectionResult,
            ) -> None:
                current.detections = full

            self._logger.debug(
                (
                    "Created LocalField analysis view: %d source(s) "
                    "from %d detected source(s)."
                ),
                local_field.selected_source_count,
                len(detections.sources),
            )

            return scoped_observation, restore

        # Existing unit tests use lightweight mocks rather than real
        # Observation dataclass instances. Preserve their behaviour without
        # weakening the production architecture.
        original_observation = observation
        original_detections = observation.detections
        observation.detections = scoped_detections

        def restore_mock(
            current: Observation,
            full: DetectionResult,
        ) -> None:
            original_observation.detections = full
            if current is not original_observation:
                current.detections = full

        return observation, restore_mock

    def _build_scarlet_source_images(
        self,
        *,
        observation: Observation,
        local_field: LocalField,
    ) -> dict[str, np.ndarray]:
        """
        Construct one source-specific deblended observation per LocalField source.

        For source ``i`` the image is defined as:

            I_deblended,i = I_residual + I_source,i

        where ``I_residual = I_observed - I_field_model`` and
        ``I_source,i`` is the observation-space PSF-rendered SCARLET model
        for that source. Because the field model is the sum of the individual
        source models, this is algebraically equivalent to subtracting all
        fitted neighbours while preserving the target/source model itself.

        The residual is deliberately retained rather than measuring the
        noiseless SCARLET source model directly. This keeps the photometric
        measurement in observation space and preserves image-domain noise and
        model residuals for the subsequent local-background estimation.
        """
        field_residual_image = observation.get_image(
            "scarlet_field_residual"
        )

        residual = np.asarray(
            field_residual_image.image.data,
            dtype=float,
        )

        if residual.ndim != 2 or residual.size == 0:
            raise PhotometricAnalysisError(
                "SCARLET field residual is not a valid two-dimensional image."
            )

        source_images: dict[str, np.ndarray] = {}

        for selected_index, source in enumerate(local_field.sources):
            product_name = f"scarlet_source_{selected_index:04d}"

            image_data = observation.images.get(product_name)

            if image_data is None:
                self._logger.warning(
                    "No SCARLET source model product found for LocalField "
                    "source '%s' at selected index %d.",
                    source.identifier,
                    selected_index,
                )
                continue

            source_model = np.asarray(
                image_data.image.data,
                dtype=float,
            )

            if source_model.shape != residual.shape:
                raise PhotometricAnalysisError(
                    "SCARLET source model and field residual shapes differ "
                    f"for source '{source.identifier}': "
                    f"source={source_model.shape}, residual={residual.shape}."
                )

            if not np.isfinite(source_model).any():
                self._logger.warning(
                    "SCARLET source model for source '%s' contains no "
                    "finite pixels; measurement will be marked invalid.",
                    source.identifier,
                )
                continue

            source_images[source.identifier] = residual + source_model

        self._logger.debug(
            "Constructed %d/%d SCARLET source-specific deblended images.",
            len(source_images),
            local_field.selected_source_count,
        )

        return source_images

    # ==================================================================
    # Differential photometry
    # ==================================================================

    def _calculate_differential_photometry(
        self,
        *,
        observation: Observation,
        target_match: TargetSourceMatch,
        reference_matches: tuple[ReferenceSourceMatch, ...],
    ):
        """
        Calculate differential photometry when configured.

        Differential photometry operates on completed aperture-
        photometry measurements and is therefore performed after
        target/reference resolution and aperture photometry.

        Differential-photometry failures are deliberately converted to
        PhotometricAnalysisError without preserving ObservationError as
        the explicit cause. The time-series controller reserves an
        explicit ObservationError cause for target-resolution loss.
        """

        if self._config.experiment_type is not ExperimentType.DIFFERENTIAL_PHOTOMETRY:
            self._logger.debug(
                "Differential photometry not required for this experiment type.",
            )
            return None

        if not reference_matches:
            raise PhotometricAnalysisError(
                (
                    "DIFFERENTIAL_PHOTOMETRY experiments require at least "
                    "one resolved reference source before differential "
                    "photometry."
                ),
            )

        if self._differential_photometer is None:
            raise PhotometricAnalysisError(
                (
                    "DIFFERENTIAL_PHOTOMETRY experiments require a "
                    "DifferentialPhotometer."
                ),
            )

        self._logger.debug(
            (
                "Running differential photometry for target '%s' "
                "using %d reference source(s)."
            ),
            target_match.target_name,
            len(reference_matches),
        )

        photometry = observation.photometry

        if photometry is None:
            raise PhotometricAnalysisError(
                (
                    "Differential photometry requires completed aperture "
                    "photometry, but no photometry result is available."
                ),
            )

        try:
            result = self._differential_photometer.calculate(
                photometry=photometry,
                target_match=target_match,
                reference_matches=reference_matches,
            )

        except ObservationError:
            self._logger.exception(
                "Differential photometry calculation failed.",
            )

            # Do not preserve ObservationError as the explicit cause.
            # The time-series controller uses that cause chain to
            # identify target-resolution termination.
            raise PhotometricAnalysisError(
                (
                    "Failed to calculate differential photometry for "
                    f"'{self._config.target.target_name}'."
                ),
            ) from None

        except PhotometricAnalysisError:
            raise

        except Exception as exc:
            self._logger.exception(
                "Unexpected differential photometry failure.",
            )

            raise PhotometricAnalysisError(
                (
                    "Failed to calculate differential photometry for "
                    f"'{self._config.target.target_name}'."
                ),
            ) from exc

        self._logger.debug(
            (
                "Differential photometry completed: target='%s', "
                "references=%d, differential_flux=%.8f, "
                "differential_magnitude=%.8f."
            ),
            result.target_source_identifier,
            result.valid_reference_count,
            result.differential_flux,
            result.differential_magnitude,
        )

        return result

    # ==================================================================
    # Target resolution
    # ==================================================================

    def _resolve_target_source(
        self,
        observation: Observation,
    ) -> TargetSourceMatch:
        """
        Resolve the configured target against the detected sources.

        Target resolution is mandatory for every experiment. The resolver
        remains authoritative regarding matching tolerance and this method
        never substitutes a nearby source when the configured target cannot
        be resolved.
        """

        if self._target_resolver is None:
            raise PhotometricAnalysisError(
                (
                    "Every target-centred experiment requires a "
                    "TargetSourceResolver, but none was configured."
                ),
            )

        self._logger.debug(
            "Resolving target '%s' against detected sources.",
            self._config.target.target_name,
        )

        try:
            match = self._target_resolver.resolve(
                observation,
            )

        except PhotometricAnalysisError:
            raise

        except Exception as exc:
            self._logger.exception(
                "Target-source resolution failed.",
            )

            raise PhotometricAnalysisError(
                (
                    "Failed to resolve target "
                    f"'{self._config.target.target_name}'."
                ),
            ) from exc

        self._logger.debug(
            (
                "Target '%s' resolved to source '%s' "
                "(separation=%.3f pixels)."
            ),
            match.target_name,
            match.source_identifier,
            match.separation_pixels,
        )

        return match

    # ==================================================================
    # Reference resolution
    # ==================================================================

    def _resolve_reference_sources(
        self,
        observation: Observation,
    ) -> tuple[ReferenceSourceMatch, ...]:
        """
        Resolve configured differential-photometry reference stars.

        Reference resolution is an association stage only.
        The resolved matches are returned to the workflow finalisation
        stage and persisted in WorkflowResult.

        This method intentionally performs no photometry and does not
        calculate differential flux.

        Raises
        ------
        PhotometricAnalysisError
            If DIFFERENTIAL_PHOTOMETRY is configured without differential-
            photometry configuration, if the reference resolver is not
            available, or if any configured reference cannot be resolved.
        """

        if self._config.experiment_type is not ExperimentType.DIFFERENTIAL_PHOTOMETRY:
            self._logger.debug(
                "Reference-source resolution not required for this experiment type.",
            )

            return ()

        differential_config = self._config.differential_photometry

        if differential_config is None:
            raise PhotometricAnalysisError(
                (
                    "DIFFERENTIAL_PHOTOMETRY experiments require "
                    "differential-photometry configuration."
                ),
            )

        if self._reference_resolver is None:
            raise PhotometricAnalysisError(
                (
                    "DIFFERENTIAL_PHOTOMETRY experiments require a "
                    "ReferenceSourceResolver."
                ),
            )

        reference_matches: list[ReferenceSourceMatch] = []

        for reference in differential_config.reference_stars:
            try:
                match = self._reference_resolver.resolve(
                    observation,
                    reference,
                )

            except ObservationError as exc:
                self._logger.exception(
                    (
                        "Reference-source resolution failed for "
                        "reference '%s'."
                    ),
                    reference.identifier,
                )

                # Do NOT preserve ObservationError in the cause chain.
                #
                # ObservationError is reserved by the time-series
                # controller as the target-resolution termination signal.
                #
                # A reference-resolution failure is a fatal differential-
                # photometry error, not a valid reason to truncate a
                # target-resolved time series.
                raise PhotometricAnalysisError(
                    (
                        "Failed to resolve differential-photometry "
                        f"reference '{reference.identifier}'."
                    ),
                ) from None

            except PhotometricAnalysisError:
                raise

            except Exception as exc:
                self._logger.exception(
                    (
                        "Unexpected reference-source resolution "
                        "failure for reference '%s'."
                    ),
                    reference.identifier,
                )

                raise PhotometricAnalysisError(
                    (
                        "Failed to resolve differential-photometry "
                        f"reference '{reference.identifier}'."
                    ),
                ) from exc

            reference_matches.append(match)

            self._logger.debug(
                (
                    "Reference '%s' resolved to source '%s' "
                    "(separation=%.3f pixels)."
                ),
                match.reference_identifier,
                match.source_identifier,
                match.separation_pixels,
            )

        self._logger.debug(
            (
                "Resolved %d differential-photometry reference "
                "source(s)."
            ),
            len(reference_matches),
        )

        return tuple(reference_matches)

    # ==================================================================
    # Target-resolution failure classification
    # ==================================================================

    @staticmethod
    def _contains_observation_error(
        exc: BaseException,
    ) -> bool:
        """
        Determine whether an exception was ultimately caused by
        ObservationError.

        Target-source resolution deliberately preserves ObservationError
        as the explicit cause of the resulting PhotometricAnalysisError.
        Reference-source resolution suppresses that cause.

        The time-series controller therefore follows only the explicit
        exception-cause chain. Exception context is intentionally ignored
        because it may represent an unrelated failure path and must not
        cause a cadence to be incorrectly classified as target loss.

        Parameters
        ----------
        exc
            Exception raised during cadence processing.

        Returns
        -------
        bool
            True when ObservationError appears in the explicit cause
            chain; otherwise False.
        """

        current: BaseException | None = exc
        visited: set[int] = set()

        while current is not None:
            current_id = id(current)

            if current_id in visited:
                break

            visited.add(current_id)

            if isinstance(current, ObservationError):
                return True

            current = current.__cause__

        return False

    # ==================================================================
    # Time-series logging context
    # ==================================================================

    @contextmanager
    def _time_series_logging_context(
        self,
    ) -> Iterator[None]:
        """
        Temporarily suppress verbose lower-level INFO logging.

        The following component/third-party loggers are restricted to
        WARNING+ while cadence-level processing is active:

            * workflows.workflow_factory
            * preprocessing.preprocessing
            * detection.detection
            * photometry.aperture
            * photometry.differential
            * evaluation.*
            * proxmin
            * scarlet.blend

        ``preprocessing.scarlet`` is intentionally NOT suppressed. The
        workflow controls production-facing SCARLET messages through
        ``_time_series_logger`` while detailed preprocessor diagnostics
        remain available for diagnostic/debug runs.

        ``target.resolver`` is intentionally NOT suppressed.

        Its per-cadence target-resolution diagnostics remain useful for
        diagnosing target motion, source detection loss and the exact
        cadence at which target resolution becomes impossible.

        Their exact logger levels are captured and restored when the
        context exits, even when processing raises an exception.

        This context affects logging only. It has no impact on scientific
        processing or evaluation.
        """

        loggers = [
            logging.getLogger(
                logger_name,
            )
            for logger_name in (
                *self._TIME_SERIES_SUPPRESSED_LOGGER_NAMES,
            )
        ]

        original_levels = {
            logger.name: logger.level
            for logger in loggers
        }

        try:
            for logger in loggers:
                logger.setLevel(
                    logging.WARNING,
                )

            yield

        finally:
            for logger in loggers:
                logger.setLevel(
                    original_levels[
                        logger.name
                    ],
                )

    # ==================================================================
    # Logging helpers
    # ==================================================================

    def _log_header(
        self,
    ) -> None:
        """
        Log a concise workflow header.

        The header is intentionally INFO-level because it provides useful
        context without generating per-cadence noise.
        """

        separator = "=" * 72

        self._logger.info(
            "",
        )

        self._logger.info(
            separator,
        )

        self._logger.info(
            "PHOTOMETRIC ANALYSIS PIPELINE",
        )

        self._logger.info(
            separator,
        )

        self._logger.info(
            "Experiment : %s",
            self._config.experiment_type.value,
        )

        self._logger.info(
            "Survey     : %s",
            self._config.survey.value,
        )

        self._logger.info(
            "Target     : %s",
            self._config.target.target_name,
        )

        self._logger.info(
            "Pipeline   : %s",
            self._config.pipeline.value,
        )

        self._logger.info(
            separator,
        )

    def _log_evaluation_debug(
        self,
        evaluation: Any,
    ) -> None:
        """
        Log detailed evaluation information at DEBUG level only.

        This keeps per-cadence diagnostics available without flooding
        production INFO logs.
        """

        self._logger.debug(
            "Evaluation detected sources: %d.",
            evaluation.detected_source_count,
        )

        self._logger.debug(
            "Evaluation measurement records: %d.",
            evaluation.measured_source_count,
        )

        self._logger.debug(
            "Evaluation valid measurements: %d.",
            evaluation.valid_measurement_count,
        )

        self._logger.debug(
            "Evaluation invalid measurements: %d.",
            evaluation.invalid_measurement_count,
        )

        self._logger.debug(
            "Evaluation missing measurements: %d.",
            evaluation.missing_measurement_count,
        )

        self._logger.debug(
            "Evaluation completeness: %.2f%%.",
            evaluation.measurement_completeness * 100.0,
        )

        self._logger.debug(
            "Evaluation valid fraction: %.2f%%.",
            evaluation.valid_measurement_fraction * 100.0,
        )

        self._logger.debug(
            "Evaluation non-positive flux: %d.",
            evaluation.non_positive_flux_count,
        )

        self._logger.debug(
            "Evaluation crowded sources: %d.",
            evaluation.crowded_source_count,
        )

        self._logger.debug(
            "Evaluation crowding fraction: %.2f%%.",
            evaluation.crowding_fraction * 100.0,
        )

    def _calculate_progress_interval(
        self,
        total_cadences: int,
    ) -> int:
        """
        Calculate a sensible INFO-level time-series progress interval.

        The returned interval produces no more than approximately
        ``_MAX_PROGRESS_LOG_MESSAGES`` periodic progress messages,
        while always allowing the first and final cadence to be logged.
        """

        if total_cadences <= 1:
            return 1

        return max(
            1,
            total_cadences
            // self._MAX_PROGRESS_LOG_MESSAGES,
        )

    def _log_time_series_completion(
        self,
        results: Sequence[WorkflowResult],
        total_cadences: int,
        *,
        stopped_by_target_resolution: bool = False,
        stopping_position: int | None = None,
    ) -> None:
        """
        Log a concise final time-series summary.

        The method reports only already-produced evaluation values.
        It performs no scientific recalculation.
        """

        valid_measurements = 0
        invalid_measurements = 0
        measured_sources = 0
        detected_sources = 0

        for result in results:
            evaluation = result.evaluation

            detected_sources += int(
                evaluation.detected_source_count,
            )

            measured_sources += int(
                evaluation.measured_source_count,
            )

            valid_measurements += int(
                evaluation.valid_measurement_count,
            )

            invalid_measurements += int(
                evaluation.invalid_measurement_count,
            )

        if stopped_by_target_resolution:
            completion_message = (
                "TESS time-series workflow completed with "
                "target-resolution termination: "
                f"{len(results)}/{total_cadences} cadences processed "
                f"(stopped at cadence {stopping_position})."
            )
        else:
            completion_message = (
                "TESS time-series workflow completed successfully: "
                f"{len(results)}/{total_cadences} cadences processed."
            )

        self._time_series_logger.info(
            completion_message,
        )

        self._time_series_logger.info(
            (
                "Time-series totals: detected=%d, measured=%d, "
                "valid=%d, invalid=%d."
            ),
            detected_sources,
            measured_sources,
            valid_measurements,
            invalid_measurements,
        )