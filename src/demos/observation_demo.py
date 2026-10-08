"""
Observation demonstration.

Production-facing visual inspection entry point for the photometric-analysis
framework.

The demonstration executes the configured production workflow and presents
the resulting Observation and analytical products. It deliberately keeps
scientific processing inside the production workflow: this module performs
orchestration, diagnostic reporting and presentation only.

Supported demonstration modes
------------------------------
Single-observation experiments
    Execute ``workflow.run()`` and render the representative observation.

Time-series experiments
    Execute ``workflow.run_time_series()`` and render:
        * the first completed cadence as the representative image;
        * the target net-flux light curve, when a target is resolved;
        * the normalised target light curve;
        * the differential-photometry light curve for
          TARGET_AND_REFERENCE experiments.

Experiment awareness
---------------------
Target-centred experiments use the completed workflow's authoritative
``LocalField``.

``FIELD_ANALYSIS`` is different: ``target_name`` identifies the field
coordinates, not a source that must be resolved. The demonstration therefore
uses the completed ``FieldAnalysisResult`` directly and presents the full detected field plus its existing crowding/neighbour analysis,
with a bounded crowded-source candidate pool used for detailed diagnostics.
It also identifies a small presentation-only shortlist of isolated sources
for later differential-photometry reference-star selection; these candidates
are rendered in the same crowded-source figure with distinct styling by the
visualizer.

SCARLET visualisations are shown only when the configured processing
pipeline is ``ProcessingPipeline.SCARLET``. Differential light curves are
only attempted when the completed workflow actually provides
differential-photometry results.

Scientific integrity
--------------------
This module does not:
    * perform source detection;
    * perform aperture photometry;
    * perform production target/reference resolution; catalogue matching here is diagnostic only;
    * calculate evaluation metrics;
    * calculate differential photometry;
    * filter, smooth, detrend or sigma-clip time-series data.

All scientific values displayed here are copied from completed workflow
results.

SCARLET field products are presentation-only outputs already produced by
the production preprocessor. This module does not fit, reconstruct or
modify SCARLET models; it only renders the stored reconstruction, residual,
intrinsic model when available, and individual-source products.

Catalogue cross-matching is an optional diagnostic operation. It is never
part of the production workflow and failures do not invalidate a successful
photometric result.
"""

from __future__ import annotations

import logging
import copy
import inspect

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from core.models import ExperimentConfig
from domain.enums import ExperimentType, ProcessingPipeline
from domain.models import DetectionResult
from utils.catalogue_crossmatch import CatalogueCrossmatch
from utils.crowded_source_coordinates import CrowdedSourceCoordinates
from utils.crowded_source_extractor import CrowdedSourceExtractor
from utils.logger import get_logger
from utils.observation_inspector import ObservationInspector
from utils.observation_visualizer import ObservationVisualizer
from workflows.models import FieldAnalysisResult, WorkflowResult
from workflows.workflow_factory import create_workflow


logger = get_logger(__name__)


# ============================================================================
# Public demonstration API
# ============================================================================


def demonstrate(
    config: ExperimentConfig,
    *,
    run_catalogue_crossmatch: bool = False,
    show_scarlet_field_selection: bool = True,
    scarlet_field_max_sources: int | None = 30,
) -> None:
    """
    Execute and visually inspect one configured experiment.

    Parameters
    ----------
    config
        Complete experiment configuration.

    run_catalogue_crossmatch
        If True, the shortlisted field-analysis source coordinates are
        cross-matched against the catalogue appropriate to the configured
        survey. This is
        diagnostic only and requires external catalogue access.

    Notes
    -----
    Time-series execution is selected from ``config.time_series.enabled``.
    The caller is responsible for applying any demo-only cadence limit.
    ``src/demo.py`` performs that limiting without changing the production
    experiment configuration.
    """
    _configure_demo_logging()

    _validate_scarlet_demo_options(
        show_scarlet_field_selection=show_scarlet_field_selection,
        scarlet_field_max_sources=scarlet_field_max_sources,
    )

    logger.info(
        "Starting observation demonstration for target '%s'.",
        config.target.target_name,
    )

    _log_experiment_configuration(config)

    workflow = create_workflow(config)

    # ------------------------------------------------------------------
    # FIELD_ANALYSIS is a separate result type and therefore gets a
    # separate execution/presentation path. Keeping this branch isolated
    # prevents the shared target-photometry path from becoming a union type.
    # ------------------------------------------------------------------

    if _is_field_analysis_experiment(config):
        field_result = workflow.run()

        if not isinstance(field_result, FieldAnalysisResult):
            raise RuntimeError(
                "FIELD_ANALYSIS workflow returned an unexpected result type: "
                f"{type(field_result).__name__}."
            )

        logger.info(
            "FIELD_ANALYSIS workflow completed; using the field result "
            "directly for source/crowding diagnostics.",
        )

        visualizer = ObservationVisualizer(logger)

        candidate_coordinates = _report_field_analysis_sources(
            result=field_result,
        )

        # Differential-photometry candidates are deliberately identified
        # separately from crowded-source candidates, but they are rendered
        # in the same FIELD_ANALYSIS figure. The scientific source catalogue
        # and neighbour counts remain authoritative; this is presentation-
        # level candidate ranking only.
        differential_coordinates = _report_differential_photometry_candidates(
            result=field_result,
            config=config,
        )

        # Also report a small diagnostic shortlist of potentially non-crowded
        # science targets. This is reconnaissance only: the existing field
        # analysis, crowded-source selection, differential-candidate
        # selection, catalogue matching, and rendering logic are unchanged.
        _report_non_crowded_target_candidates(
            result=field_result,
            config=config,
        )

        # Resolve the already-selected differential candidates through the
        # same catalogue cross-match pathway used for crowded sources.
        # This is deliberately a post-selection identification step: it does
        # not alter the existing differential candidate selection/ranking.
        if run_catalogue_crossmatch and differential_coordinates:
            differential_catalogue_results = _run_catalogue_crossmatch(
                config=config,
                source_coordinates=differential_coordinates,
            )
            _log_differential_catalogue_resolution(
                config=config,
                source_coordinates=differential_coordinates,
                catalogue_results=differential_catalogue_results,
            )

        # ``candidate_coordinates`` is the bounded crowded-source candidate
        # pool. Catalogue matching, when enabled, is a final-stage ranking
        # criterion rather than a source-discovery criterion.
        source_coordinates = candidate_coordinates

        if run_catalogue_crossmatch:
            catalogue_results = _run_catalogue_crossmatch(
                config=config,
                source_coordinates=candidate_coordinates,
            )
            source_coordinates = _select_catalogue_identified_sources(
                source_coordinates=candidate_coordinates,
                catalogue_results=catalogue_results,
                neighbour_counts=field_result.neighbour_counts,
                peak_values=_field_peak_values(result=field_result),
            )
        else:
            source_coordinates = _select_default_field_visualisation_sources(
                source_coordinates=candidate_coordinates,
                neighbour_counts=field_result.neighbour_counts,
            )

        _render_field_analysis_observation(
            visualizer=visualizer,
            result=field_result,
            source_coordinates=source_coordinates,
            differential_coordinates=differential_coordinates,
            config=config,
        )

        logger.info(
            "Observation demonstration completed successfully for target '%s'.",
            config.target.target_name,
        )
        return

    # ------------------------------------------------------------------
    # All remaining branches are target-centred and therefore return the
    # normal WorkflowResult type.
    # ------------------------------------------------------------------

    if _is_time_series_experiment(config):
        results = _run_time_series_workflow(
            workflow=workflow,
            config=config,
        )

        if not results:
            raise RuntimeError(
                "Time-series demonstration produced no WorkflowResult objects."
            )

        result: WorkflowResult = results[0]

        logger.info(
            "Using first completed cadence as representative observation "
            "for time-series visualisation.",
        )

    else:
        result = _run_target_workflow(workflow=workflow)
        results: tuple[WorkflowResult, ...] = (result,)

    observation = result.observation

    logger.info(
        "Production workflow completed: %d representative result(s).",
        len(results),
    )

    visualizer = ObservationVisualizer(logger)

    # ------------------------------------------------------------------
    # Target-centred diagnostic inspection.
    # ------------------------------------------------------------------

    _inspect_result(
        result=result,
        config=config,
    )

    source_coordinates = _report_crowded_sources(
        result=result,
    )

    if run_catalogue_crossmatch:
        _run_catalogue_crossmatch(
            config=config,
            source_coordinates=source_coordinates,
        )

    _render_representative_observation(
        visualizer=visualizer,
        result=result,
        source_coordinates=source_coordinates,
        show_scarlet_field_selection=show_scarlet_field_selection,
        scarlet_field_max_sources=scarlet_field_max_sources,
        config=config,
    )

    if _is_time_series_experiment(config):
        _render_time_series_products(
            visualizer=visualizer,
            results=results,
            config=config,
        )

    logger.info(
        "Observation demonstration completed successfully for target '%s'.",
        config.target.target_name,
    )


# ============================================================================
# Workflow execution
# ============================================================================


def _run_target_workflow(
    *,
    workflow: Any,
) -> WorkflowResult:
    """Execute a target-centred workflow and narrow its result type."""
    workflow_result = workflow.run()

    if not isinstance(workflow_result, WorkflowResult):
        raise RuntimeError(
            "Target-centred workflow returned an unexpected result type: "
            f"{type(workflow_result).__name__}."
        )

    return workflow_result


def _run_time_series_workflow(
    *,
    workflow: Any,
    config: ExperimentConfig,
) -> tuple[WorkflowResult, ...]:
    """
    Execute the production time-series workflow.

    ``src/demo.py`` may provide a demo-only ``cadence_indices`` sequence.
    When it is absent, the workflow receives None and therefore follows the
    configured production/default cadence selection.
    """
    time_series_config = getattr(
        config,
        "time_series",
        None,
    )

    cadence_indices = (
        getattr(
            time_series_config,
            "cadence_indices",
            None,
        )
        if time_series_config is not None
        else None
    )

    logger.info(
        "Executing TESS time-series demonstration.",
    )

    if cadence_indices is None:
        logger.info(
            "Cadence selection: workflow/default cadence sequence.",
        )
    else:
        logger.info(
            "Cadence selection: %d requested cadence index/indices.",
            len(cadence_indices),
        )

    results = workflow.run_time_series(
        cadence_indices=cadence_indices,
    )

    result_sequence = tuple(results)

    logger.info(
        "TESS time-series workflow returned %d completed cadence(s).",
        len(result_sequence),
    )

    return result_sequence


def _is_field_analysis_experiment(
    config: ExperimentConfig,
) -> bool:
    """Return whether the configured experiment is field-only analysis."""
    return getattr(config, "experiment_type", None) is ExperimentType.FIELD_ANALYSIS


def _is_time_series_experiment(
    config: ExperimentConfig,
) -> bool:
    """Return whether the configured experiment enables time-series mode."""
    time_series_config = getattr(
        config,
        "time_series",
        None,
    )

    return bool(
        time_series_config is not None
        and getattr(
            time_series_config,
            "enabled",
            False,
        )
    )


# ============================================================================
# Configuration reporting
# ============================================================================


def _log_experiment_configuration(
    config: ExperimentConfig,
) -> None:
    """Log the experiment dimensions that control demonstration behaviour."""
    experiment_type = _enum_value(
        getattr(config, "experiment_type", None),
    )
    survey = _enum_value(
        getattr(config, "survey", None),
    )
    pipeline = _enum_value(
        getattr(config, "pipeline", None),
    )

    logger.info(
        "Experiment configuration:",
    )
    logger.info(
        "  Experiment type : %s",
        experiment_type or "N/A",
    )
    logger.info(
        "  Survey          : %s",
        survey or "N/A",
    )
    logger.info(
        "  Pipeline        : %s",
        pipeline or "N/A",
    )
    logger.info(
        "  Time-series     : %s",
        "enabled" if _is_time_series_experiment(config) else "disabled",
    )


def _enum_value(
    value: Any,
) -> str | None:
    """Return an enum value as text without assuming a specific enum type."""
    if value is None:
        return None

    enum_value = getattr(
        value,
        "value",
        None,
    )

    if enum_value is not None:
        return str(enum_value)

    return str(value)


# ============================================================================
# Result inspection
# ============================================================================


def _inspect_result(
    *,
    result: WorkflowResult,
    config: ExperimentConfig,
) -> None:
    """
    Run the diagnostic ObservationInspector against a completed result.
    """
    logger.info(
        "Inspecting completed workflow result.",
    )

    ObservationInspector(
        logger,
    ).inspect(
        result.observation,
        evaluation=result.evaluation,
        target_match=result.target_match,
    )

    if result.reference_matches:
        logger.info(
            "Resolved differential reference sources: %d.",
            len(result.reference_matches),
        )

        for reference_match in result.reference_matches:
            logger.info(
                (
                    "Reference %s -> source %s "
                    "(separation=%.3f px)."
                ),
                reference_match.reference_identifier,
                reference_match.source_identifier,
                reference_match.separation_pixels,
            )

    if result.differential_photometry is not None:
        differential = result.differential_photometry

        logger.info(
            (
                "Differential photometry available: target=%s, "
                "references=%d, differential_flux=%.8g, "
                "differential_magnitude=%.8f."
            ),
            differential.target_source_identifier,
            differential.valid_reference_count,
            differential.differential_flux,
            differential.differential_magnitude,
        )
    else:
        logger.info(
            "Differential photometry: not available for this result.",
        )


# ============================================================================
# FIELD_ANALYSIS diagnostics
# ============================================================================


def _report_field_analysis_sources(
    *,
    result: FieldAnalysisResult,
) -> tuple:
    """
    Report the complete FIELD_ANALYSIS source population and return a
    bounded crowded-source candidate pool for downstream diagnostics.

    The complete detection catalogue and neighbour counts remain
    authoritative and are never truncated. Only crowded sources are passed
    to the diagnostic/catalogue layer. The candidate pool is ranked using
    two image-derived properties that are available before catalogue
    matching:

    * crowding: number of detected neighbouring sources; and
    * brightness: finite ``peak_value`` when available.

    Up to 50 crowded sources are retained. Catalogue association is evaluated
    only after this bounded pool has been cross-matched, so external catalogue
    access remains controlled while the final target/display ranking can use
    catalogue evidence.
    """
    observation = result.observation
    detections = observation.detections

    if detections is None:
        logger.warning(
            "FIELD_ANALYSIS result contains no detection result.",
        )
        return tuple()

    source_by_identifier = {
        source.identifier: source
        for source in detections.sources
    }

    crowded_identifiers = tuple(
        identifier
        for identifier, count in result.neighbour_counts.items()
        if count > 0 and identifier in source_by_identifier
    )

    non_crowded_count = sum(
        1
        for identifier, count in result.neighbour_counts.items()
        if count == 0 and identifier in source_by_identifier
    )

    logger.info(
        "FIELD_ANALYSIS source summary: %d detected source(s).",
        len(detections.sources),
    )
    logger.info(
        (
            "FIELD_ANALYSIS crowding summary: %d source(s) have neighbours; "
            "%d source(s) have no neighbours."
        ),
        len(crowded_identifiers),
        non_crowded_count,
    )

    crowded_limit = 50

    peak_values = {
        identifier: _as_finite_float(
            getattr(source_by_identifier[identifier], "peak_value", None),
        )
        for identifier in crowded_identifiers
    }

    def _brightness_sort_key(identifier: str) -> tuple[float, str]:
        peak = peak_values.get(identifier)
        return (
            -(peak if peak is not None else 0.0),
            identifier,
        )

    brightness_order = tuple(
        sorted(
            (
                identifier
                for identifier in crowded_identifiers
                if peak_values.get(identifier) is not None
            ),
            key=_brightness_sort_key,
        )
    )

    crowding_order = tuple(
        sorted(
            crowded_identifiers,
            key=lambda identifier: (
                -result.neighbour_counts.get(identifier, 0),
                identifier,
            ),
        )
    )

    crowding_rank = {
        identifier: _normalised_rank(
            position=position,
            count=len(crowding_order),
        )
        for position, identifier in enumerate(crowding_order)
    }

    brightness_rank = {
        identifier: _normalised_rank(
            position=position,
            count=len(brightness_order),
        )
        for position, identifier in enumerate(brightness_order)
    }

    def candidate_score(identifier: str) -> float:
        """Rank a crowded source before catalogue matching."""
        return (
            0.65 * crowding_rank.get(identifier, 0.0)
            + 0.35 * brightness_rank.get(identifier, 0.0)
        )

    ranked_crowded = tuple(
        sorted(
            crowded_identifiers,
            key=lambda identifier: (
                -candidate_score(identifier),
                -result.neighbour_counts.get(identifier, 0),
                -(peak_values[identifier] or 0.0),
                identifier,
            ),
        )
    )

    selected_identifiers = ranked_crowded[:crowded_limit]

    logger.info(
        (
            "FIELD_ANALYSIS crowded candidate pool: %d source(s) "
            "selected from %d crowded source(s); ranking = 65%% crowding + "
            "35%% peak brightness."
        ),
        len(selected_identifiers),
        len(crowded_identifiers),
    )

    coordinate_inputs = []
    for identifier in selected_identifiers:
        source = source_by_identifier[identifier]
        coordinate_inputs.append(
            (
                source.identifier,
                source.x_centroid,
                source.y_centroid,
            )
        )

    source_coordinates = tuple(
        CrowdedSourceCoordinates().convert(
            observation=observation,
            sources=tuple(coordinate_inputs),
        )
    )

    logger.info(
        (
            "FIELD_ANALYSIS crowded candidate sky-coordinate conversion "
            "completed: %d source(s)."
        ),
        len(source_coordinates),
    )

    for coordinate in source_coordinates:
        identifier = coordinate.source_identifier
        logger.info(
            (
                "Shortlisted source %s: X=%.3f, Y=%.3f, "
                "RA=%.8f deg, Dec=%.8f deg, neighbours=%d, peak=%s, "
                "pre-catalogue-score=%.4f."
            ),
            identifier,
            coordinate.x_pixel,
            coordinate.y_pixel,
            coordinate.right_ascension,
            coordinate.declination,
            result.neighbour_counts.get(identifier, 0),
            _format_optional_float(
                peak_values.get(identifier),
                precision=3,
            ),
            candidate_score(identifier),
        )

    return source_coordinates


def _report_non_crowded_target_candidates(
    *,
    result: FieldAnalysisResult,
    config: ExperimentConfig,
    max_candidates: int = 5,
) -> tuple:
    """
    Log a small shortlist of potentially non-crowded science targets.

    This is a FIELD_ANALYSIS reconnaissance diagnostic only. A candidate must
    have zero detected neighbours and must be sufficiently far from the image
    edge to support the existing aperture/background geometry. Candidates are
    ranked primarily by detected peak brightness so that the shortlist favours
    sources likely to provide measurable target photometry. Target separation
    is reported when WCS information is available, but it is not used to
    modify the existing production analysis.

    The returned coordinates are not consumed by the production workflow.
    They are provided only so the caller can inspect the shortlisted source
    positions if needed.
    """
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")

    observation = result.observation
    detections = observation.detections

    if detections is None or not detections.sources:
        logger.info(
            "Non-crowded target candidate search skipped: no detected sources."
        )
        return tuple()

    source_by_identifier = {
        source.identifier: source
        for source in detections.sources
    }

    isolated_identifiers = tuple(
        identifier
        for identifier, count in result.neighbour_counts.items()
        if count == 0 and identifier in source_by_identifier
    )

    logger.info(
        "FIELD_ANALYSIS non-crowded target search: %d isolated source(s) available.",
        len(isolated_identifiers),
    )

    if not isolated_identifiers:
        logger.warning(
            "No isolated sources are available for non-crowded target selection."
        )
        return tuple()

    image_shape = _observation_image_shape(observation)
    minimum_edge_margin = 5.0

    target_pixel = _field_target_pixel_position(
        observation=observation,
        config=config,
    )

    eligible_identifiers: list[str] = []
    edge_distances: dict[str, float | None] = {}
    target_separations: dict[str, float | None] = {}
    peak_values: dict[str, float | None] = {}

    for identifier in isolated_identifiers:
        source = source_by_identifier[identifier]
        source_x = _as_finite_float(getattr(source, "x_centroid", None))
        source_y = _as_finite_float(getattr(source, "y_centroid", None))
        peak_values[identifier] = _as_finite_float(
            getattr(source, "peak_value", None),
        )

        if source_x is None or source_y is None:
            edge_distances[identifier] = None
            target_separations[identifier] = None
            continue

        if image_shape is not None:
            height, width = image_shape
            edge_distance = min(
                source_x,
                source_y,
                (width - 1) - source_x,
                (height - 1) - source_y,
            )
            edge_distances[identifier] = edge_distance

            if edge_distance < minimum_edge_margin:
                continue
        else:
            edge_distances[identifier] = None

        if target_pixel is not None:
            dx = source_x - target_pixel[0]
            dy = source_y - target_pixel[1]
            target_separations[identifier] = (dx * dx + dy * dy) ** 0.5
        else:
            target_separations[identifier] = None

        eligible_identifiers.append(identifier)

    if not eligible_identifiers:
        logger.warning(
            "No isolated sources remain after non-crowded target edge-safety filtering."
        )
        return tuple()

    ranked_identifiers = tuple(
        sorted(
            eligible_identifiers,
            key=lambda identifier: (
                -(peak_values.get(identifier) or 0.0),
                -(target_separations.get(identifier) or 0.0),
                identifier,
            ),
        )
    )

    selected_identifiers = ranked_identifiers[:max_candidates]

    coordinate_inputs = []
    for identifier in selected_identifiers:
        source = source_by_identifier[identifier]
        coordinate_inputs.append(
            (
                source.identifier,
                source.x_centroid,
                source.y_centroid,
            )
        )

    coordinates = tuple(
        CrowdedSourceCoordinates().convert(
            observation=observation,
            sources=tuple(coordinate_inputs),
        )
    )

    logger.info(
        (
            "FIELD_ANALYSIS non-crowded target shortlist: %d source(s) selected "
            "from %d isolated source(s); ranking = peak brightness, with "
            "%.1f px minimum edge margin."
        ),
        len(coordinates),
        len(isolated_identifiers),
        minimum_edge_margin,
    )

    for rank, coordinate in enumerate(coordinates, start=1):
        identifier = coordinate.source_identifier
        logger.info(
            (
                "Non-crowded target candidate %d: source=%s, X=%.3f, Y=%.3f, "
                "RA=%.8f deg, Dec=%.8f deg, neighbours=%d, peak=%s, "
                "edge_distance=%s px, target_separation=%s px."
            ),
            rank,
            identifier,
            coordinate.x_pixel,
            coordinate.y_pixel,
            coordinate.right_ascension,
            coordinate.declination,
            result.neighbour_counts.get(identifier, 0),
            _format_optional_float(
                peak_values.get(identifier),
                precision=3,
            ),
            _format_optional_float(
                edge_distances.get(identifier),
                precision=2,
            ),
            _format_optional_float(
                target_separations.get(identifier),
                precision=2,
            ),
        )

    return coordinates


def _field_peak_values(
    *,
    result: FieldAnalysisResult,
) -> dict[str, float | None]:
    """Return finite peak values for the detected field sources."""
    detections = result.observation.detections
    if detections is None:
        return {}

    return {
        source.identifier: _as_finite_float(
            getattr(source, "peak_value", None),
        )
        for source in detections.sources
    }


def _render_field_analysis_observation(
    *,
    visualizer: ObservationVisualizer,
    result: FieldAnalysisResult,
    source_coordinates: tuple,
    differential_coordinates: tuple,
    config: ExperimentConfig,
) -> None:
    """Render the compact field-analysis diagnostic sequence."""
    del config

    observation = result.observation
    detection_image = _select_detection_image(observation)

    # Detector-wide context once.
    visualizer.render(
        observation,
        image_name="raw",
        title=_image_title(observation, "Raw Field Image"),
    )

    # Background-removed image and detections are one diagnostic, not two.
    if detection_image != "raw":
        visualizer.render(
            observation,
            image_name=detection_image,
            show_detections=True,
            title=_image_title(
                observation,
                "Background Removed • Detected Sources",
            ),
        )

    if source_coordinates or differential_coordinates:
        _render_field_source_overlay(
            visualizer=visualizer,
            observation=observation,
            source_coordinates=source_coordinates,
            differential_coordinates=differential_coordinates,
            image_name=detection_image,
            title=_image_title(observation, "Crowded Sources / Field Analysis"),
        )

    logger.info(
        "FIELD_ANALYSIS visualisation completed: %d detected source(s), %d source(s) with neighbours.",
        len(observation.detections.sources) if observation.detections is not None else 0,
        sum(count > 0 for count in result.neighbour_counts.values()),
    )


# ============================================================================
# FIELD_ANALYSIS differential-photometry candidate diagnostics
# ============================================================================


def _report_differential_photometry_candidates(
    *,
    result: FieldAnalysisResult,
    config: ExperimentConfig,
    max_candidates: int = 3,
) -> tuple:
    """
    Identify a small set of isolated differential-photometry candidates.

    This is a FIELD_ANALYSIS diagnostic, not a production reference-star
    selector. It deliberately favours sources that are:

    * isolated according to the authoritative neighbour-count analysis;
    * reasonably bright in the detected image; and
    * spatially separated from the configured target position when the
      observation WCS permits that conversion.

    The result is only a shortlist for subsequent human inspection and
    time-series validation. It is not written into the differential-
    photometry experiment configuration.
    """
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")

    observation = result.observation
    detections = observation.detections

    if detections is None or not detections.sources:
        logger.info(
            "Differential-photometry candidate search skipped: no detected sources."
        )
        return tuple()

    source_by_identifier = {
        source.identifier: source
        for source in detections.sources
    }

    isolated_identifiers = tuple(
        identifier
        for identifier, count in result.neighbour_counts.items()
        if count == 0 and identifier in source_by_identifier
    )

    logger.info(
        "FIELD_ANALYSIS differential-photometry search: %d isolated source(s) available.",
        len(isolated_identifiers),
    )

    if not isolated_identifiers:
        logger.warning(
            "No isolated sources are available for differential-photometry candidate selection."
        )
        return tuple()

    # Differential photometry requires the complete aperture + background
    # geometry to fit inside the image. A source can therefore be formally
    # isolated yet still be a poor reference if it sits at the detector edge.
    # For the TESS differential-photometry study, the production geometry uses
    # a 4-pixel outer background annulus. Require a small additional margin so
    # the reference is not merely inside the annulus boundary but comfortably
    # away from the edge.
    image_shape = _observation_image_shape(observation)
    minimum_edge_margin = 3.0

    if image_shape is None:
        logger.warning(
            "Could not determine FIELD_ANALYSIS image dimensions; "
            "edge-safety filtering will not be applied."
        )

    target_pixel = _field_target_pixel_position(
        observation=observation,
        config=config,
    )

    edge_distance_values: dict[str, float | None] = {}
    separation_values: dict[str, float | None] = {}
    eligible_identifiers: list[str] = []

    for identifier in isolated_identifiers:
        source = source_by_identifier[identifier]
        source_x = _as_finite_float(getattr(source, "x_centroid", None))
        source_y = _as_finite_float(getattr(source, "y_centroid", None))

        if source_x is None or source_y is None:
            edge_distance_values[identifier] = None
            separation_values[identifier] = None
            continue

        if image_shape is not None:
            height, width = image_shape
            edge_distance = min(
                source_x,
                source_y,
                (width - 1) - source_x,
                (height - 1) - source_y,
            )
            edge_distance_values[identifier] = edge_distance

            if edge_distance < minimum_edge_margin:
                logger.info(
                    "Differential candidate %s excluded: %.2f px from "
                    "nearest image edge (minimum %.2f px).",
                    identifier,
                    edge_distance,
                    minimum_edge_margin,
                )
                continue
        else:
            edge_distance_values[identifier] = None

        # Exclude the configured science target itself if it appears in the
        # isolated population. This is a reference-star shortlist, not a
        # second way of selecting the target.
        if target_pixel is not None:
            dx = source_x - target_pixel[0]
            dy = source_y - target_pixel[1]
            separation = (dx * dx + dy * dy) ** 0.5
            separation_values[identifier] = separation

            if separation < minimum_edge_margin:
                logger.info(
                    "Differential candidate %s excluded: %.2f px from "
                    "configured target position (minimum %.2f px).",
                    identifier,
                    separation,
                    minimum_edge_margin,
                )
                continue
        else:
            separation_values[identifier] = None

        eligible_identifiers.append(identifier)

    #eligible_identifiers = tuple(eligible_identifiers)

    logger.info(
        (
            "FIELD_ANALYSIS differential-photometry eligibility: %d of %d "
            "isolated source(s) remain after edge/target separation filters "
            "(minimum edge margin %.1f px)."
        ),
        len(eligible_identifiers),
        len(isolated_identifiers),
        minimum_edge_margin,
    )

    if not eligible_identifiers:
        logger.warning(
            "No isolated sources remain after differential-photometry "
            "edge/target safety filtering."
        )
        return tuple()

    peak_values = {
        identifier: _as_finite_float(
            getattr(source_by_identifier[identifier], "peak_value", None),
        )
        for identifier in eligible_identifiers
    }

    brightness_order = tuple(
        sorted(
            eligible_identifiers,
            key=lambda identifier: (
                -(peak_values.get(identifier) or 0.0),
                identifier,
            ),
        )
    )

    brightness_rank = {
        identifier: _normalised_rank(
            position=position,
            count=len(brightness_order),
        )
        for position, identifier in enumerate(brightness_order)
    }

    finite_separations = tuple(
        identifier
        for identifier in eligible_identifiers
        if separation_values.get(identifier) is not None
    )

    separation_order = tuple(
        sorted(
            finite_separations,
            key=lambda identifier: (
                -(separation_values[identifier] or 0.0),
                identifier,
            ),
        )
    )

    separation_rank = {
        identifier: _normalised_rank(
            position=position,
            count=len(separation_order),
        )
        for position, identifier in enumerate(separation_order)
    }

    # All candidates already satisfy the strongest criterion: zero detected
    # neighbours. Within that isolated population, brightness is useful for
    # obtaining a measurable reference, while target separation helps avoid
    # selecting a source whose PSF/aperture is likely to interact with the
    # target. This is a display/reconnaissance heuristic only.
    def candidate_score(identifier: str) -> float:
        return (
            0.65 * brightness_rank.get(identifier, 0.0)
            + 0.35 * separation_rank.get(identifier, 0.0)
        )

    ranked_identifiers = tuple(
        sorted(
            eligible_identifiers,
            key=lambda identifier: (
                -candidate_score(identifier),
                -(peak_values.get(identifier) or 0.0),
                -(separation_values.get(identifier) or 0.0),
                identifier,
            ),
        )
    )

    selected_identifiers = ranked_identifiers[:max_candidates]

    coordinate_inputs = []
    for identifier in selected_identifiers:
        source = source_by_identifier[identifier]
        coordinate_inputs.append(
            (
                source.identifier,
                source.x_centroid,
                source.y_centroid,
            )
        )

    coordinates = tuple(
        CrowdedSourceCoordinates().convert(
            observation=observation,
            sources=tuple(coordinate_inputs),
        )
    )

    logger.info(
        (
            "FIELD_ANALYSIS differential-photometry shortlist: %d source(s) "
            "selected from %d edge-safe isolated source(s); ranking = 65%% "
            "brightness + 35%% target separation."
        ),
        len(coordinates),
        len(eligible_identifiers),
    )

    for rank, coordinate in enumerate(coordinates, start=1):
        identifier = coordinate.source_identifier
        logger.info(
            (
                "Differential candidate %d: source %s, X=%.3f, Y=%.3f, "
                "RA=%.8f deg, Dec=%.8f deg, neighbours=0, peak=%s, "
                "target-separation=%s px, score=%.4f."
            ),
            rank,
            identifier,
            coordinate.x_pixel,
            coordinate.y_pixel,
            coordinate.right_ascension,
            coordinate.declination,
            _format_optional_float(peak_values.get(identifier), precision=3),
            _format_optional_float(separation_values.get(identifier), precision=2),
            candidate_score(identifier),
        )

    if target_pixel is None:
        logger.info(
            "Target-pixel position could not be derived from the observation WCS; "
            "candidate ranking therefore uses brightness among isolated sources only."
        )

    return coordinates


def _observation_image_shape(observation: Any) -> tuple[int, int] | None:
    """Return the raw observation image shape as ``(height, width)``."""
    images = getattr(observation, "images", None)
    if not isinstance(images, dict):
        return None

    raw_image = images.get("raw")
    raw_nddata = getattr(raw_image, "image", None)
    data = getattr(raw_nddata, "data", None)
    shape = getattr(data, "shape", None)

    if not shape or len(shape) < 2:
        return None

    try:
        return int(shape[-2]), int(shape[-1])
    except (TypeError, ValueError):
        return None



def _field_target_pixel_position(
    *,
    observation: Any,
    config: ExperimentConfig,
) -> tuple[float, float] | None:
    """Convert the configured FIELD_ANALYSIS target coordinate to pixels when possible."""
    target = getattr(config, "target", None)
    ra = _as_finite_float(getattr(target, "right_ascension", None))
    dec = _as_finite_float(getattr(target, "declination", None))

    if ra is None or dec is None:
        return None

    wcs_candidates = [
        getattr(observation, "wcs", None),
    ]

    raw_image = getattr(observation, "images", {}).get("raw")
    raw_nddata = getattr(raw_image, "image", None)
    wcs_candidates.append(getattr(raw_nddata, "wcs", None))

    for wcs in wcs_candidates:
        if wcs is None:
            continue

        converter = getattr(wcs, "world_to_pixel_values", None)
        if not callable(converter):
            continue

        try:
            pixel = cast(Sequence[float], converter(ra, dec))
            if len(pixel) < 2:
                continue

            x = _as_finite_float(pixel[0])
            y = _as_finite_float(pixel[1])

            if x is not None and y is not None:
                return x, y
        except Exception:
            continue

    return None


def _render_field_source_overlay(
    *,
    visualizer: ObservationVisualizer,
    observation: Any,
    source_coordinates: tuple,
    differential_coordinates: tuple,
    image_name: str,
    title: str,
) -> None:
    """Render crowded and isolated differential candidates in one figure.

    The current visualizer API does not yet expose the differential-candidate
    styling argument. Signature inspection keeps this demo backward-compatible
    while allowing the next visualizer revision to render the two classes with
    distinct markers/colours in the same figure.
    """
    # Treat the renderer as dynamically extensible: the visualizer is being
    # evolved to accept differential-candidate styling, while this demo must
    # remain compatible with older visualizer versions. Using ``Any`` here is
    # intentional because the supported call shape is determined below by
    # signature inspection.
    render_method: Any = visualizer.render_crowded_sources
    parameters = inspect.signature(render_method).parameters

    if "differential_coordinates" in parameters:
        render_method(
            observation,
            coordinates=source_coordinates,
            differential_coordinates=differential_coordinates,
            image_name=image_name,
            annotate_coordinates=False,
            title=title,
        )
        return

    # Compatibility path for the currently attached visualizer. It preserves
    # the existing crowded-source figure until the visualizer is updated.
    logger.info(
        (
            "Current ObservationVisualizer does not yet expose "
            "'differential_coordinates'; rendering the existing crowded-source "
            "overlay unchanged. The next visualizer revision will add the "
            "isolated-source markers to this same figure."
        )
    )

    render_method(
        observation,
        coordinates=source_coordinates,
        image_name=image_name,
        annotate_coordinates=False,
        title=title,
    )

# ============================================================================
# Crowded-source diagnostics
# ============================================================================
def _create_local_field_observation(
    *,
    observation: Any,
    local_field: Any,
) -> Any:
    """
    Create a shallow diagnostic view of the Observation whose detection
    catalogue contains only the authoritative LocalField sources.

    The production Observation is never modified.

    The complete detection catalogue remains available on the original
    WorkflowResult for full-field visualisation and provenance. This
    LocalField-scoped view exists only for diagnostics that require the
    detection catalogue and evaluation results to describe the same source
    population.
    """
    if local_field is None:
        raise ValueError(
            "Cannot create a LocalField-scoped observation without a LocalField."
        )

    detection_result = observation.detections

    if detection_result is None:
        raise ValueError(
            "Cannot create a LocalField-scoped observation without "
            "a completed detection result."
        )

    selected_sources = local_field.sources

    selected_identifiers = {
        source.identifier
        for source in selected_sources
    }

    detection_sources = tuple(
        source
        for source in detection_result.sources
        if source.identifier in selected_identifiers
    )

    if len(detection_sources) != len(selected_sources):
        missing = sorted(
            selected_identifiers
            - {
                source.identifier
                for source in detection_sources
            }
        )

        raise ValueError(
            "LocalField contains source identifiers that are absent from "
            f"the detection catalogue: {missing}."
        )

    scoped_observation = copy.copy(observation)

    scoped_observation.detections = DetectionResult(
        sources=detection_sources,
        algorithm=detection_result.algorithm,
        detection_threshold=detection_result.detection_threshold,
        detection_timestamp=detection_result.detection_timestamp,
    )

    return scoped_observation

def _report_crowded_sources(
    *,
    result: WorkflowResult,
) -> tuple:
    """
    Extract crowded-source classifications from the authoritative LocalField.

    The production Observation retains the complete detection catalogue.
    Crowded-source diagnostics instead receive a shallow LocalField-scoped
    Observation so that detection records and photometric evaluation records
    describe exactly the same analytical source population.
    """
    observation = result.observation
    local_field = getattr(result, "local_field", None)

    if local_field is None:
        logger.info(
            "Crowded-source diagnostic skipped: workflow returned no LocalField."
        )
        return tuple()

    scoped_observation = _create_local_field_observation(
        observation=observation,
        local_field=local_field,
    )

    logger.info(
        (
            "Running crowded-source diagnostic against authoritative "
            "LocalField: %d source(s) from %d total detection(s)."
        ),
        local_field.selected_source_count,
        local_field.available_source_count,
    )

    crowded_sources = CrowdedSourceExtractor(
        logger,
    ).report(
        observation=scoped_observation,
        evaluation=result.evaluation,
    )

    if not crowded_sources:
        logger.info(
            "No crowded sources identified within the LocalField."
        )
        return tuple()

    logger.info(
        "Crowded sources identified within LocalField: %d.",
        len(crowded_sources),
    )

    detections = observation.detections

    if detections is None:
        logger.warning(
            "No source detections are available for crowded-source reporting."
        )
        return ()

    detected_sources_by_identifier = {
        source.identifier: source
        for source in detections.sources
    }

    coordinate_inputs = []

    for crowded_source in crowded_sources:
        detected_source = detected_sources_by_identifier.get(
            crowded_source.source_identifier
        )

        if detected_source is None:
            logger.warning(
                "Unable to resolve detected source '%s' for crowded-source "
                "coordinate reporting.",
                crowded_source.source_identifier,
            )
            continue

        coordinate_inputs.append(
            (
                detected_source.identifier,
                detected_source.x_centroid,
                detected_source.y_centroid,
            )
        )

    source_coordinates = tuple(
        CrowdedSourceCoordinates().convert(
            observation=observation,
            sources=tuple(coordinate_inputs),
        )
    )

    logger.info(
        "Crowded-source sky-coordinate conversion completed: %d source(s).",
        len(source_coordinates),
    )

    for coordinate in source_coordinates:
        logger.info(
            (
                "Crowded source %s: "
                "X=%.3f, Y=%.3f, "
                "RA=%.8f deg, Dec=%.8f deg."
            ),
            coordinate.source_identifier,
            coordinate.x_pixel,
            coordinate.y_pixel,
            coordinate.right_ascension,
            coordinate.declination,
        )

    return source_coordinates
# ============================================================================
# Representative observation rendering
# ============================================================================


def _render_representative_observation(
    *,
    visualizer: ObservationVisualizer,
    result: WorkflowResult,
    source_coordinates: tuple,
    config: ExperimentConfig,
    show_scarlet_field_selection: bool = True,
    scarlet_field_max_sources: int | None = 50,
) -> None:
    """
    Render the completed representative observation using a compact,
    non-redundant presentation sequence.

    BASELINE deliberately shows the observational/photometric evidence once:

        1. Raw image
        2. Estimated background
        3. Background-removed image + detections
        4. Crowded-source context within the authoritative LocalField
        5. Target-centred LocalField
        6. Clean target-context view for presentation use
        7. Combined photometric assessment (quality + geometry + association)
        8. Target detail

    SCARLET deliberately does *not* repeat those baseline views. It renders
    only stored SCARLET products. The authoritative LocalField is activated
    as presentation context without being rendered, so SCARLET products can
    still use the same target-centred crop.
    """
    observation = result.observation
    detection_image = _select_detection_image(observation)
    local_field = getattr(result, "local_field", None)

    # ------------------------------------------------------------------
    # SCARLET: presentation of SCARLET products only.
    # ------------------------------------------------------------------
    if config.pipeline is ProcessingPipeline.SCARLET:
        if local_field is not None:
            visualizer.activate_local_field(
                observation,
                local_field=local_field,
            )

        _render_scarlet_deblend_products(
            visualizer=visualizer,
            observation=observation,
            source_preview_count=9,
        )
        return

    # ------------------------------------------------------------------
    # BASELINE: observational and photometric diagnostics.
    # ------------------------------------------------------------------

    # 1. Raw image — detector-wide observational context.
    visualizer.render(
        observation,
        image_name="raw",
        title=_image_title(observation, "Raw Image"),
    )

    # 2. Estimated background — this is a key preprocessing diagnostic and
    # should be visible explicitly rather than inferred from the subtraction.
    if "background" in observation.images:
        visualizer.render(
            observation,
            image_name="background",
            title=_image_title(
                observation,
                "Estimated Background",
            ),
        )

    # 3. Background-removed image with the complete detection catalogue.
    # This combines two previously separate views into one useful diagnostic.
    if detection_image != "raw":
        visualizer.render(
            observation,
            image_name=detection_image,
            show_detections=True,
            title=_image_title(
                observation,
                "Background Removed • Detected Sources",
            ),
        )

    # 4. Crowded-source context. For target-centric experiments the crowded
    # source coordinates come from the authoritative LocalField, so activate
    # that context before rendering. The image is therefore focused on the
    # analytical neighbourhood rather than repeating the full detector field.
    # FIELD_ANALYSIS does not activate a LocalField and keeps its full-field
    # crowding diagnostic unchanged. Sky coordinates remain in the log.
    if local_field is not None:
        visualizer.activate_local_field(
            observation,
            local_field=local_field,
        )

    if source_coordinates:
        visualizer.render_crowded_sources(
            observation,
            coordinates=source_coordinates,
            image_name=detection_image,
            annotate_coordinates=False,
            title=_image_title(
                observation,
                "Crowded Sources • LocalField" if local_field is not None else "Crowded Sources",
            ),
        )

    # 5. Authoritative target-centred LocalField. This is now a focused view;
    # the detector-wide context has already been provided by steps 1–4.
    if local_field is not None:
        _render_local_field_diagnostics(
            visualizer=visualizer,
            observation=observation,
            local_field=local_field,
            image_name=detection_image,
            pipeline=config.pipeline,
        )

        # 6. Clean target-context visualisation for the presentation.
        # This uses the same authoritative LocalField and production image,
        # but intentionally removes diagnostic clutter such as the LocalField
        # selection boundary, source counts and detector axes. It is an
        # additional visualisation only; no scientific state is modified.
        visualizer.render_target_context(
            observation,
            local_field=local_field,
            image_name=detection_image,
            title=_image_title(
                observation,
                "Target Context • Local Source Environment",
            ),
            view_radius=75.0,
        )

    # 7. One combined photometric assessment rather than separate quality,
    # geometry and target-association figures.
    if observation.detections is not None:
        visualizer.render(
            observation,
            image_name=detection_image,
            show_photometry=observation.photometry is not None,
            show_target=result.target_match is not None,
            show_quality=True,
            target_match=result.target_match,
            evaluation=result.evaluation,
            title=_image_title(
                observation,
                "Photometric Assessment",
            ),
            max_photometry_overlays=_recommended_overlay_limit(observation),
        )

    # 7. Final target-centric detail view. This retains the high-value target
    # separation/aperture/quality evidence without another full-field figure.
    if result.target_match is not None:
        visualizer.render_target_detail(
            observation,
            target_match=result.target_match,
            evaluation=result.evaluation,
            image_name=detection_image,
        )


def _validate_scarlet_demo_options(
    *,
    show_scarlet_field_selection: bool,
    scarlet_field_max_sources: int | None,
) -> None:
    """Validate optional SCARLET visualisation controls."""
    if not isinstance(show_scarlet_field_selection, bool):
        raise TypeError(
            "show_scarlet_field_selection must be a boolean."
        )

    if scarlet_field_max_sources is not None:
        if isinstance(scarlet_field_max_sources, bool):
            raise TypeError(
                "scarlet_field_max_sources must be an integer or None."
            )
        if not isinstance(scarlet_field_max_sources, int):
            raise TypeError(
                "scarlet_field_max_sources must be an integer or None."
            )
        if scarlet_field_max_sources <= 0:
            raise ValueError(
                "scarlet_field_max_sources must be positive or None."
            )
        logger.debug(
            (
                "Legacy scarlet_field_max_sources=%d is retained for "
                "src/demo.py compatibility but does not control LocalField "
                "selection; the production LocalFieldSelector result is "
                "authoritative."
            ),
            scarlet_field_max_sources,
        )


def _render_local_field_diagnostics(
    *,
    visualizer: ObservationVisualizer,
    observation: Any,
    local_field: Any,
    image_name: str,
    pipeline: ProcessingPipeline,
) -> None:
    """
    Render the LocalField selected by the completed production workflow.

    The LocalField is consumed directly from ``WorkflowResult`` rather than
    reconstructed from the complete detection catalogue. This guarantees
    that the diagnostic image represents the exact target-plus-neighbour
    population selected by ``LocalFieldSelector``.
    """
    if local_field is None:
        logger.info(
            "LocalField visualisation skipped: workflow returned no LocalField."
        )
        return

    logger.info(
        (
            "Rendering authoritative LocalField: target=%s, selected=%d, "
            "neighbours=%d, available=%d, max_sources=%d, radius=%.3f px, "
            "selection_method=%s."
        ),
        local_field.target_source.identifier,
        local_field.selected_source_count,
        local_field.selected_neighbour_count,
        local_field.available_source_count,
        local_field.max_sources,
        local_field.selection_radius_pixels,
        local_field.selection_method,
    )

    visualizer.render_local_field(
        observation,
        local_field=local_field,
        image_name=image_name,
        title=_image_title(
            observation,
            (
                "Target-Centred LocalField "
                f"({local_field.selected_source_count}/"
                f"{local_field.available_source_count} sources)"
            ),
        ),
        annotate_sources=False,
    )

    if pipeline is ProcessingPipeline.SCARLET:
        logger.info(
            (
                "The same LocalField supplies %d source centre(s) to "
                "the SCARLET preprocessing stage."
            ),
            len(local_field.source_centers),
        )

def _render_scarlet_deblend_products(
    *,
    visualizer: ObservationVisualizer,
    observation: Any,
    source_preview_count: int = 9,
) -> None:
    """
    Render SCARLET image products already attached to the Observation.

    This function is presentation-only. It never runs SCARLET, modifies the
    Observation, or derives a new photometric measurement.

    The visualisation deliberately uses the dedicated SCARLET methods on
    ``ObservationVisualizer`` rather than treating SCARLET products as
    ordinary baseline images. This is important because individual source
    models are stored as full-frame products and need source-centred cutouts
    for useful visual inspection.
    """
    images = getattr(observation, "images", {})

    if not images:
        logger.info(
            "SCARLET product visualisation skipped: Observation contains no images."
        )
        return

    rendered_product = False

    # --------------------------------------------------------------
    # 1. Observation-space vs intrinsic SCARLET model comparison
    # --------------------------------------------------------------
    model_name = "scarlet_field_model"
    intrinsic_name = "scarlet_field_model_intrinsic"

    if model_name in images and intrinsic_name in images:
        logger.info(
            "Rendering SCARLET observation-space/intrinsic model comparison."
        )
        visualizer.render_scarlet_field_model_comparison(
            observation,
            model_image_name=model_name,
            intrinsic_image_name=intrinsic_name,
            title=_image_title(
                observation,
                "SCARLET FIELD Model Comparison",
            ),
        )
        rendered_product = True
    else:
        if model_name not in images:
            logger.warning(
                "SCARLET field reconstruction product '%s' is unavailable.",
                model_name,
            )
        if intrinsic_name not in images:
            logger.info(
                "SCARLET intrinsic field model product '%s' is unavailable; "
                "continuing without model comparison.",
                intrinsic_name,
            )

    # --------------------------------------------------------------
    # 3. Observation-minus-model residual
    # --------------------------------------------------------------
    residual_name = "scarlet_field_residual"

    if residual_name in images:
        logger.info(
            "Rendering SCARLET field residual product '%s'.",
            residual_name,
        )
        visualizer.render_scarlet_field_residual(
            observation,
            image_name=residual_name,
            title=_image_title(
                observation,
                "SCARLET FIELD Residual (Observed − Model)",
            ),
        )
        rendered_product = True
    else:
        logger.warning(
            "SCARLET field residual product '%s' is unavailable.",
            residual_name,
        )

    # --------------------------------------------------------------
    # 4. Final target-centred deblended LocalField
    # --------------------------------------------------------------
    source_names = sorted(
        name
        for name in images
        if isinstance(name, str)
        and name.startswith("scarlet_source_")
    )

    if source_names:
        logger.info(
            "Rendering final SCARLET deblended LocalField from %d stored "
            "source product(s).",
            len(source_names),
        )
        visualizer.render_scarlet_deblended_local_field(
            observation,
            title=_image_title(
                observation,
                "SCARLET Deblended LocalField",
            ),
            include_target_only=True,
        )
        rendered_product = True
    else:
        logger.info(
            "Final SCARLET deblended LocalField visualisation skipped: "
            "no individual source products are available.",
        )

    # --------------------------------------------------------------
    # 5. Individual source-model montage
    # --------------------------------------------------------------
    if source_names:
        preview_count = min(
            max(1, source_preview_count),
            len(source_names),
        )

        logger.info(
            "Rendering SCARLET individual-source montage: %d of %d stored "
            "source product(s).",
            preview_count,
            len(source_names),
        )

        visualizer.render_scarlet_source_models(
            observation,
            source_names=source_names[:preview_count],
            max_sources=preview_count,
            columns=3,
            crop_radius=25,
            title=_image_title(
                observation,
                (
                    "SCARLET FIELD Individual Source Models "
                    f"(showing {preview_count}/{len(source_names)})"
                ),
            ),
        )
        rendered_product = True

    # --------------------------------------------------------------
    # 6. Final thesis-ready baseline vs SCARLET LocalField comparison
    # --------------------------------------------------------------
    # This is deliberately the final rendering step. Both panels use the
    # same detector coordinates and the same LocalField crop, while the
    # SCARLET panel is reconstructed only from the stored source models.
    # The figure is therefore suitable for direct side-by-side inclusion
    # in the thesis without implying that SCARLET altered the observed
    # pixels in place.
    if source_names:
        baseline_comparison_image = _select_detection_image(observation)
        logger.info(
            "Rendering final thesis-ready baseline versus SCARLET "
            "deblended LocalField comparison using '%s'.",
            baseline_comparison_image,
        )
        visualizer.render_baseline_vs_scarlet_deblended_local_field(
            observation,
            image_name=baseline_comparison_image,
            title=_image_title(
                observation,
                "Baseline vs SCARLET Deblended LocalField",
            ),
        )
        rendered_product = True
    else:
        logger.info(
            "No individual SCARLET source image products are available "
            "for preview.",
        )

    if rendered_product:
        logger.info(
            "SCARLET image-product visualisation completed."
        )
    else:
        logger.warning(
            "SCARLET image-product visualisation produced no renderable "
            "products. Verify that the SCARLET preprocessor stored its "
            "field products on Observation.images."
        )


def _select_detection_image(
    observation: Any,
) -> str:
    """Return the image used by downstream detection/photometry stages."""
    if "background_removed" in observation.images:
        return "background_removed"

    return "raw"


def _image_title(
    observation: Any,
    stage: str,
) -> str:
    """Build a consistent survey/target/stage image title."""
    survey = _enum_value(
        getattr(
            observation.provenance,
            "survey",
            None,
        )
    ) or "Observation"

    target_name = getattr(
        observation.metadata,
        "target_name",
        "Target",
    )

    return (
        f"{survey} • {target_name}\n"
        f"{stage}"
    )


def _recommended_overlay_limit(
    observation: Any,
) -> int | None:
    """
    Keep very crowded image overlays readable.

    The limit affects presentation only. It never changes the underlying
    measurements or evaluation.
    """
    detections = getattr(
        observation,
        "detections",
        None,
    )

    if detections is None:
        return None

    source_count = len(
        getattr(
            detections,
            "sources",
            (),
        )
    )

    if source_count > 200:
        return 100

    if source_count > 100:
        return 75

    return None


# ============================================================================
# Time-series visualisation
# ============================================================================


def _render_time_series_products(
    *,
    visualizer: ObservationVisualizer,
    results: Sequence[WorkflowResult],
    config: ExperimentConfig,
) -> None:
    """
    Render cadence-level target and differential time-series products.

    The completed WorkflowResult sequence is the sole scientific source for
    the plotted values. No resampling, smoothing, detrending or clipping is
    performed here.
    """
    if not results:
        logger.warning(
            "Time-series visualisation skipped: no completed results.",
        )
        return

    logger.info(
        "Preparing time-series visualisation from %d completed cadence(s).",
        len(results),
    )

    target_series = _build_target_series(
        results,
    )

    if target_series is not None:
        time_values, flux_values, valid_values = target_series

        visualizer.render_light_curve(
            time_values,
            flux_values,
            target_name=config.target.target_name,
            time_label=_time_axis_label(results),
            flux_label="Target Net Flux",
            title=(
                f"{config.target.target_name} • Target Light Curve"
            ),
            normalise=False,
            valid=valid_values,
        )

        visualizer.render_light_curve(
            time_values,
            flux_values,
            target_name=config.target.target_name,
            time_label=_time_axis_label(results),
            flux_label="Target Net Flux",
            title=(
                f"{config.target.target_name} • "
                "Normalised Target Light Curve"
            ),
            normalise=True,
            valid=valid_values,
        )
    else:
        logger.info(
            "Target light-curve visualisation skipped: "
            "no consistently resolved target measurement was available.",
        )

    differential_series = _build_differential_series(
        results,
    )

    if differential_series is not None:
        time_values, magnitude_values, valid_values = differential_series

        visualizer.render_differential_light_curve(
            time_values,
            magnitude_values,
            target_name=config.target.target_name,
            time_label=_time_axis_label(results),
            title=(
                f"{config.target.target_name} • "
                "Differential Photometry"
            ),
            valid=valid_values,
            show_reference_level=True,
        )
    else:
        logger.info(
            "Differential light-curve visualisation skipped: "
            "completed results contain no differential-photometry series.",
        )


def _build_target_series(
    results: Sequence[WorkflowResult],
) -> tuple[list[Any], list[Any], list[bool]] | None:
    """
    Build one continuous target net-flux series for demonstration plots.

    This is presentation-only logic. The production workflow remains the
    sole source of scientific measurements.

    TESS can contain multiple nearby detections within a cadence. The
    authoritative TargetSourceMatch identifies the production association,
    but its source identifier is not necessarily stable between cadences.

    For the demonstration light curve, the expected target position supplied
    by TargetSourceMatch is therefore used as the spatial anchor. On each
    cadence, the already-computed photometric measurement nearest to that
    expected position is selected.

    This prevents a nearby companion/source from producing a second apparent
    light curve when source identifiers change between cadences.

    No scientific processing is performed here. In particular, this function
    does not alter:
        * WorkflowResult
        * Observation
        * source detection
        * target resolution
        * aperture photometry
        * evaluation
        * exported production measurements
    """
    time_values: list[Any] = []
    flux_values: list[Any] = []
    valid_values: list[bool] = []

    position_selected_count = 0
    authoritative_fallback_count = 0

    for result in results:
        target_match = result.target_match

        if target_match is None:
            continue

        photometry = result.observation.photometry

        if photometry is None:
            continue

        measurements = tuple(
            photometry.measurements,
        )

        if not measurements:
            continue

        # --------------------------------------------------------------
        # Authoritative production target measurement
        # --------------------------------------------------------------
        authoritative_measurement = _find_measurement(
            measurements,
            target_match.source_identifier,
        )

        measurement = authoritative_measurement

        # --------------------------------------------------------------
        # Demo-only target-position association
        # --------------------------------------------------------------
        #
        # TargetSourceMatch.target_x / target_y represent the expected
        # detector position of the astronomical target. These coordinates
        # are preferable to source_identifier for maintaining a continuous
        # demonstration series when source identifiers change between
        # cadences.
        #
        target_x = _as_finite_float(
            getattr(
                target_match,
                "target_x",
                None,
            ),
        )

        target_y = _as_finite_float(
            getattr(
                target_match,
                "target_y",
                None,
            ),
        )

        if target_x is not None and target_y is not None:
            nearest_measurement = _nearest_measurement(
                measurements,
                x=target_x,
                y=target_y,
                max_distance=None,
            )

            if nearest_measurement is not None:
                measurement = nearest_measurement

                if (
                    authoritative_measurement is not None
                    and nearest_measurement
                    is not authoritative_measurement
                ):
                    position_selected_count += 1

        # --------------------------------------------------------------
        # Authoritative fallback
        # --------------------------------------------------------------
        #
        # Older/incomplete TargetSourceMatch objects may not expose a
        # usable expected target position. In that case retain the original
        # authoritative source-identifier behaviour.
        #
        if measurement is None:
            authoritative_fallback_count += 1
            continue

        # --------------------------------------------------------------
        # Observation time
        # --------------------------------------------------------------
        time_value = getattr(
            result.observation.metadata,
            "observation_time",
            None,
        )

        if time_value is None:
            continue

        # --------------------------------------------------------------
        # Append the already-computed production measurement
        # --------------------------------------------------------------
        time_values.append(
            time_value,
        )

        flux_values.append(
            measurement.net_flux,
        )

        valid_values.append(
            bool(measurement.valid),
        )

    if position_selected_count:
        logger.info(
            (
                "Target light-curve demo selected the photometric "
                "measurement nearest the expected target position on "
                "%d cadence(s), rather than the authoritative source "
                "identifier."
            ),
            position_selected_count,
        )

    if authoritative_fallback_count:
        logger.info(
            (
                "Target light-curve demo could not resolve a photometric "
                "measurement on %d cadence(s)."
            ),
            authoritative_fallback_count,
        )

    if not time_values:
        return None

    return (
        time_values,
        flux_values,
        valid_values,
    )


def _build_differential_series(
    results: Sequence[WorkflowResult],
) -> tuple[list[Any], list[Any], list[bool]] | None:
    """
    Build a cadence-level differential-magnitude series.

    The differential value is copied directly from
    ``WorkflowResult.differential_photometry``. A sample is marked valid
    only when the upstream differential result exists and contains a finite
    value.
    """
    time_values: list[Any] = []
    magnitude_values: list[Any] = []
    valid_values: list[bool] = []

    for result in results:
        differential = result.differential_photometry

        if differential is None:
            continue

        time_value = getattr(
            result.observation.metadata,
            "observation_time",
            None,
        )

        if time_value is None:
            continue

        magnitude = differential.differential_magnitude

        time_values.append(time_value)
        magnitude_values.append(magnitude)
        valid_values.append(
            _is_finite_number(magnitude),
        )

    if not time_values:
        return None

    return (
        time_values,
        magnitude_values,
        valid_values,
    )


def _find_measurement(
    measurements: Sequence[Any],
    source_identifier: str,
) -> Any | None:
    """Return one authoritative photometry measurement by source identifier."""
    for measurement in measurements:
        if (
            getattr(
                measurement,
                "source_identifier",
                None,
            )
            == source_identifier
        ):
            return measurement

    return None

def _nearest_measurement(
    measurements: Sequence[Any],
    *,
    x: float,
    y: float,
    max_distance: float | None,
) -> Any | None:
    """
    Return the photometric measurement whose centroid is nearest to (x, y).

    This helper only selects an already-computed measurement for
    demonstration purposes. It does not create, modify or recalculate
    photometric measurements.
    """
    nearest: Any | None = None
    nearest_distance_squared = float("inf")

    for measurement in measurements:
        measurement_x = _as_finite_float(
            getattr(
                measurement,
                "x_centroid",
                None,
            ),
        )

        measurement_y = _as_finite_float(
            getattr(
                measurement,
                "y_centroid",
                None,
            ),
        )

        if measurement_x is None or measurement_y is None:
            continue

        dx = measurement_x - x
        dy = measurement_y - y

        distance_squared = (
            dx * dx
            + dy * dy
        )

        if distance_squared < nearest_distance_squared:
            nearest_distance_squared = distance_squared
            nearest = measurement

    if nearest is None:
        return None

    if max_distance is not None:
        if nearest_distance_squared > (
            max_distance * max_distance
        ):
            return None

    return nearest


def _as_finite_float(
    value: Any,
) -> float | None:
    """
    Convert a value to a finite float without exposing an unsafe Any/None
    value to static type checking.
    """
    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not (
        numeric_value == numeric_value
        and abs(numeric_value) != float("inf")
    ):
        return None

    return numeric_value

def _normalised_rank(
    *,
    position: int,
    count: int,
) -> float:
    """Return a deterministic rank in the inclusive range [0, 1]."""
    if count <= 1:
        return 1.0 if count == 1 else 0.0

    return 1.0 - (position / (count - 1))


def _is_finite_number(
    value: Any,
) -> bool:
    """Return whether a scalar value can be represented as finite float."""
    if value is None:
        return False

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False

    return numeric == numeric and abs(numeric) != float("inf")


def _time_axis_label(
    results: Sequence[WorkflowResult],
) -> str:
    """
    Return a useful time-axis label based on the authoritative observation
    metadata.

    TESS currently exposes numeric BTJD-like values, so the label remains
    survey-neutral while identifying the underlying time representation.
    """
    if not results:
        return "Observation Time"

    first_time = getattr(
        results[0].observation.metadata,
        "observation_time",
        None,
    )

    if first_time is None:
        return "Observation Time"

    # Numeric TESS observation times are BTJD-like values.
    if isinstance(first_time, (int, float)):
        return "Observation Time (BTJD)"

    return "Observation Time"


# ============================================================================
# Optional catalogue cross-match
# ============================================================================


def _run_catalogue_crossmatch(
    *,
    config: ExperimentConfig,
    source_coordinates: tuple,
) -> tuple:
    """
    Cross-match the supplied source coordinates against the configured
    survey catalogue.

    This function is deliberately outside the production workflow.
    """
    if not source_coordinates:
        logger.info(
            "Catalogue cross-match skipped: no source coordinates supplied.",
        )
        return tuple()

    survey = getattr(
        config,
        "survey",
        None,
    )

    if survey is None:
        logger.warning(
            "Catalogue cross-match skipped: survey is not configured.",
        )
        return tuple()

    logger.info(
        "Starting %s catalogue cross-match for %d source(s).",
        _enum_value(survey) or "configured survey",
        len(source_coordinates),
    )

    crossmatcher = CatalogueCrossmatch(
        survey=survey,
        logger=logger,
        search_radius_arcsec=5.0,
    )

    try:
        results = crossmatcher.search_many(
            coordinates=source_coordinates,
        )
    except RuntimeError as exc:
        logger.error(
            "%s catalogue cross-match failed: %s",
            _enum_value(survey) or "Configured survey",
            exc,
        )
        return tuple()

    logger.info(
        "%s catalogue cross-match completed.",
        _enum_value(survey) or "Configured survey",
    )

    total_candidates = sum(
        len(result.candidates)
        for result in results
    )

    logger.info(
        (
            "Catalogue cross-match summary: "
            "%d source(s), %d candidate object(s)."
        ),
        len(results),
        total_candidates,
    )

    for result in results:
        logger.info(
            (
                "Catalogue results for source %s "
                "(RA=%.8f deg, Dec=%.8f deg, "
                "search radius=%.2f arcsec): "
                "%d candidate(s)."
            ),
            result.source_identifier,
            result.source_right_ascension,
            result.source_declination,
            result.search_radius_arcsec,
            len(result.candidates),
        )

        if not result.candidates:
            logger.info(
                "  No %s catalogue candidates found.",
                _enum_value(survey) or "configured",
            )
            continue

        for index, candidate in enumerate(
            result.candidates,
            start=1,
        ):
            _log_catalogue_candidate(
                survey=survey,
                index=index,
                candidate=candidate,
            )

    return tuple(results)


def _log_differential_catalogue_resolution(
    *,
    config: ExperimentConfig,
    source_coordinates: tuple,
    catalogue_results: tuple,
) -> None:
    """
    Log catalogue identification for selected differential candidates.

    Catalogue identification follows the same nearest-association semantics
    already used by the crowded-source catalogue workflow. The existing
    cross-match radius remains the boundary; no differential-only positional
    threshold is introduced here.

    The existing differential candidate selection logic is untouched.
    """
    del config

    results_by_source = {
        result.source_identifier: result
        for result in catalogue_results
    }

    logger.info(
        "Differential catalogue resolution: evaluating %d selected candidate(s).",
        len(source_coordinates),
    )

    resolved_count = 0

    for rank, coordinate in enumerate(source_coordinates, start=1):
        result = results_by_source.get(coordinate.source_identifier)

        if result is None:
            logger.warning(
                "Differential candidate %d (source %s): no catalogue result returned.",
                rank,
                coordinate.source_identifier,
            )
            continue

        candidates = tuple(getattr(result, "candidates", ()))

        if not candidates:
            logger.warning(
                (
                    "Differential candidate %d (source %s): UNRESOLVED — "
                    "no catalogue target within %.2f arcsec."
                ),
                rank,
                coordinate.source_identifier,
                getattr(result, "search_radius_arcsec", 5.0),
            )
            continue

        def _candidate_separation_key(candidate: Any) -> float:
            separation_value = _as_finite_float(
                getattr(candidate, "separation_arcsec", None)
            )
            return (
                separation_value
                if separation_value is not None
                else float("inf")
            )

        nearest = min(
            candidates,
            key=_candidate_separation_key,
        )

        separation = _as_finite_float(
            getattr(nearest, "separation_arcsec", None)
        )
        catalogue = getattr(nearest, "catalogue", "UNKNOWN")
        object_identifier = getattr(
            nearest,
            "object_identifier",
            "UNKNOWN",
        )

        # Match the existing crowded-source catalogue semantics: a source
        # with one or more catalogue candidates inside the configured
        # cross-match radius is catalogue-identified by its nearest
        # association. Do not introduce a differential-only positional
        # threshold here.
        is_resolved = (
            separation is not None
            and object_identifier not in (None, "UNKNOWN", "")
        )

        if is_resolved:
            resolved_count += 1
            logger.info(
                (
                    "RESOLVED differential target %d: source %s -> %s %s "
                    "(separation=%.3f arcsec, RA=%s deg, Dec=%s deg, "
                    "catalogue candidates=%d)."
                ),
                rank,
                coordinate.source_identifier,
                catalogue,
                object_identifier,
                separation,
                _format_optional_float(
                    getattr(nearest, "right_ascension", None),
                    precision=8,
                ),
                _format_optional_float(
                    getattr(nearest, "declination", None),
                    precision=8,
                ),
                len(candidates),
            )

            if len(candidates) > 1:
                logger.info(
                    (
                        "  Differential source %s had %d catalogue "
                        "candidate(s); nearest association selected: %s "
                        "(%.3f arcsec)."
                    ),
                    coordinate.source_identifier,
                    len(candidates),
                    object_identifier,
                    separation,
                )
        else:
            logger.warning(
                (
                    "Differential candidate %d (source %s): catalogue "
                    "candidate returned without a usable object identifier; "
                    "NOT RESOLVED."
                ),
                rank,
                coordinate.source_identifier,
            )

            for index, candidate in enumerate(candidates, start=1):
                _log_catalogue_candidate(
                    survey=getattr(nearest, "catalogue", None),
                    index=index,
                    candidate=candidate,
                )

    logger.info(
        (
            "Differential catalogue resolution completed: %d/%d selected "
            "candidate(s) resolved to confident catalogue targets."
        ),
        resolved_count,
        len(source_coordinates),
    )


def _select_catalogue_identified_sources(
    *,
    source_coordinates: tuple,
    catalogue_results: tuple,
    neighbour_counts: dict,
    peak_values: dict[str, float | None],
    max_sources: int = 20,
) -> tuple:
    """
    Rank the crowded candidate pool after catalogue cross-matching.

    Catalogue association is deliberately a *final-stage* criterion rather
    than an initial discovery criterion. The bounded candidate pool has
    already been selected from genuinely crowded detections. Within that
    pool, the final diagnostic ranking combines:

    * 45% crowding rank;
    * 30% image brightness rank; and
    * 25% catalogue-association quality.

    Catalogue association quality is based primarily on the nearest catalogue
    separation, not raw candidate count. A very large candidate count inside
    a 5 arcsec search radius can reflect field density rather than a reliable
    identification. A close counterpart therefore carries more information.

    Sources without a catalogue candidate remain in the candidate pool but
    receive zero catalogue-association score and are ranked below otherwise
    comparable identified sources.
    """
    coordinate_by_identifier = {
        coordinate.source_identifier: coordinate
        for coordinate in source_coordinates
    }

    result_by_identifier = {
        result.source_identifier: result
        for result in catalogue_results
    }

    candidate_identifiers = tuple(coordinate_by_identifier)

    crowding_order = tuple(
        sorted(
            candidate_identifiers,
            key=lambda identifier: (
                -neighbour_counts.get(identifier, 0),
                identifier,
            ),
        )
    )

    finite_brightness = tuple(
        identifier
        for identifier in candidate_identifiers
        if peak_values.get(identifier) is not None
    )
    def _catalogue_brightness_sort_key(identifier: str) -> tuple[float, str]:
        peak = peak_values.get(identifier)
        return (
            -(float(peak) if peak is not None else 0.0),
            identifier,
        )

    brightness_order = tuple(
        sorted(
            finite_brightness,
            key=_catalogue_brightness_sort_key,
        )
    )

    crowding_rank = {
        identifier: _normalised_rank(
            position=position,
            count=len(crowding_order),
        )
        for position, identifier in enumerate(crowding_order)
    }
    brightness_rank = {
        identifier: _normalised_rank(
            position=position,
            count=len(brightness_order),
        )
        for position, identifier in enumerate(brightness_order)
    }

    def nearest_catalogue_separation(identifier: str) -> float | None:
        result = result_by_identifier.get(identifier)
        if result is None:
            return None

        separations = tuple(
            separation
            for candidate in getattr(result, "candidates", ())
            for separation in (
                _as_finite_float(
                    getattr(candidate, "separation_arcsec", None),
                ),
            )
            if separation is not None
        )

        if not separations:
            return None

        return min(separations)

    separation_by_identifier = {
        identifier: nearest_catalogue_separation(identifier)
        for identifier in candidate_identifiers
    }

    # Convert the nearest separation into a bounded association-quality score.
    # A 0 arcsec match approaches 1.0; a 5 arcsec edge-of-search-radius
    # match approaches 0.0. The exponential form gives meaningful preference
    # to sub-arcsecond associations without treating 0.01 vs 0.02 arcsec as
    # a dramatic scientific difference.
    def catalogue_association_score(identifier: str) -> float:
        separation = separation_by_identifier.get(identifier)
        if separation is None:
            return 0.0

        search_radius = 5.0
        return max(
            0.0,
            min(
                1.0,
                1.0 - separation / search_radius,
            ),
        )

    def final_score(identifier: str) -> float:
        return (
            0.45 * crowding_rank.get(identifier, 0.0)
            + 0.30 * brightness_rank.get(identifier, 0.0)
            + 0.25 * catalogue_association_score(identifier)
        )

    ranked_identifiers = tuple(
        sorted(
            candidate_identifiers,
            key=lambda identifier: (
                -final_score(identifier),
                separation_by_identifier[identifier]
                if separation_by_identifier[identifier] is not None
                else float("inf"),
                -neighbour_counts.get(identifier, 0),
                -(peak_values[identifier] or 0.0),
                identifier,
            ),
        )
    )

    selected = tuple(
        coordinate_by_identifier[identifier]
        for identifier in ranked_identifiers[:max_sources]
    )

    identified_count = sum(
        separation_by_identifier[identifier] is not None
        for identifier in ranked_identifiers
    )

    logger.info(
        (
            "FIELD_ANALYSIS catalogue-aware ranking: %d candidate source(s), "
            "%d catalogue-identified, final visualisation cap=%d; "
            "ranking = 45%% crowding + 30%% brightness + 25%% catalogue association."
        ),
        len(candidate_identifiers),
        identified_count,
        max_sources,
    )

    for rank, identifier in enumerate(
        ranked_identifiers[:max_sources],
        start=1,
    ):
        result = result_by_identifier.get(identifier)
        candidate_count = (
            len(getattr(result, "candidates", ()))
            if result is not None
            else 0
        )

        logger.info(
            (
                "Final crowded candidate rank %d: source %s, score=%.4f, "
                "neighbours=%d, peak=%s, nearest-catalogue-separation=%s arcsec, "
                "catalogue-candidates=%d."
            ),
            rank,
            identifier,
            final_score(identifier),
            neighbour_counts.get(identifier, 0),
            _format_optional_float(
                peak_values.get(identifier),
                precision=3,
            ),
            _format_optional_float(
                separation_by_identifier.get(identifier),
                precision=3,
            ),
            candidate_count,
        )

    if identified_count < min(10, len(candidate_identifiers)):
        logger.warning(
            (
                "Only %d of %d crowded candidate source(s) have a catalogue "
                "association within the configured search radius. The final "
                "ranking still includes all crowded candidates, but unmatched "
                "sources receive zero catalogue-association score."
            ),
            identified_count,
            len(candidate_identifiers),
        )

    return selected


def _select_default_field_visualisation_sources(
    *,
    source_coordinates: tuple,
    neighbour_counts: dict,
) -> tuple:
    """Return a bounded crowded-only diagnostic display subset."""
    crowded = sorted(
        (
            coordinate
            for coordinate in source_coordinates
            if neighbour_counts.get(coordinate.source_identifier, 0) > 0
        ),
        key=lambda coordinate: (
            -neighbour_counts.get(coordinate.source_identifier, 0),
            coordinate.source_identifier,
        ),
    )

    return tuple(crowded[:20])


def _log_catalogue_candidate(
    *,
    survey: Any,
    index: int,
    candidate: Any,
) -> None:
    """Log a catalogue candidate using only optional public attributes."""
    catalogue_name = getattr(
        candidate,
        "catalogue",
        _enum_value(survey) or "UNKNOWN",
    )

    object_identifier = getattr(
        candidate,
        "object_identifier",
        "UNKNOWN",
    )

    separation = getattr(
        candidate,
        "separation_arcsec",
        None,
    )

    right_ascension = getattr(
        candidate,
        "right_ascension",
        None,
    )

    declination = getattr(
        candidate,
        "declination",
        None,
    )

    object_type = getattr(
        candidate,
        "object_type",
        None,
    )

    logger.info(
        (
            "  Candidate %d: catalogue=%s, ID=%s, "
            "separation=%s arcsec, RA=%s deg, Dec=%s deg, type=%s."
        ),
        index,
        catalogue_name,
        object_identifier,
        _format_optional_float(
            separation,
            precision=3,
        ),
        _format_optional_float(
            right_ascension,
            precision=8,
        ),
        _format_optional_float(
            declination,
            precision=8,
        ),
        object_type if object_type is not None else "N/A",
    )

    sdss_fields = (
        ("u", "u_magnitude"),
        ("g", "g_magnitude"),
        ("r", "r_magnitude"),
        ("i", "i_magnitude"),
        ("z", "z_magnitude"),
    )

    available_sdss_fields: list[str] = []

    for band, attribute in sdss_fields:
        value = getattr(
            candidate,
            attribute,
            None,
        )

        if value is not None:
            available_sdss_fields.append(
                f"{band}={_format_magnitude(value)}",
            )

    if available_sdss_fields:
        logger.info(
            "    SDSS photometry: %s.",
            ", ".join(available_sdss_fields),
        )

    tic_attributes = (
        ("Tmag", "tmag"),
        ("Gmag", "gmag"),
        ("Vmag", "vmag"),
        ("Bmag", "bmag"),
    )

    available_tic_fields: list[str] = []

    for label, attribute in tic_attributes:
        value = getattr(
            candidate,
            attribute,
            None,
        )

        if value is not None:
            available_tic_fields.append(
                f"{label}={_format_magnitude(value)}",
            )

    if available_tic_fields:
        logger.info(
            "    TIC photometry: %s.",
            ", ".join(available_tic_fields),
        )


# ============================================================================
# Formatting and logging
# ============================================================================


def _format_optional_float(
    value: Any,
    *,
    precision: int,
) -> str:
    """Format an optional numeric value without assuming its concrete type."""
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_magnitude(
    value: Any,
) -> str:
    """Format an optional catalogue magnitude."""
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _configure_demo_logging() -> None:
    """Keep third-party plotting output from overwhelming the demo log."""
    logging.getLogger("matplotlib").setLevel(
        logging.WARNING,
    )
    logging.getLogger("PIL").setLevel(
        logging.WARNING,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Run the demonstration through src/demo.py so the configured "
        "experiment suite and demo-only cadence limits are applied."
    )
