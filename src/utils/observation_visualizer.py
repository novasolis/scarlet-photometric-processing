"""
Observation visualisation utilities.

Provides visual rendering of Observation domain objects and associated
photometric analysis results.

The visualiser operates entirely on the project's survey-independent
domain models, allowing images from different astronomical surveys and
processing stages to be displayed using the same interface.

Visualisation is strictly a presentation concern.

The visualiser:

    * does not modify the Observation;
    * does not perform source detection;
    * does not perform photometry;
    * does not perform target resolution;
    * does not calculate scientific evaluation metrics.

All scientific quantities displayed by this module are obtained from
already-computed domain results.

Supported visualisations include:

    * raw images;
    * estimated background;
    * background-subtracted images;
    * detected source positions;
    * aperture and background-annulus geometry;
    * resolved target-source association;
    * source-quality classifications;
    * target-region detail views;
    * target photometry summaries;
    * crowded-source pixel/sky-coordinate diagnostics;
    * target light curves from authoritative time-series records;
    * normalised light-curve presentation views;
    * differential-photometry light curves;
    * experiment-set summary tables;
    * authoritative LocalField selection;
    * SCARLET field reconstruction and intrinsic model views;
    * SCARLET field residual diagnostics; and
    * representative individual SCARLET source-model cutouts.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from collections.abc import Iterable, Mapping
from logging import Logger
from math import isfinite
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle
import numpy as np
from astropy.visualization import (
    AsinhStretch,
    ImageNormalize,
    PercentileInterval,
    ZScaleInterval,
)
from numpy.typing import NDArray

from domain.models import LocalField, Observation, TargetSourceMatch
from evaluation.models import PhotometryEvaluation, SourceEvaluation
from utils.logger import get_logger

if TYPE_CHECKING:
    from utils.crowded_source_coordinates import CrowdedSourceCoordinate


class ObservationVisualizer:
    """
    Visualise Observation objects and completed analysis products.

    Images may be selected from any image stored within an Observation.
    Optional overlays can display detected sources, photometric geometry,
    target-source resolution and source-level quality classifications.

    The visualiser is intentionally independent of survey-specific
    acquisition libraries and photometric implementations.

    Parameters
    ----------
    logger
        Optional logger used for diagnostic messages.
    """

    def __init__(
        self,
        logger: Logger | None = None,
    ) -> None:
        self._logger = logger or get_logger(__name__)

        # The authoritative LocalField is established by the production
        # workflow and consumed by the demo/visualisation sequence. Once the
        # LocalField diagnostic has been rendered, subsequent image-based
        # diagnostics for the same Observation automatically use the same
        # target-centred field of view. This keeps the presentation aligned
        # with the actual photometric analysis region without changing any
        # scientific data or selection logic.
        self._active_local_field: LocalField | None = None
        self._active_local_field_observation_id: int | None = None

    # Presentation constants for the target-centred LocalField view. The
    # LocalField selection radius is deliberately not used as the complete
    # display boundary: a modest surrounding margin is retained so that the
    # selected sources can be seen in their immediate image context.
    _LOCAL_FIELD_VIEW_SCALE = 2.75
    _LOCAL_FIELD_MIN_VIEW_RADIUS = 30.0

    # The demo owns presentation sequencing. Baseline diagnostics and
    # SCARLET products remain separate; this visualizer exposes rendering
    # primitives but never invokes baseline rendering from a SCARLET method.
    # LocalField activation below is presentation context only.

    def activate_local_field(
        self,
        observation: Observation,
        *,
        local_field: LocalField,
    ) -> None:
        """Set the authoritative LocalField as display context without rendering it.

        This is intentionally presentation-only. It does not alter the
        Observation or perform source selection. It allows SCARLET-only
        rendering to use the production LocalField crop without re-rendering
        baseline diagnostics.
        """
        if local_field is None:
            raise ValueError("LocalField presentation context cannot be None.")

        self._active_local_field = local_field
        self._active_local_field_observation_id = id(observation)

        self._logger.info(
            "Activated LocalField presentation context: target=%s, selected=%d, "
            "neighbours=%d, radius=%.3f px.",
            local_field.target_source.identifier,
            local_field.selected_source_count,
            local_field.selected_neighbour_count,
            local_field.selection_radius_pixels,
        )

    def render_light_curve(
        self,
        time: object,
        flux: object,
        *,
        target_name: str = "Target",
        time_label: str = "Observation Time",
        flux_label: str = "Net Flux",
        title: str | None = None,
        normalise: bool = False,
        valid: object | None = None,
        show_grid: bool = True,
    ) -> None:
        """Render an authoritative target light curve.

        This method is presentation-only. ``time`` and ``flux`` must be
        supplied by an upstream time-series workflow or persisted export.
        The only transformation performed here is optional division by the
        median finite flux for visual normalisation; the supplied data are
        never modified.

        ``valid`` may be supplied as a boolean sequence. Invalid samples are
        displayed separately rather than silently discarded.
        """
        time_values, flux_values = self._prepare_series(
            time,
            flux,
            valid=valid,
        )

        if normalise:
            finite = np.isfinite(flux_values)
            if not np.any(finite):
                raise ValueError(
                    "Cannot normalise a light curve with no finite flux values."
                )
            scale = float(np.nanmedian(flux_values[finite]))
            if not isfinite(scale) or scale == 0.0:
                raise ValueError(
                    "Cannot normalise a light curve with a zero or non-finite median flux."
                )
            display_flux = flux_values / scale
            display_label = f"Normalised {flux_label}"
        else:
            display_flux = flux_values
            display_label = flux_label

        if title is None:
            suffix = " — Normalised" if normalise else ""
            title = f"{target_name} • Target Light Curve{suffix}"

        self._render_series(
            time_values,
            display_flux,
            title=title,
            time_label=time_label,
            value_label=display_label,
            valid=valid,
            show_grid=show_grid,
        )

    def render_differential_light_curve(
        self,
        time: object,
        differential_magnitude: object,
        *,
        target_name: str = "Target",
        time_label: str = "Observation Time",
        title: str | None = None,
        valid: object | None = None,
        show_reference_level: bool = True,
        show_grid: bool = True,
    ) -> None:
        """Render an authoritative differential-photometry light curve.

        The supplied differential magnitudes are plotted unchanged. A
        reference line, when requested, is the median of the finite supplied
        differential magnitudes and is therefore a presentation aid only.
        No differential-photometry calculation, filtering, smoothing,
        detrending or reference-ensemble reconstruction occurs here.
        """
        time_values, magnitude_values = self._prepare_series(
            time,
            differential_magnitude,
            valid=valid,
        )

        if title is None:
            title = (
                f"{target_name} • Differential Photometry"
            )

        self._render_series(
            time_values,
            magnitude_values,
            title=title,
            time_label=time_label,
            value_label="Differential Magnitude",
            valid=valid,
            reference_value=(
                float(np.nanmedian(magnitude_values))
                if show_reference_level
                and np.any(np.isfinite(magnitude_values))
                else None
            ),
            reference_label="Median differential magnitude",
            invert_y=True,
            show_grid=show_grid,
        )

    def render_experiment_summary(
        self,
        rows: Iterable[object],
        *,
        title: str = "Experiment Set Summary",
        columns: tuple[str, ...] | None = None,
    ) -> None:
        """Render a compact experiment-set summary table.

        ``rows`` may contain mappings or objects with attributes matching
        ``columns``. This method performs no scientific aggregation; all
        displayed values must already have been produced by the workflow or
        export layer.
        """
        if rows is None:
            raise ValueError("Experiment summary rows cannot be None.")

        row_sequence = tuple(rows)
        if not row_sequence:
            raise ValueError("Experiment summary rows cannot be empty.")

        if columns is None:
            columns = self._infer_summary_columns(row_sequence[0])

        if not columns:
            raise ValueError("Experiment summary requires at least one column.")

        table_rows = [
            [
                self._format_table_value(
                    self._get_record_value(row, column),
                )
                for column in columns
            ]
            for row in row_sequence
        ]

        figure_height = max(
            3.2,
            min(18.0, 1.8 + 0.42 * len(table_rows)),
        )

        figure, axis = plt.subplots(
            figsize=(12, figure_height),
        )

        try:
            axis.axis("off")
            table = axis.table(
                cellText=table_rows,
                colLabels=[self._humanise_column(column) for column in columns],
                cellLoc="center",
                colLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.45)

            axis.set_title(
                title,
                fontsize=14,
                fontweight="bold",
                pad=12,
            )

            figure.tight_layout()
            plt.show()
        finally:
            plt.close(figure)

    def _render_series(
        self,
        time_values: NDArray[np.float64],
        value_values: NDArray[np.float64],
        *,
        title: str,
        time_label: str,
        value_label: str,
        valid: object | None = None,
        reference_value: float | None = None,
        reference_label: str | None = None,
        invert_y: bool = False,
        show_grid: bool = True,
    ) -> None:
        """Render a prepared numerical series without scientific processing."""
        finite = np.isfinite(time_values) & np.isfinite(value_values)

        if not np.any(finite):
            raise ValueError("Time-series data contain no finite samples.")

        valid_mask = self._coerce_valid_mask(valid, len(time_values))
        valid_finite = finite & valid_mask
        invalid_finite = finite & ~valid_mask

        figure, axis = plt.subplots(figsize=(10, 6))

        try:
            if np.any(valid_finite):
                axis.plot(
                    time_values[valid_finite],
                    value_values[valid_finite],
                    linestyle="none",
                    marker="o",
                    markersize=3.0,
                    alpha=0.78,
                    label="Valid measurements",
                )

            if np.any(invalid_finite):
                axis.plot(
                    time_values[invalid_finite],
                    value_values[invalid_finite],
                    linestyle="none",
                    marker="x",
                    markersize=5.0,
                    alpha=0.80,
                    label="Invalid measurements",
                )

            if reference_value is not None:
                axis.axhline(
                    reference_value,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.60,
                    label=reference_label or "Reference level",
                )

            axis.set_title(
                title,
                fontsize=14,
                fontweight="bold",
                pad=10,
            )
            axis.set_xlabel(time_label)
            axis.set_ylabel(value_label)

            if invert_y:
                axis.invert_yaxis()

            if show_grid:
                axis.grid(
                    True,
                    alpha=0.20,
                    linewidth=0.7,
                )

            self._refresh_legend(axis)
            figure.tight_layout()
            plt.show()
        finally:
            plt.close(figure)

    @staticmethod
    def _prepare_series(
        time: object,
        values: object,
        *,
        valid: object | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validate and copy a time/value series for presentation."""
        if time is None or values is None:
            raise ValueError("Time-series time and values cannot be None.")

        try:
            time_values = np.asarray(time, dtype=float).reshape(-1)
            value_values = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Time-series time and values must be one-dimensional numeric sequences."
            ) from exc

        if len(time_values) != len(value_values):
            raise ValueError(
                "Time-series time and value lengths must match: "
                f"{len(time_values)} != {len(value_values)}."
            )

        if len(time_values) == 0:
            raise ValueError("Time-series data cannot be empty.")

        if valid is not None:
            try:
                valid_values = np.asarray(valid, dtype=bool).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Valid-measurement mask must be a boolean sequence."
                ) from exc

            if len(valid_values) != len(time_values):
                raise ValueError(
                    "Valid-measurement mask length must match the time-series length."
                )

        return time_values.copy(), value_values.copy()

    @staticmethod
    def _coerce_valid_mask(
        valid: object | None,
        length: int,
    ) -> NDArray[np.bool_]:
        """Return a presentation mask; missing masks mean all samples valid."""
        if valid is None:
            return np.ones(length, dtype=bool)

        try:
            mask = np.asarray(valid, dtype=bool).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Valid-measurement mask must be a boolean sequence."
            ) from exc

        if len(mask) != length:
            raise ValueError(
                "Valid-measurement mask length must match the time-series length."
            )

        return mask

    @staticmethod
    def _get_record_value(
        record: object,
        field: str,
    ) -> object:
        """Read a field from either a mapping or an object attribute."""
        if isinstance(record, Mapping):
            return record.get(field)
        return getattr(record, field, None)

    @staticmethod
    def _infer_summary_columns(record: object) -> tuple[str, ...]:
        """Infer useful public fields without relying on private attributes."""
        if isinstance(record, Mapping):
            return tuple(str(key) for key in record.keys())

        if is_dataclass(record) and not isinstance(record, type):
            return tuple(field.name for field in fields(record))

        return tuple(
            name
            for name in dir(record)
            if not name.startswith("_")
            and not callable(getattr(record, name, None))
        )

    @staticmethod
    def _format_table_value(value: object) -> str:
        """Format a table value without changing its semantic meaning."""
        if value is None:
            return "N/A"

        if isinstance(value, float):
            if not isfinite(value):
                return "N/A"
            return f"{value:.5g}"

        return str(value)

    @staticmethod
    def _humanise_column(column: str) -> str:
        """Convert an internal field name into a readable table heading."""
        return column.replace("_", " ").strip().title()

    def render(
        self,
        observation: Observation,
        *,
        image_name: str | None = None,
        title: str | None = None,
        show_colorbar: bool = True,
        show_detections: bool = False,
        show_photometry: bool = False,
        show_target: bool = False,
        show_quality: bool = False,
        target_match: TargetSourceMatch | None = None,
        evaluation: PhotometryEvaluation | None = None,
        max_photometry_overlays: int | None = None,
        max_quality_overlays: int | None = None,
    ) -> None:
        """
        Render an Observation image.

        Parameters
        ----------
        observation
            Observation to display.

        image_name
            Name of the stored image to render. If omitted, the active
            image is displayed.

        title
            Optional plot title. If omitted, an informative title is
            generated automatically.

        show_colorbar
            Display the image colour scale.

        show_detections
            Overlay detected source centroids.

        show_photometry
            Overlay source apertures and local background annuli.

        show_target
            Overlay the resolved target-source association.

            ``target_match`` must be supplied for this overlay to be
            rendered.

        show_quality
            Classify detected sources using the already-computed
            SourceEvaluation records.

            ``evaluation`` must be supplied for this overlay to be
            rendered.

        target_match
            Completed target-source association produced by the target
            resolver.

        evaluation
            Completed photometric evaluation containing source-level
            quality records.

        max_photometry_overlays
            Maximum number of photometric measurement geometries to
            render. ``None`` renders every available measurement.

        max_quality_overlays
            Maximum number of source-quality classifications to render.
            ``None`` renders every detected source.

        Raises
        ------
        ValueError
            If the requested image does not exist, is not two-dimensional,
            contains no finite values, or an overlay limit is invalid.
        """

        self._validate_render_options(
            max_photometry_overlays=max_photometry_overlays,
            max_quality_overlays=max_quality_overlays,
        )

        image_data, image_label = self._get_image(
            observation,
            image_name,
        )

        image = np.asarray(
            image_data.image.data,
        )

        self._validate_image(
            image,
            image_label,
        )

        if title is None:
            title = self._create_title(
                observation=observation,
                image_name=image_label,
                show_detections=show_detections,
                show_photometry=show_photometry,
                show_target=show_target,
                show_quality=show_quality,
                evaluation=evaluation,
            )

        self._logger.info(
            "Rendering image '%s' for target '%s'.",
            image_label,
            observation.metadata.target_name,
        )

        normalisation = self._create_normalisation(
            cast(
                NDArray[np.number],
                image,
            ),
            image_label,
        )

        figure, axis = plt.subplots(
            figsize=(9, 9),
        )

        try:
            rendered = axis.imshow(
                image,
                origin="lower",
                cmap="gray",
                interpolation="nearest",
                norm=normalisation,
            )

            # The standalone detection stage is the only downstream image
            # view that needs the complete detector-wide source catalogue.
            # Repeating all detections under quality/photometry/target
            # overlays obscures the actual LocalField result.
            show_detection_overlay = (
                show_detections
                and not (show_quality or show_photometry or show_target)
            )
            if show_detection_overlay:
                self._overlay_detections(
                    axis,
                    observation,
                )

            if show_quality:
                self._overlay_quality(
                    axis,
                    observation,
                    evaluation=evaluation,
                    max_overlays=max_quality_overlays,
                )

            if show_photometry:
                self._overlay_photometry(
                    axis,
                    observation,
                    max_overlays=max_photometry_overlays,
                    target_match=target_match,
                )

            if show_target:
                self._overlay_target(
                    axis,
                    target_match=target_match,
                )

            self._apply_active_local_field_view(
                axis,
                observation,
                image.shape,
            )
            self._overlay_active_local_field(
                axis,
                observation,
                image.shape,
            )

            # In the representative baseline workflow an authoritative LocalField
            # is activated before this diagnostic is rendered. Keep FIELD_ANALYSIS
            # unchanged: it does not activate a LocalField and therefore remains
            # detector-wide.
            self._apply_active_local_field_view(
                axis,
                observation,
                image.shape,
            )

            axis.set_title(
                title,
                fontsize=14,
                fontweight="bold",
            )

            axis.set_xlabel(
                "Pixel X",
            )

            axis.set_ylabel(
                "Pixel Y",
            )

            if show_colorbar:
                figure.colorbar(
                    rendered,
                    ax=axis,
                    label="Normalised Intensity",
                )

            self._refresh_legend(
                axis,
            )

            figure.tight_layout()

            plt.show()

        finally:
            plt.close(
                figure,
            )

    def render_target_detail(
        self,
        observation: Observation,
        *,
        target_match: TargetSourceMatch,
        evaluation: PhotometryEvaluation | None = None,
        image_name: str | None = None,
        title: str | None = None,
        zoom_radius: float | None = None,
        show_colorbar: bool = True,
    ) -> None:
        """
        Render a detailed view centred on the resolved target source.

        The view displays the target's expected WCS position, the matched
        detected-source centroid and the production aperture/annulus
        geometry.

        Parameters
        ----------
        observation
            Completed Observation to display.

        target_match
            Completed target-source association produced by the target
            resolver.

        evaluation
            Optional completed photometric evaluation used to display
            target quality information.

        image_name
            Name of the image to render. If omitted, the active image is
            displayed.

        title
            Optional plot title.

        zoom_radius
            Half-width of the displayed target region in pixels.
            If omitted, the value is derived from the production
            photometry geometry.

        show_colorbar
            Display the image colour scale.

        Raises
        ------
        ValueError
            If the image is invalid or the zoom radius is invalid.
        """

        if zoom_radius is not None:
            if not isfinite(zoom_radius) or zoom_radius <= 0:
                raise ValueError(
                    "zoom_radius must be finite and greater than zero.",
                )

        image_data, image_label = self._get_image(
            observation,
            image_name,
        )

        image = np.asarray(
            image_data.image.data,
        )

        self._validate_image(
            image,
            image_label,
        )

        if zoom_radius is None:
            active_view_radius = self._active_local_field_view_radius(
                observation,
            )
            if active_view_radius is not None:
                zoom_radius = active_view_radius
            else:
                zoom_radius = self._calculate_target_zoom_radius(
                    observation,
                )

        centre_x = float(
            target_match.source_x,
        )

        centre_y = float(
            target_match.source_y,
        )

        height, width = image.shape

        x_min = max(
            0.0,
            centre_x - zoom_radius,
        )

        x_max = min(
            float(width - 1),
            centre_x + zoom_radius,
        )

        y_min = max(
            0.0,
            centre_y - zoom_radius,
        )

        y_max = min(
            float(height - 1),
            centre_y + zoom_radius,
        )

        normalisation = self._create_normalisation(
            cast(
                NDArray[np.number],
                image,
            ),
            image_label,
        )

        if title is None:
            title = (
                f"{observation.provenance.survey.value} • "
                f"{observation.metadata.target_name}\n"
                f"Target Detail • "
                f"{target_match.source_identifier}"
            )

        self._logger.info(
            "Rendering target detail for '%s' "
            "around source '%s'.",
            target_match.target_name,
            target_match.source_identifier,
        )

        figure, axis = plt.subplots(
            figsize=(8, 8),
        )

        try:
            rendered = axis.imshow(
                image,
                origin="lower",
                cmap="gray",
                interpolation="nearest",
                norm=normalisation,
            )

            self._set_view_limits(
                axis,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )

            self._apply_active_local_field_view(
                axis,
                observation,
                image.shape,
            )
            self._overlay_active_local_field(
                axis,
                observation,
                image.shape,
            )

            self._overlay_target(
                axis,
                target_match=target_match,
                include_separation=True,
            )

            self._overlay_target_photometry(
                axis,
                observation,
                target_match=target_match,
            )

            if evaluation is not None:
                self._overlay_target_quality(
                    axis,
                    evaluation=evaluation,
                    target_match=target_match,
                )

            axis.set_title(
                title,
                fontsize=14,
                fontweight="bold",
            )

            axis.set_xlabel(
                "Pixel X",
            )

            axis.set_ylabel(
                "Pixel Y",
            )

            if show_colorbar:
                figure.colorbar(
                    rendered,
                    ax=axis,
                    label="Normalised Intensity",
                )

            self._refresh_legend(
                axis,
            )

            figure.tight_layout()

            plt.show()

        finally:
            plt.close(
                figure,
            )

    def render_local_field(
        self,
        observation: Observation,
        *,
        local_field: LocalField,
        image_name: str | None = None,
        title: str | None = None,
        show_colorbar: bool = True,
        annotate_sources: bool = False,
        show_all_detections: bool = False,
    ) -> None:
        """Render the authoritative target-centred LocalField.

        The detector-wide context is deliberately not repeated here. Earlier
        baseline diagnostics provide the raw/background-removed field and
        crowded-source context. This figure therefore focuses only on the
        actual target-centred region used by downstream photometry.

        ``show_all_detections`` is retained for API compatibility but defaults
        to False because the complete detector catalogue is not useful in the
        focused analytical view.
        """
        if local_field is None:
            raise ValueError("LocalField visualisation requires a LocalField.")

        image_data, image_label = self._get_image(observation, image_name)
        image = np.asarray(image_data.image.data)
        self._validate_image(image, image_label)

        detection_result = observation.detections
        if detection_result is None:
            raise ValueError("LocalField visualisation requires source detections.")

        detected_by_id = {
            str(source.identifier): source
            for source in detection_result.sources
        }
        field_sources = (local_field.target_source, *local_field.neighbour_sources)
        missing = [
            str(source.identifier)
            for source in field_sources
            if str(source.identifier) not in detected_by_id
        ]
        if missing:
            raise ValueError(
                "LocalField contains sources absent from the observation "
                "detection catalogue: " + ", ".join(missing)
            )

        self.activate_local_field(
            observation,
            local_field=local_field,
        )

        target = local_field.target_source
        neighbours = tuple(local_field.neighbour_sources)
        radius = float(local_field.selection_radius_pixels)
        target_x = float(target.x_centroid)
        target_y = float(target.y_centroid)

        if title is None:
            title = (
                f"{observation.provenance.survey.value} • "
                f"{observation.metadata.target_name}\n"
                f"Target-Centred LocalField "
                f"({len(field_sources)}/{local_field.available_source_count} sources)"
            )

        self._logger.info(
            "Rendering focused LocalField: target=%s, selected=%d, neighbours=%d, "
            "available=%d, radius=%.3f px.",
            target.identifier,
            len(field_sources),
            len(neighbours),
            local_field.available_source_count,
            radius,
        )

        normalisation = self._create_normalisation(
            cast(NDArray[np.number], image),
            image_label,
        )
        view_radius = self._active_local_field_view_radius(observation) or 60.0
        height, width = image.shape
        x_min = max(0.0, target_x - view_radius)
        x_max = min(float(width - 1), target_x + view_radius)
        y_min = max(0.0, target_y - view_radius)
        y_max = min(float(height - 1), target_y + view_radius)

        figure, axis = plt.subplots(figsize=(8.5, 8.0))
        try:
            rendered = axis.imshow(
                image,
                origin="lower",
                cmap="gray",
                interpolation="nearest",
                norm=normalisation,
            )

            if show_all_detections:
                all_sources = tuple(detection_result.sources)
                axis.scatter(
                    [float(source.x_centroid) for source in all_sources],
                    [float(source.y_centroid) for source in all_sources],
                    s=18,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=0.45,
                    alpha=0.25,
                    label=f"All detected sources ({len(all_sources)})",
                    zorder=5,
                )

            if neighbours:
                axis.scatter(
                    [float(source.x_centroid) for source in neighbours],
                    [float(source.y_centroid) for source in neighbours],
                    s=120,
                    facecolors="none",
                    edgecolors="cyan",
                    linewidths=2.0,
                    label=f"Neighbours ({len(neighbours)})",
                    zorder=8,
                )

            axis.scatter(
                [target_x],
                [target_y],
                s=190,
                marker="*",
                facecolors="none",
                edgecolors="yellow",
                linewidths=2.2,
                label=f"Target ({target.identifier})",
                zorder=10,
            )

            if isfinite(radius) and radius > 0:
                axis.add_patch(
                    Circle(
                        (target_x, target_y),
                        radius=radius,
                        fill=False,
                        linestyle="--",
                        linewidth=1.5,
                        edgecolor="yellow",
                        alpha=0.95,
                        label=f"Selection radius ({radius:.2f} px)",
                        zorder=7,
                    )
                )

            if annotate_sources:
                for source in field_sources:
                    axis.annotate(
                        str(source.identifier),
                        (float(source.x_centroid), float(source.y_centroid)),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=8,
                        color="white",
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "facecolor": "black",
                            "alpha": 0.70,
                            "edgecolor": "none",
                        },
                        zorder=11,
                    )

            axis.set_xlim(x_min, x_max)
            axis.set_ylim(y_min, y_max)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(title, fontsize=14, fontweight="bold")
            axis.set_xlabel("Pixel X")
            axis.set_ylabel("Pixel Y")

            axis.text(
                0.02,
                0.02,
                "FOCUSED LOCALFIELD\n"
                f"Target: {target.identifier}\n"
                f"Neighbours: {len(neighbours)}\n"
                f"Selected: {len(field_sources)}",
                transform=axis.transAxes,
                va="bottom",
                ha="left",
                fontsize=9,
                family="monospace",
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": "black",
                    "alpha": 0.78,
                    "edgecolor": "white",
                },
                zorder=20,
            )

            if show_colorbar:
                figure.colorbar(
                    rendered,
                    ax=axis,
                    label="Normalised Intensity",
                )

            self._refresh_legend(axis)
            figure.tight_layout()
            plt.show()
        finally:
            plt.close(figure)

    def render_target_context(
        self,
        observation: Observation,
        *,
        local_field: LocalField,
        image_name: str | None = None,
        title: str | None = None,
        view_radius: float = 90.0,
        show_colorbar: bool = False,
        show_axes: bool = False,
        show_legend: bool = True,
    ) -> None:
        """Render a clean target-centred context view for presentation use.

        This is an additional presentation-only visualisation. It uses the
        authoritative production ``LocalField`` and, when available, the
        production photometry geometry. Unlike ``render_local_field`` it does
        not display the LocalField selection boundary, source counts, source
        identifiers, detector axes or a colour scale by default.

        The view is intended to make one concept visually obvious: a target
        aperture sits inside a locally crowded field. Neighbouring detected
        sources are shown with restrained markers, while the production
        aperture and background annulus are shown when photometry has already
        been computed.

        No scientific selection, source detection, photometry or image
        transformation is performed by this method.
        """
        if local_field is None:
            raise ValueError("Target context visualisation requires a LocalField.")
        if not isfinite(float(view_radius)) or float(view_radius) <= 0.0:
            raise ValueError("view_radius must be finite and greater than zero.")

        image_data, image_label = self._get_image(observation, image_name)
        image = np.asarray(image_data.image.data)
        self._validate_image(image, image_label)

        detection_result = observation.detections
        if detection_result is None:
            raise ValueError("Target context visualisation requires source detections.")

        detected_by_id = {
            str(source.identifier): source
            for source in detection_result.sources
        }
        target = local_field.target_source
        neighbours = tuple(local_field.neighbour_sources)
        field_sources = (target, *neighbours)

        missing = [
            str(source.identifier)
            for source in field_sources
            if str(source.identifier) not in detected_by_id
        ]
        if missing:
            raise ValueError(
                "LocalField contains sources absent from the observation "
                "detection catalogue: " + ", ".join(missing)
            )

        self.activate_local_field(
            observation,
            local_field=local_field,
        )

        target_x = float(target.x_centroid)
        target_y = float(target.y_centroid)
        height, width = image.shape
        radius = float(view_radius)

        x_min = max(0.0, target_x - radius)
        x_max = min(float(width - 1), target_x + radius)
        y_min = max(0.0, target_y - radius)
        y_max = min(float(height - 1), target_y + radius)

        if title is None:
            title = "Target-centred field"

        normalisation = self._create_normalisation(
            cast(NDArray[np.number], image),
            image_label,
        )

        figure, axis = plt.subplots(figsize=(7.0, 7.0))
        try:
            rendered = axis.imshow(
                image,
                origin="lower",
                cmap="gray",
                interpolation="nearest",
                norm=normalisation,
            )

            if neighbours:
                axis.scatter(
                    [float(source.x_centroid) for source in neighbours],
                    [float(source.y_centroid) for source in neighbours],
                    s=85,
                    marker="o",
                    facecolors="none",
                    edgecolors="cyan",
                    linewidths=1.4,
                    alpha=0.85,
                    label="Neighbouring sources",
                    zorder=8,
                )

            axis.scatter(
                [target_x],
                [target_y],
                s=180,
                marker="*",
                facecolors="none",
                edgecolors="yellow",
                linewidths=1.9,
                label="Target",
                zorder=12,
            )

            # Use the already-computed production photometry geometry when it
            # exists. This keeps the presentation graphic tied to the actual
            # measurement rather than introducing an illustrative aperture.
            photometry_result = observation.photometry
            target_measurement = None
            if photometry_result is not None:
                target_measurement = next(
                    (
                        measurement
                        for measurement in photometry_result.measurements
                        if str(measurement.source_identifier)
                        == str(target.identifier)
                    ),
                    None,
                )

            if (
                photometry_result is not None
                and target_measurement is not None
            ):
                measurement_x = float(target_measurement.x_centroid)
                measurement_y = float(target_measurement.y_centroid)
                aperture_radius = float(photometry_result.aperture_radius)
                annulus_inner_radius = float(
                    photometry_result.annulus_inner_radius,
                )
                annulus_outer_radius = float(
                    photometry_result.annulus_outer_radius,
                )

                axis.add_patch(
                    Circle(
                        (measurement_x, measurement_y),
                        radius=aperture_radius,
                        fill=False,
                        edgecolor="lime",
                        linewidth=2.2,
                        label="Target aperture",
                        zorder=10,
                    )
                )
                axis.add_patch(
                    Circle(
                        (measurement_x, measurement_y),
                        radius=annulus_inner_radius,
                        fill=False,
                        edgecolor="lime",
                        linestyle=":",
                        linewidth=1.1,
                        alpha=0.8,
                        zorder=9,
                    )
                )
                axis.add_patch(
                    Circle(
                        (measurement_x, measurement_y),
                        radius=annulus_outer_radius,
                        fill=False,
                        edgecolor="lime",
                        linestyle="--",
                        linewidth=1.1,
                        alpha=0.8,
                        label="Background annulus",
                        zorder=9,
                    )
                )

            axis.set_xlim(x_min, x_max)
            axis.set_ylim(y_min, y_max)
            axis.set_aspect("equal", adjustable="box")

            if show_axes:
                axis.set_xlabel("Pixel X")
                axis.set_ylabel("Pixel Y")
            else:
                axis.set_axis_off()

            axis.set_title(title, fontsize=13, fontweight="bold")

            if show_colorbar:
                figure.colorbar(
                    rendered,
                    ax=axis,
                    label="Normalised Intensity",
                )

            if show_legend:
                self._refresh_legend(axis)

            figure.tight_layout(pad=0.4)
            plt.show()

        finally:
            plt.close(figure)

    def render_scarlet_field_model(
        self,
        observation: Observation,
        *,
        image_name: str = "scarlet_field_model",
        title: str | None = None,
        show_colorbar: bool = True,
    ) -> None:
        """Render the already-generated observation-space SCARLET field model."""
        self._render_scarlet_product(
            observation,
            image_name=image_name,
            title=title,
            show_colorbar=show_colorbar,
        )

    def render_scarlet_field_intrinsic_model(
        self,
        observation: Observation,
        *,
        image_name: str = "scarlet_field_model_intrinsic",
        title: str | None = None,
        show_colorbar: bool = True,
    ) -> None:
        """Render the already-generated intrinsic SCARLET field model."""
        self._render_scarlet_product(
            observation,
            image_name=image_name,
            title=title,
            show_colorbar=show_colorbar,
        )

    def render_scarlet_field_model_comparison(
        self,
        observation: Observation,
        *,
        model_image_name: str = "scarlet_field_model",
        intrinsic_image_name: str = "scarlet_field_model_intrinsic",
        title: str | None = None,
        show_colorbars: bool = True,
    ) -> None:
        """Render the principal SCARLET field products as a thesis-ready plate.

        The figure is presentation-only. Panels A and B show the stored
        observation-space and intrinsic SCARLET products using independent
        robust display stretches. Panel C is retained as a *display-only*
        visual difference; it must not be interpreted as a scientific residual
        because the two products may not share the same model frame.

        The active LocalField controls the spatial extent when available.
        No stored SCARLET product is modified.
        """
        model_data, model_label = self._get_image(observation, model_image_name)
        intrinsic_data, intrinsic_label = self._get_image(
            observation, intrinsic_image_name
        )
        model = np.asarray(model_data.image.data, dtype=np.float64)
        intrinsic = np.asarray(intrinsic_data.image.data, dtype=np.float64)
        self._validate_image(model, model_label)
        self._validate_image(intrinsic, intrinsic_label)

        if model.shape != intrinsic.shape:
            raise ValueError(
                "SCARLET model comparison requires matching image shapes: "
                f"{model.shape} versus {intrinsic.shape}."
            )

        local_field = self._active_local_field
        radius = self._active_local_field_view_radius(observation)
        if local_field is not None and radius is not None:
            target = local_field.target_source
            centre_x = float(target.x_centroid)
            centre_y = float(target.y_centroid)
            height, width = model.shape
            x0 = max(0, int(np.floor(centre_x - radius)))
            x1 = min(width, int(np.ceil(centre_x + radius)) + 1)
            y0 = max(0, int(np.floor(centre_y - radius)))
            y1 = min(height, int(np.ceil(centre_y + radius)) + 1)
        else:
            x0, y0 = 0, 0
            y1, x1 = model.shape

        model_crop = self._prepare_scarlet_display_array(model[y0:y1, x0:x1])
        intrinsic_crop = self._prepare_scarlet_display_array(
            intrinsic[y0:y1, x0:x1]
        )
        difference_crop = intrinsic_crop - model_crop

        model_norm = self._create_scarlet_asinh_normalisation(model_crop)
        intrinsic_norm = self._create_scarlet_asinh_normalisation(intrinsic_crop)

        finite_difference = difference_crop[np.isfinite(difference_crop)]
        difference_limit = (
            float(np.nanpercentile(np.abs(finite_difference), 99.5))
            if finite_difference.size
            else 0.0
        )
        if not isfinite(difference_limit) or difference_limit <= 0.0:
            difference_limit = 1.0
        difference_norm = Normalize(vmin=-difference_limit, vmax=difference_limit)

        if title is None:
            title = (
                f"{observation.provenance.survey.value} • "
                f"{observation.metadata.target_name}\n"
                "SCARLET Field Model Comparison"
            )

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(20.0, 7.0),
            squeeze=False,
            constrained_layout=False,
        )

        try:
            flat_axes = list(axes.ravel())
            panels = (
                (
                    flat_axes[0], model_crop, model_norm,
                    "Observation-Space Model", "SCARLET model intensity", "gray",
                    "A",
                ),
                (
                    flat_axes[1], intrinsic_crop, intrinsic_norm,
                    "Intrinsic Model (Model Frame)", "Intrinsic model intensity", "gray",
                    "B",
                ),
                (
                    flat_axes[2], difference_crop, difference_norm,
                    "Display-Only Comparison", "Intrinsic − observation-space", "RdBu_r",
                    "C",
                ),
            )

            for axis, image, normalisation, panel_title, colourbar_label, cmap, panel_label in panels:
                rendered = axis.imshow(
                    image,
                    origin="lower",
                    cmap=cmap,
                    interpolation="nearest",
                    norm=normalisation,
                    extent=(x0, x1 - 1, y0, y1 - 1),
                )
                axis.set_title(panel_title, fontsize=12.5, fontweight="bold", pad=8)
                axis.set_xlabel("Pixel X", fontsize=9.5)
                axis.set_ylabel("Pixel Y", fontsize=9.5)
                axis.set_aspect("equal", adjustable="box")
                axis.tick_params(labelsize=8.5)

                # Panel identifier is deliberately inside each axes, avoiding
                # any figure-level legend/footer collision.
                axis.text(
                    0.025, 0.965, panel_label,
                    transform=axis.transAxes,
                    ha="left", va="top",
                    fontsize=11, fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.22",
                        facecolor="black", edgecolor="white",
                        linewidth=0.7, alpha=0.72,
                    ),
                    zorder=20,
                )

                if show_colorbars:
                    cbar = figure.colorbar(
                        rendered,
                        ax=axis,
                        fraction=0.046,
                        pad=0.035,
                        aspect=28,
                    )
                    cbar.set_label(colourbar_label, fontsize=8.5)
                    cbar.ax.tick_params(labelsize=7.5)

            # The caveat belongs to Panel C, not the figure footer. This keeps
            # the x-axis labels completely unobstructed and makes the meaning
            # of the comparison immediately visible to a thesis reader.
            flat_axes[2].text(
                0.025, 0.035,
                "Display-only visual comparison\n"
                "not a scientific residual*",
                transform=flat_axes[2].transAxes,
                ha="left", va="bottom",
                fontsize=8.5, color="0.10",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white", edgecolor="0.35",
                    linewidth=0.7, alpha=0.88,
                ),
                zorder=20,
            )

            figure.suptitle(
                title,
                fontsize=17,
                fontweight="bold",
                y=0.975,
            )
            figure.subplots_adjust(
                left=0.035, right=0.975,
                bottom=0.105, top=0.875,
                wspace=0.28,
            )

            self._logger.info(
                "Rendered SCARLET field model comparison over extent "
                "x=[%d,%d], y=[%d,%d].",
                x0, x1 - 1, y0, y1 - 1,
            )
            plt.show()
        finally:
            plt.close(figure)


    def render_scarlet_field_residual(
        self,
        observation: Observation,
        *,
        image_name: str = "scarlet_field_residual",
        title: str | None = None,
        show_colorbar: bool = True,
    ) -> None:
        """Render the stored SCARLET observed-minus-model residual."""
        image_data, image_label = self._get_image(observation, image_name)
        image = np.asarray(image_data.image.data)
        self._validate_image(image, image_label)

        if title is None:
            title = (
                f"{observation.provenance.survey.value} • "
                f"{observation.metadata.target_name}\n"
                "SCARLET FIELD Residual (Observed − Model)"
            )

        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError(
                f"Image '{image_label}' contains no finite residual values."
            )

        # Residuals are centred around zero. A symmetric robust percentile
        # keeps faint residual structure visible without allowing a few
        # extreme pixels to dominate the display.
        absolute_values = np.abs(finite)
        display_limit = float(np.nanpercentile(absolute_values, 99.5))
        if not isfinite(display_limit) or display_limit <= 0.0:
            display_limit = float(np.max(absolute_values))
        if not isfinite(display_limit) or display_limit <= 0.0:
            display_limit = 1.0

        normalisation = Normalize(vmin=-display_limit, vmax=display_limit)

        self._logger.info(
            "Rendering SCARLET residual product '%s'.",
            image_label,
        )

        figure, axis = plt.subplots(figsize=(10, 9))
        try:
            rendered = axis.imshow(
                image,
                origin="lower",
                cmap="RdBu_r",
                interpolation="nearest",
                norm=normalisation,
            )
            self._apply_active_local_field_view(axis, observation, image.shape)
            self._overlay_active_local_field(axis, observation, image.shape)
            axis.set_title(title, fontsize=14, fontweight="bold")
            axis.set_xlabel("Pixel X")
            axis.set_ylabel("Pixel Y")

            if show_colorbar:
                figure.colorbar(
                    rendered,
                    ax=axis,
                    label="Residual (Observed − Model)",
                )

            figure.tight_layout()
            plt.show()
        finally:
            plt.close(figure)

    def render_scarlet_deblended_local_field(
        self,
        observation: Observation,
        *,
        title: str | None = None,
        show_colorbar: bool = True,
        include_target_only: bool = True,
    ) -> None:
        """Render a clean target-centred SCARLET deblended LocalField.

        Only already-stored SCARLET source products are combined for display.
        No fitting, source selection, photometry, or image modification is
        performed here.

        Important presentation detail: SCARLET source products commonly use
        non-finite pixels outside their model support. Those values are
        replaced by zero *only in the temporary display arrays* so Matplotlib
        does not render transparent/NaN pixels as a checkerboard pattern.
        """
        local_field = self._active_local_field
        if local_field is None or self._active_local_field_observation_id != id(observation):
            raise ValueError(
                "SCARLET deblended LocalField visualisation requires an active "
                "LocalField for this Observation. Render the LocalField first."
            )

        images = getattr(observation, "images", {})
        source_products = sorted(
            name for name in images
            if isinstance(name, str) and name.startswith("scarlet_source_")
        )
        if not source_products:
            self._logger.warning(
                "No individual SCARLET source products are available for the "
                "deblended LocalField visualisation."
            )
            return

        selected_sources = (local_field.target_source, *local_field.neighbour_sources)

        component_images: list[tuple[str, NDArray[np.float64], float, float]] = []
        for name in source_products:
            image_data = images[name]
            nddata = getattr(image_data, "image", None)
            if nddata is None:
                continue
            image = np.asarray(nddata.data, dtype=np.float64)
            self._validate_image(image, name)
            metadata = getattr(nddata, "meta", None) or {}
            try:
                centre_x = float(metadata["scarlet_source_center_x"])
                centre_y = float(metadata["scarlet_source_center_y"])
            except (KeyError, TypeError, ValueError):
                self._logger.warning(
                    "SCARLET source product '%s' has no valid source-centre metadata; skipping.",
                    name,
                )
                continue
            if isfinite(centre_x) and isfinite(centre_y):
                component_images.append((name, image, centre_x, centre_y))

        if not component_images:
            self._logger.warning("No valid individual SCARLET source products could be rendered.")
            return

        matched_components: list[tuple[object, NDArray[np.float64], float, float]] = []
        used_components: set[int] = set()
        for source in selected_sources:
            source_x = float(source.x_centroid)
            source_y = float(source.y_centroid)
            candidates = [
                (index, item)
                for index, item in enumerate(component_images)
                if index not in used_components
            ]
            if not candidates:
                break
            selected_index, selected_item = min(
                candidates,
                key=lambda indexed: (
                    indexed[1][2] - source_x
                ) ** 2 + (indexed[1][3] - source_y) ** 2,
            )
            used_components.add(selected_index)
            matched_components.append(
                (source, selected_item[1], selected_item[2], selected_item[3])
            )

        if not matched_components:
            self._logger.warning(
                "Could not match LocalField sources to stored SCARLET source products."
            )
            return

        reference_shape = matched_components[0][1].shape
        field_model = np.zeros(reference_shape, dtype=np.float64)
        target_model: NDArray[np.float64] | None = None
        target_identifier = str(local_field.target_source.identifier)
        target_fallback: tuple[float, NDArray[np.float64]] | None = None

        for source, component, component_x, component_y in matched_components:
            if component.shape != reference_shape:
                self._logger.warning(
                    "Skipping SCARLET component with shape %s; expected %s.",
                    component.shape,
                    reference_shape,
                )
                continue

            display_component = self._prepare_scarlet_display_array(component)
            field_model += display_component

            distance_sq = (
                component_x - float(local_field.target_source.x_centroid)
            ) ** 2 + (
                component_y - float(local_field.target_source.y_centroid)
            ) ** 2
            if target_fallback is None or distance_sq < target_fallback[0]:
                target_fallback = (distance_sq, display_component)

            if str(getattr(source, "identifier", "")) == target_identifier:
                target_model = display_component

        if target_model is None and target_fallback is not None:
            target_model = target_fallback[1]
        if target_model is None:
            self._logger.warning("Could not identify an isolated SCARLET target component.")
            return

        source_xs = [float(component_x) for _, _, component_x, _ in matched_components]
        source_ys = [float(component_y) for _, _, _, component_y in matched_components]
        source_xs.append(float(local_field.target_source.x_centroid))
        source_ys.append(float(local_field.target_source.y_centroid))

        span_x = max(source_xs) - min(source_xs)
        span_y = max(source_ys) - min(source_ys)
        margin = max(28.0, min(50.0, 0.30 * max(span_x, span_y, 1.0)))

        height, width = field_model.shape
        x0 = max(0, int(np.floor(min(source_xs) - margin)))
        x1 = min(width, int(np.ceil(max(source_xs) + margin)) + 1)
        y0 = max(0, int(np.floor(min(source_ys) - margin)))
        y1 = min(height, int(np.ceil(max(source_ys) + margin)) + 1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("SCARLET deblended LocalField crop is invalid.")

        field_crop = self._prepare_scarlet_display_array(field_model[y0:y1, x0:x1])
        target_crop = self._prepare_scarlet_display_array(target_model[y0:y1, x0:x1])
        if not np.any(np.isfinite(field_crop)) or not np.any(np.isfinite(target_crop)):
            raise ValueError("SCARLET deblended LocalField contains no finite image data.")

        field_norm = self._create_scarlet_asinh_normalisation(field_crop)
        target_norm = self._create_scarlet_asinh_normalisation(target_crop)

        if title is None:
            title = (
                f"{observation.provenance.survey.value} • "
                f"{observation.metadata.target_name}\n"
                "SCARLET Deblended LocalField"
            )

        panel_count = 2 if include_target_only else 1
        figure, axes = plt.subplots(
            1,
            panel_count,
            figsize=(7.4 * panel_count, 6.8),
            squeeze=False,
        )
        try:
            flat_axes = list(axes.ravel())
            rendered = flat_axes[0].imshow(
                field_crop,
                origin="lower",
                cmap="gray",
                interpolation="nearest",
                norm=field_norm,
                extent=(x0, x1 - 1, y0, y1 - 1),
            )
            flat_axes[0].set_title("Deblended LocalField", fontsize=13, fontweight="bold")
            flat_axes[0].set_xlabel("Pixel X")
            flat_axes[0].set_ylabel("Pixel Y")
            flat_axes[0].set_aspect("equal", adjustable="box")
            if show_colorbar:
                figure.colorbar(
                    rendered, ax=flat_axes[0], label="SCARLET model intensity",
                    fraction=0.046, pad=0.04,
                )

            if include_target_only:
                target_rendered = flat_axes[1].imshow(
                    target_crop,
                    origin="lower",
                    cmap="gray",
                    interpolation="nearest",
                    norm=target_norm,
                    extent=(x0, x1 - 1, y0, y1 - 1),
                )
                flat_axes[1].set_title("Isolated Target Component", fontsize=13, fontweight="bold")
                flat_axes[1].set_xlabel("Pixel X")
                flat_axes[1].set_ylabel("Pixel Y")
                flat_axes[1].set_aspect("equal", adjustable="box")
                if show_colorbar:
                    figure.colorbar(
                        target_rendered, ax=flat_axes[1], label="Target model intensity",
                        fraction=0.046, pad=0.04,
                    )

            figure.suptitle(title, fontsize=16, fontweight="bold")
            figure.tight_layout(rect=(0, 0, 1, 0.93))
            plt.show()
        finally:
            plt.close(figure)

    def render_baseline_vs_scarlet_deblended_local_field(
        self,
        observation: Observation,
        *,
        image_name: str = "background_removed",
        title: str | None = None,
        show_colorbars: bool = True,
    ) -> None:
        """Render a thesis-ready side-by-side LocalField comparison.

        Panel A shows the observed baseline LocalField using the same pixel
        coordinates as Panel B. Panel B is reconstructed *only* from the
        already-stored SCARLET source products selected by the authoritative
        LocalField. No SCARLET fitting, source selection, photometry, or image
        modification is performed here.

        The two panels therefore represent the same physical detector region:

            A — observed/background-subtracted LocalField
            B — SCARLET model-only deblended LocalField

        Display stretches are independent because the observed image and the
        SCARLET model have different intensity distributions. This is a
        presentation choice only; neither underlying image is modified.
        """
        local_field = self._active_local_field
        if (
            local_field is None
            or self._active_local_field_observation_id != id(observation)
        ):
            raise ValueError(
                "Baseline-versus-SCARLET comparison requires an active "
                "LocalField for this Observation. Render or activate the "
                "LocalField first."
            )

        baseline_data, baseline_label = self._get_image(
            observation,
            image_name,
        )
        baseline = np.asarray(baseline_data.image.data, dtype=np.float64)
        self._validate_image(baseline, baseline_label)

        images = getattr(observation, "images", {})
        source_products = sorted(
            name
            for name in images
            if isinstance(name, str) and name.startswith("scarlet_source_")
        )
        if not source_products:
            self._logger.warning(
                "Baseline-versus-SCARLET comparison skipped: no individual "
                "SCARLET source products are available."
            )
            return

        selected_sources = (
            local_field.target_source,
            *local_field.neighbour_sources,
        )

        component_images: list[tuple[str, NDArray[np.float64], float, float]] = []
        for name in source_products:
            image_data = images[name]
            nddata = getattr(image_data, "image", None)
            if nddata is None:
                continue

            component = np.asarray(nddata.data, dtype=np.float64)
            self._validate_image(component, name)
            metadata = getattr(nddata, "meta", None) or {}
            try:
                centre_x = float(metadata["scarlet_source_center_x"])
                centre_y = float(metadata["scarlet_source_center_y"])
            except (KeyError, TypeError, ValueError):
                continue

            if isfinite(centre_x) and isfinite(centre_y):
                component_images.append(
                    (name, component, centre_x, centre_y)
                )

        if not component_images:
            self._logger.warning(
                "Baseline-versus-SCARLET comparison skipped: no valid "
                "SCARLET source products could be matched."
            )
            return

        matched_components: list[
            tuple[object, NDArray[np.float64], float, float]
        ] = []
        used_components: set[int] = set()

        for source in selected_sources:
            source_x = float(source.x_centroid)
            source_y = float(source.y_centroid)
            candidates = [
                (index, item)
                for index, item in enumerate(component_images)
                if index not in used_components
            ]
            if not candidates:
                break

            selected_index, selected_item = min(
                candidates,
                key=lambda indexed: (
                    indexed[1][2] - source_x
                ) ** 2 + (
                    indexed[1][3] - source_y
                ) ** 2,
            )
            used_components.add(selected_index)
            matched_components.append(
                (
                    source,
                    selected_item[1],
                    selected_item[2],
                    selected_item[3],
                )
            )

        if not matched_components:
            self._logger.warning(
                "Baseline-versus-SCARLET comparison skipped: LocalField "
                "sources could not be matched to SCARLET products."
            )
            return

        reference_shape = matched_components[0][1].shape
        if baseline.shape != reference_shape:
            raise ValueError(
                "Baseline and SCARLET source products must share the same "
                "detector frame shape: "
                f"{baseline.shape} versus {reference_shape}."
            )

        field_model = np.zeros(reference_shape, dtype=np.float64)
        for _, component, _, _ in matched_components:
            if component.shape != reference_shape:
                continue
            field_model += self._prepare_scarlet_display_array(component)

        source_xs = [
            float(component_x)
            for _, _, component_x, _ in matched_components
        ]
        source_ys = [
            float(component_y)
            for _, _, _, component_y in matched_components
        ]
        source_xs.append(float(local_field.target_source.x_centroid))
        source_ys.append(float(local_field.target_source.y_centroid))

        span_x = max(source_xs) - min(source_xs)
        span_y = max(source_ys) - min(source_ys)
        margin = max(
            28.0,
            min(50.0, 0.30 * max(span_x, span_y, 1.0)),
        )

        height, width = field_model.shape
        x0 = max(0, int(np.floor(min(source_xs) - margin)))
        x1 = min(width, int(np.ceil(max(source_xs) + margin)) + 1)
        y0 = max(0, int(np.floor(min(source_ys) - margin)))
        y1 = min(height, int(np.ceil(max(source_ys) + margin)) + 1)

        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                "Baseline-versus-SCARLET LocalField crop is invalid."
            )

        baseline_crop = baseline[y0:y1, x0:x1]
        scarlet_crop = self._prepare_scarlet_display_array(
            field_model[y0:y1, x0:x1]
        )

        self._validate_image(baseline_crop, "baseline LocalField crop")
        self._validate_image(scarlet_crop, "SCARLET LocalField crop")

        baseline_norm = self._create_normalisation(
            cast(NDArray[np.number], baseline_crop),
            baseline_label,
        )
        scarlet_norm = self._create_scarlet_asinh_normalisation(
            scarlet_crop,
        )

        if title is None:
            title = (
                f"{observation.provenance.survey.value} • "
                f"{observation.metadata.target_name}\n"
                "Baseline vs SCARLET Deblended LocalField"
            )

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(15.5, 7.0),
            squeeze=False,
            constrained_layout=False,
        )

        try:
            flat_axes = list(axes.ravel())
            panels = (
                (
                    flat_axes[0],
                    baseline_crop,
                    baseline_norm,
                    "A  Baseline LocalField (Observed)",
                    "Normalised intensity",
                ),
                (
                    flat_axes[1],
                    scarlet_crop,
                    scarlet_norm,
                    "B  SCARLET Deblended LocalField (Model)",
                    "SCARLET model intensity",
                ),
            )

            for axis, image, norm, panel_title, colorbar_label in panels:
                rendered = axis.imshow(
                    image,
                    origin="lower",
                    cmap="gray",
                    interpolation="nearest",
                    norm=norm,
                    extent=(x0, x1 - 1, y0, y1 - 1),
                )
                axis.set_title(
                    panel_title,
                    fontsize=13,
                    fontweight="bold",
                    pad=9,
                )
                axis.set_xlabel("Pixel X")
                axis.set_ylabel("Pixel Y")
                axis.set_aspect("equal", adjustable="box")

                # Mark only the sources actually represented by the
                # SCARLET LocalField reconstruction. Detections outside this
                # LocalField are not part of the model shown in Panel B and
                # should not be marked as though they were deblended here.
                # Coordinates come directly from the matched SCARLET source
                # products. This overlay is presentation-only.
                if axis is flat_axes[0]:
                    for _, _, component_x, component_y in matched_components:
                        marker = Circle(
                            (float(component_x), float(component_y)),
                            radius=0.65,
                            fill=False,
                            linestyle="-",
                            linewidth=0.8,
                            edgecolor="red",
                            alpha=0.65,
                            zorder=6,
                            clip_on=True,
                        )
                        axis.add_patch(marker)

                    # Show the authoritative LocalField boundary on the
                    # observed baseline panel. This is also presentation-only:
                    # the radius comes directly from the established
                    # LocalField and does not alter the image or SCARLET model.
                    local_field_radius = float(
                        local_field.selection_radius_pixels
                    )
                    target_x = float(local_field.target_source.x_centroid)
                    target_y = float(local_field.target_source.y_centroid)
                    if (
                        isfinite(local_field_radius)
                        and local_field_radius > 0.0
                        and isfinite(target_x)
                        and isfinite(target_y)
                    ):
                        boundary = Circle(
                            (target_x, target_y),
                            local_field_radius,
                            fill=False,
                            linestyle=":",
                            linewidth=1.8,
                            edgecolor="yellow",
                            alpha=0.95,
                            zorder=5,
                            clip_on=True,
                        )
                        axis.add_patch(boundary)

                # Keep both image panels locked to the same detector crop.
                # In particular, the LocalField boundary is allowed to extend
                # beyond the image but must not expand the baseline axes; this
                # keeps the observed image filling the same square visual area
                # as the SCARLET panel.
                axis.set_xlim(x0, x1 - 1)
                axis.set_ylim(y0, y1 - 1)

                if show_colorbars:
                    figure.colorbar(
                        rendered,
                        ax=axis,
                        label=colorbar_label,
                        fraction=0.046,
                        pad=0.04,
                    )

            figure.suptitle(
                title,
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )
            modelled_count = len(matched_components)
            figure.text(
                0.5,
                0.015,
                "Same detector coordinates and LocalField crop • "
                "red outlines = SCARLET-modelled source positions "
                f"({modelled_count}) • "
                "yellow dotted circle = LocalField boundary • "
                "display stretches are independent • SCARLET panel is model-only",
                ha="center",
                va="bottom",
                fontsize=9,
                alpha=0.75,
            )
            figure.tight_layout(rect=(0, 0.045, 1, 0.94))
            plt.show()
        finally:
            plt.close(figure)

    @staticmethod
    def _prepare_scarlet_display_array(
        image: NDArray[np.number],
    ) -> NDArray[np.float64]:
        """Return a display-only finite SCARLET image array.

        SCARLET products can legitimately contain NaN/non-finite pixels where
        a component has no model support. Passing those values directly to
        ``imshow`` can expose Matplotlib's transparent/axes background as a
        checkerboard pattern. For presentation only, unsupported pixels are
        rendered as zero intensity. The original Observation image is never
        changed.
        """
        values = np.asarray(image, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("SCARLET display image must be two-dimensional.")
        return np.where(np.isfinite(values), values, 0.0)

    @staticmethod
    def _create_scarlet_asinh_normalisation(
        image: NDArray[np.number],
    ) -> Normalize:
        """Create a robust positive SCARLET display normalisation."""
        values = np.asarray(image, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return cast(
                Normalize,
                ImageNormalize(
                    vmin=0.0,
                    vmax=1.0,
                    stretch=AsinhStretch(a=0.02),
                ),
            )

        positive = finite[finite > 0.0]
        if positive.size == 0:
            positive = np.abs(finite)
        if positive.size == 0 or not np.any(np.isfinite(positive)):
            return cast(
                Normalize,
                ImageNormalize(
                    vmin=0.0,
                    vmax=1.0,
                    stretch=AsinhStretch(a=0.02),
                ),
            )

        high = float(np.nanpercentile(positive, 99.7))
        if not isfinite(high) or high <= 0.0:
            high = float(np.nanmax(positive))
        if not isfinite(high) or high <= 0.0:
            high = 1.0

        return cast(
            Normalize,
            ImageNormalize(
                vmin=0.0,
                vmax=high,
                stretch=AsinhStretch(a=0.02),
                clip=True,
            ),
        )

    def render_scarlet_source_models(
        self,
        observation: Observation,
        *,
        source_names: Iterable[str] | None = None,
        max_sources: int = 9,
        crop_radius: int = 25,
        columns: int = 3,
        title: str | None = None,
        show_colorbar: bool = False,
    ) -> None:
        """
        Render a compact montage of individual SCARLET source models.

        SCARLET stores each source as a full-frame Observation image. This
        method crops those products around the source centre recorded in
        their metadata, avoiding dozens of large full-field figures.
        """
        if max_sources <= 0:
            raise ValueError("max_sources must be positive.")
        if crop_radius <= 0:
            raise ValueError("crop_radius must be positive.")
        if columns <= 0:
            raise ValueError("columns must be positive.")

        images = getattr(observation, "images", {})
        if source_names is None:
            candidates = sorted(
                name for name in images
                if isinstance(name, str)
                and name.startswith("scarlet_source_")
            )
        else:
            candidates = list(source_names)

        if not candidates:
            self._logger.warning(
                "No SCARLET individual-source products are available."
            )
            return

        candidates = candidates[:max_sources]
        panels: list[tuple[str, NDArray[np.float64]]] = []

        for name in candidates:
            if name not in images:
                self._logger.warning(
                    "SCARLET source product '%s' is unavailable; skipping.",
                    name,
                )
                continue

            image_data = images[name]
            nddata = getattr(image_data, "image", None)
            if nddata is None:
                self._logger.warning(
                    "SCARLET source product '%s' contains no image data; skipping.",
                    name,
                )
                continue

            image = np.asarray(nddata.data)
            self._validate_image(image, name)
            metadata = getattr(nddata, "meta", None) or {}

            try:
                centre_y = float(metadata["scarlet_source_center_y"])
                centre_x = float(metadata["scarlet_source_center_x"])
            except (KeyError, TypeError, ValueError):
                self._logger.warning(
                    "SCARLET source product '%s' has no valid source-centre "
                    "metadata; skipping.",
                    name,
                )
                continue

            if not isfinite(centre_x) or not isfinite(centre_y):
                self._logger.warning(
                    "SCARLET source product '%s' has non-finite centre metadata; skipping.",
                    name,
                )
                continue

            height, width = image.shape
            x = int(round(centre_x))
            y = int(round(centre_y))

            x0 = max(0, x - crop_radius)
            x1 = min(width, x + crop_radius + 1)
            y0 = max(0, y - crop_radius)
            y1 = min(height, y + crop_radius + 1)

            crop = self._prepare_scarlet_display_array(
                np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
            )
            if crop.size == 0 or not np.any(np.isfinite(crop)):
                self._logger.warning(
                    "SCARLET source product '%s' has an invalid cutout; skipping.",
                    name,
                )
                continue

            panels.append((name, crop))

        if not panels:
            self._logger.warning(
                "SCARLET source-model visualisation produced no valid cutouts."
            )
            return

        rows = int(np.ceil(len(panels) / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.0 * columns, 4.0 * rows),
            squeeze=False,
        )

        try:
            flat_axes = list(axes.ravel())

            for axis, (name, crop) in zip(flat_axes, panels):
                normalisation = self._create_scarlet_asinh_normalisation(crop)
                axis.imshow(
                    crop,
                    origin="lower",
                    cmap="gray",
                    interpolation="nearest",
                    norm=normalisation,
                )
                axis.set_title(name, fontsize=10, fontweight="bold")
                axis.set_xlabel("Local X")
                axis.set_ylabel("Local Y")

            for axis in flat_axes[len(panels):]:
                axis.axis("off")

            if title is None:
                title = (
                    f"{observation.provenance.survey.value} • "
                    f"{observation.metadata.target_name}\n"
                    "SCARLET FIELD Individual Source Models"
                )

            figure.suptitle(title, fontsize=15, fontweight="bold")
            figure.tight_layout(rect=(0, 0, 1, 0.95))

            self._logger.info(
                "Rendered %d SCARLET individual-source cutout(s) "
                "from %d requested product(s).",
                len(panels),
                len(candidates),
            )

            plt.show()
        finally:
            plt.close(figure)

    def _render_scarlet_product(
        self,
        observation: Observation,
        *,
        image_name: str,
        title: str | None,
        show_colorbar: bool,
    ) -> None:
        """Render one existing SCARLET image product without processing it."""
        image_data, image_label = self._get_image(observation, image_name)
        image = np.asarray(image_data.image.data)
        self._validate_image(image, image_label)

        if title is None:
            product_titles = {
                "scarlet_field_model": "SCARLET FIELD Reconstruction (Observation Space)",
                "scarlet_field_model_intrinsic": "SCARLET FIELD Intrinsic Model",
            }
            product_title = product_titles.get(
                image_name,
                image_name.replace("_", " ").title(),
            )
            title = (
                f"{observation.provenance.survey.value} • "
                f"{observation.metadata.target_name}\n"
                f"{product_title}"
            )

        display_image = self._prepare_scarlet_display_array(
            np.asarray(image, dtype=np.float64)
        )
        normalisation = self._create_scarlet_normalisation(
            cast(NDArray[np.number], display_image),
            observation,
            image_label,
        )

        self._logger.info(
            "Rendering SCARLET product '%s'.",
            image_label,
        )

        figure, axis = plt.subplots(figsize=(10, 9))
        try:
            rendered = axis.imshow(
                image,
                origin="lower",
                cmap="gray",
                interpolation="nearest",
                norm=normalisation,
            )
            self._apply_active_local_field_view(
                axis,
                observation,
                display_image.shape,
            )
            self._overlay_active_local_field(
                axis,
                observation,
                image.shape,
            )
            axis.set_title(title, fontsize=14, fontweight="bold")
            axis.set_xlabel("Pixel X")
            axis.set_ylabel("Pixel Y")

            if show_colorbar:
                figure.colorbar(
                    rendered,
                    ax=axis,
                    label="Normalised Intensity",
                )

            figure.tight_layout()
            plt.show()
        finally:
            plt.close(figure)

    def render_crowded_sources(
        self,
        observation: Observation,
        *,
        coordinates: tuple[CrowdedSourceCoordinate, ...],
        differential_coordinates: tuple[CrowdedSourceCoordinate, ...] = (),
        image_name: str | None = None,
        title: str | None = None,
        show_colorbar: bool = True,
        annotate_coordinates: bool = True,
    ) -> None:
        """
        Render crowded-source candidates on the observation image.

        This is a diagnostic visualisation only. The supplied sky
        coordinates must already have been produced by
        ``CrowdedSourceCoordinates``. No WCS transformation, source
        detection or crowding classification is performed here.

        Parameters
        ----------
        observation
            Completed Observation containing the image on which the
            crowded-source candidates were detected.

        coordinates
            Immutable crowded-source coordinate records containing the
            source identifier, detector pixel position and converted
            RA/Dec values.

        differential_coordinates
            Optional coordinate records identifying isolated/low-crowding
            candidates for differential photometry. These are rendered in
            the same field-analysis figure using a distinct marker and
            colour. They are presentation-only and are not re-ranked or
            otherwise interpreted by this method.

        image_name
            Name of the Observation image to display. If omitted, the
            active image is displayed.

        title
            Optional figure title.

        show_colorbar
            Display the image colour scale.

        annotate_coordinates
            Display the RA/Dec values next to each crowded source.

        Raises
        ------
        ValueError
            If the Observation, coordinate records or image data are
            invalid.
        """

        if coordinates is None:
            raise ValueError(
                "Crowded-source coordinates cannot be None.",
            )

        if differential_coordinates is None:
            raise ValueError(
                "Differential-photometry coordinates cannot be None.",
            )

        image_data, image_label = self._get_image(
            observation,
            image_name,
        )

        image = np.asarray(
            image_data.image.data,
        )

        self._validate_image(
            image,
            image_label,
        )

        for coordinate in coordinates:
            self._validate_crowded_source_coordinate(
                coordinate,
            )

        for coordinate in differential_coordinates:
            self._validate_crowded_source_coordinate(
                coordinate,
            )

        # Representative observations have an authoritative LocalField. In
        # that workflow the crowded-source diagnostic is intentionally scoped
        # to the analytical LocalField rather than showing the whole detector.
        # FIELD_ANALYSIS does not activate a LocalField, so its detector-wide
        # crowded-source rendering remains unchanged.
        display_coordinates = coordinates
        display_differential_coordinates = differential_coordinates
        local_field = self._active_local_field
        if (
            local_field is not None
            and self._active_local_field_observation_id == id(observation)
        ):
            target = local_field.target_source
            centre_x = float(target.x_centroid)
            centre_y = float(target.y_centroid)
            selection_radius = float(local_field.selection_radius_pixels)
            if (
                isfinite(centre_x)
                and isfinite(centre_y)
                and isfinite(selection_radius)
                and selection_radius > 0.0
            ):
                radius_sq = selection_radius ** 2
                display_coordinates = tuple(
                    coordinate
                    for coordinate in coordinates
                    if (
                        (float(coordinate.x_pixel) - centre_x) ** 2
                        + (float(coordinate.y_pixel) - centre_y) ** 2
                        <= radius_sq
                    )
                )
                display_differential_coordinates = tuple(
                    coordinate
                    for coordinate in differential_coordinates
                    if (
                        (float(coordinate.x_pixel) - centre_x) ** 2
                        + (float(coordinate.y_pixel) - centre_y) ** 2
                        <= radius_sq
                    )
                )

        if title is None:
            scope_label = (
                "LocalField"
                if (
                    self._active_local_field is not None
                    and self._active_local_field_observation_id == id(observation)
                )
                else "Full Field"
            )
            title = (
                f"{observation.provenance.survey.value} • "
                f"{observation.metadata.target_name}\n"
                f"Crowded Sources • {scope_label} ({len(display_coordinates)})"
            )

        self._logger.info(
            (
                "Rendering %d crowded-source and %d differential-photometry "
                "candidate coordinate(s) on image '%s'."
            ),
            len(display_coordinates),
            len(display_differential_coordinates),
            image_label,
        )

        normalisation = self._create_normalisation(
            cast(
                NDArray[np.number],
                image,
            ),
            image_label,
        )

        figure, axis = plt.subplots(
            figsize=(10, 9),
        )

        try:
            rendered = axis.imshow(
                image,
                origin="lower",
                cmap="gray",
                interpolation="nearest",
                norm=normalisation,
            )

            # The crowded-source diagnostic uses the same authoritative
            # LocalField presentation context as the LocalField diagnostic.
            # The coordinate list above is filtered to the LocalField; the
            # image must also be viewed at the same target-centred extent.
            # FIELD_ANALYSIS has no active LocalField, so this is a no-op
            # there and preserves its detector-wide presentation.
            self._apply_active_local_field_view(
                axis,
                observation,
                image.shape,
            )

            if display_coordinates:
                x_coordinates = [
                    coordinate.x_pixel
                    for coordinate in display_coordinates
                ]
                y_coordinates = [
                    coordinate.y_pixel
                    for coordinate in display_coordinates
                ]

                axis.scatter(
                    x_coordinates,
                    y_coordinates,
                    s=130,
                    marker="o",
                    facecolors="none",
                    edgecolors="cyan",
                    linewidths=1.8,
                    label=(
                        f"Crowded Sources ({len(display_coordinates)})"
                    ),
                    zorder=10,
                )

                for coordinate in display_coordinates:
                    annotation = coordinate.source_identifier

                    if annotate_coordinates:
                        annotation += (
                            "\n"
                            f"RA={coordinate.right_ascension:.6f}°\n"
                            f"Dec={coordinate.declination:.6f}°"
                        )

                    axis.annotate(
                        annotation,
                        xy=(
                            coordinate.x_pixel,
                            coordinate.y_pixel,
                        ),
                        xytext=(
                            8,
                            8,
                        ),
                        textcoords="offset points",
                        fontsize=8,
                        ha="left",
                        va="bottom",
                        color="white",
                        bbox={
                            "boxstyle": "round,pad=0.25",
                            "facecolor": "black",
                            "alpha": 0.70,
                            "edgecolor": "none",
                        },
                        zorder=11,
                    )

            # Differential-photometry candidates are deliberately shown in
            # the same field-analysis figure rather than as a separate
            # visualisation. A star marker distinguishes these isolated/
            # low-crowding candidates from the cyan crowded-source circles.
            # No ranking or scientific selection is performed here.
            if display_differential_coordinates:
                x_coordinates = [
                    coordinate.x_pixel
                    for coordinate in display_differential_coordinates
                ]
                y_coordinates = [
                    coordinate.y_pixel
                    for coordinate in display_differential_coordinates
                ]

                axis.scatter(
                    x_coordinates,
                    y_coordinates,
                    s=190,
                    marker="*",
                    facecolors="none",
                    edgecolors="yellow",
                    linewidths=1.5,
                    label=(
                        "Differential Candidates "
                        f"({len(display_differential_coordinates)})"
                    ),
                    zorder=12,
                )

                for coordinate in display_differential_coordinates:
                    annotation = coordinate.source_identifier

                    if annotate_coordinates:
                        annotation += (
                            "\n"
                            f"RA={coordinate.right_ascension:.6f}°\n"
                            f"Dec={coordinate.declination:.6f}°"
                        )

                    axis.annotate(
                        annotation,
                        xy=(
                            coordinate.x_pixel,
                            coordinate.y_pixel,
                        ),
                        xytext=(
                            8,
                            -12,
                        ),
                        textcoords="offset points",
                        fontsize=8,
                        ha="left",
                        va="top",
                        color="black",
                        bbox={
                            "boxstyle": "round,pad=0.25",
                            "facecolor": "yellow",
                            "alpha": 0.85,
                            "edgecolor": "none",
                        },
                        zorder=13,
                    )

            axis.set_title(
                title,
                fontsize=14,
                fontweight="bold",
            )

            axis.set_xlabel(
                "Pixel X",
            )

            axis.set_ylabel(
                "Pixel Y",
            )

            if show_colorbar:
                figure.colorbar(
                    rendered,
                    ax=axis,
                    label="Normalised Intensity",
                )

            self._refresh_legend(
                axis,
            )

            figure.tight_layout()
            plt.show()

        finally:
            plt.close(
                figure,
            )

    def render_crowded_source_table(
        self,
        *,
        coordinates: tuple[CrowdedSourceCoordinate, ...],
        title: str = "Crowded-Source Sky Coordinates",
    ) -> None:
        """
        Render crowded-source detector and sky coordinates as a table.

        This method is presentation-only. It consumes coordinate records
        already converted by ``CrowdedSourceCoordinates`` and performs no
        astronomical calculation.

        Parameters
        ----------
        coordinates
            Immutable crowded-source coordinate records.

        title
            Figure title.

        Raises
        ------
        ValueError
            If the coordinate collection or any record is invalid.
        """

        if coordinates is None:
            raise ValueError(
                "Crowded-source coordinates cannot be None.",
            )

        for coordinate in coordinates:
            self._validate_crowded_source_coordinate(
                coordinate,
            )

        row_count = len(coordinates)
        figure_height = max(
            3.0,
            min(
                18.0,
                1.5 + (0.48 * max(row_count, 1)),
            ),
        )

        figure, axis = plt.subplots(
            figsize=(11, figure_height),
        )

        try:
            axis.axis("off")

            if not coordinates:
                axis.text(
                    0.5,
                    0.5,
                    "No crowded sources identified.",
                    transform=axis.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                )
            else:
                rows = [
                    [
                        coordinate.source_identifier,
                        f"{coordinate.x_pixel:.3f}",
                        f"{coordinate.y_pixel:.3f}",
                        f"{coordinate.right_ascension:.8f}",
                        f"{coordinate.declination:.8f}",
                    ]
                    for coordinate in coordinates
                ]

                table = axis.table(
                    cellText=rows,
                    colLabels=[
                        "Source",
                        "X pixel",
                        "Y pixel",
                        "RA (deg)",
                        "Dec (deg)",
                    ],
                    cellLoc="center",
                    colLoc="center",
                    loc="center",
                    colWidths=[
                        0.18,
                        0.18,
                        0.18,
                        0.23,
                        0.23,
                    ],
                )

                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(
                    1.0,
                    1.5,
                )

            axis.set_title(
                title,
                fontsize=14,
                fontweight="bold",
                pad=12,
            )

            figure.tight_layout()
            plt.show()

        finally:
            plt.close(
                figure,
            )

    @staticmethod
    def _validate_crowded_source_coordinate(
        coordinate: CrowdedSourceCoordinate,
    ) -> None:
        """
        Validate a diagnostic crowded-source coordinate record.

        Validation here protects the presentation layer from producing
        misleading figures. It does not recalculate or alter any value.
        """

        if coordinate is None:
            raise ValueError(
                "Crowded-source coordinate cannot be None.",
            )

        if not isinstance(
            coordinate.source_identifier,
            str,
        ) or not coordinate.source_identifier.strip():
            raise ValueError(
                "Crowded-source coordinate has an empty source identifier.",
            )

        values = (
            ("X pixel", coordinate.x_pixel),
            ("Y pixel", coordinate.y_pixel),
            ("right ascension", coordinate.right_ascension),
            ("declination", coordinate.declination),
        )

        for name, value in values:
            try:
                finite = isfinite(
                    float(value),
                )
            except (TypeError, ValueError):
                finite = False

            if not finite:
                raise ValueError(
                    f"Source '{coordinate.source_identifier}' has a "
                    f"non-finite {name}: {value!r}."
                )

        right_ascension = float(
            coordinate.right_ascension,
        )
        declination = float(
            coordinate.declination,
        )

        if not 0.0 <= right_ascension < 360.0:
            raise ValueError(
                f"Source '{coordinate.source_identifier}' has an invalid "
                f"right ascension: {right_ascension!r} degrees."
            )

        if not -90.0 <= declination <= 90.0:
            raise ValueError(
                f"Source '{coordinate.source_identifier}' has an invalid "
                f"declination: {declination!r} degrees."
            )

    def _validate_render_options(
        self,
        *,
        max_photometry_overlays: int | None,
        max_quality_overlays: int | None,
    ) -> None:
        """
        Validate visualisation options.
        """

        if (
            max_photometry_overlays is not None
            and max_photometry_overlays <= 0
        ):
            raise ValueError(
                "max_photometry_overlays must be positive or None.",
            )

        if (
            max_quality_overlays is not None
            and max_quality_overlays <= 0
        ):
            raise ValueError(
                "max_quality_overlays must be positive or None.",
            )

    def _get_image(
        self,
        observation: Observation,
        image_name: str | None,
    ):
        """
        Retrieve the requested Observation image.
        """

        if image_name is None:
            return (
                observation.active,
                "active",
            )

        try:
            image_data = observation.get_image(
                image_name,
            )

        except KeyError as exc:
            raise ValueError(
                f"Observation does not contain "
                f"image '{image_name}'."
            ) from exc

        return (
            image_data,
            image_name,
        )

    def _validate_image(
        self,
        image: NDArray[np.number],
        image_name: str,
    ) -> None:
        """
        Validate an image before rendering.
        """

        if image.size == 0:
            raise ValueError(
                f"Image '{image_name}' contains no data.",
            )

        if image.ndim != 2:
            raise ValueError(
                f"Image '{image_name}' must be two-dimensional "
                f"for visualisation; received {image.ndim}-D data.",
            )

        if not np.any(
            np.isfinite(image),
        ):
            raise ValueError(
                f"Image '{image_name}' contains no finite values.",
            )

    def _create_title(
        self,
        *,
        observation: Observation,
        image_name: str,
        show_detections: bool,
        show_photometry: bool,
        show_target: bool,
        show_quality: bool,
        evaluation: PhotometryEvaluation | None,
    ) -> str:
        """
        Create an informative figure title.
        """

        survey = observation.provenance.survey.value
        target = observation.metadata.target_name

        stage_names = {
            "raw": "Raw Image",
            "background": "Estimated Background",
            "background_removed": "Background Removed",
            "active": "Active Image",
        }

        if show_target:
            stage = "Target Source Resolution"

        elif show_quality:
            stage = "Photometric Quality"

        elif show_photometry:
            photometry_result = observation.photometry

            measurement_count = (
                0
                if photometry_result is None
                else len(
                    photometry_result.measurements,
                )
            )

            stage = (
                f"Aperture Photometry "
                f"({measurement_count})"
            )

        elif show_detections:
            detection_result = observation.detections

            detection_count = (
                0
                if detection_result is None
                else len(
                    detection_result.sources,
                )
            )

            stage = (
                f"Detected Sources "
                f"({detection_count})"
            )

        else:
            stage = stage_names.get(
                image_name,
                image_name.replace(
                    "_",
                    " ",
                ).title(),
            )

        title = (
            f"{survey} • {target}\n"
            f"{stage}"
        )

        if evaluation is not None:
            title += (
                " • "
                f"Valid {evaluation.valid_measurement_count}/"
                f"{evaluation.detected_source_count}"
            )

        return title

    def _overlay_detections_soft(
        self,
        axis: Axes,
        observation: Observation,
    ) -> None:
        """Overlay detected source positions with a subtle outline.

        This presentation-only overlay is intended for the baseline panel of
        the Baseline-vs-SCARLET LocalField comparison. It makes the source
        catalogue explicit without implying that the observed image itself
        has been altered or that SCARLET must reproduce every visible pixel
        feature.
        """
        detection_result = observation.detections

        if detection_result is None:
            self._logger.debug(
                "Observation contains no source detections for the "
                "baseline-vs-SCARLET overlay.",
            )
            return

        sources = detection_result.sources
        if not sources:
            self._logger.debug(
                "No detected sources available for the baseline-vs-SCARLET "
                "overlay.",
            )
            return

        # Use small, hollow, semi-transparent rings rather than the stronger
        # standalone detection styling. The purpose here is to establish
        # correspondence between the observed baseline and the SCARLET source
        # model without obscuring the underlying stellar profiles.
        axis.scatter(
            [source.x_centroid for source in sources],
            [source.y_centroid for source in sources],
            # Small hollow rings keep the detected positions identifiable
            # while avoiding the visual weight of the standalone detection
            # plot.  No fill is used, so the underlying stellar morphology
            # remains visible.
            s=24,
            facecolors="none",
            edgecolors="red",
            linewidths=0.65,
            alpha=0.55,
            zorder=6,
            clip_on=True,
        )

        self._logger.info(
            "Rendered %d detected sources as a soft baseline overlay for "
            "the SCARLET comparison.",
            len(sources),
        )

    def _overlay_detections(
        self,
        axis: Axes,
        observation: Observation,
    ) -> None:
        """
        Overlay detected source centroids.
        """

        detection_result = observation.detections

        if detection_result is None:
            self._logger.debug(
                "Observation contains no source detections.",
            )
            return

        sources = detection_result.sources

        if not sources:
            self._logger.debug(
                "No detected sources available for rendering.",
            )
            return

        x_coordinates = [
            source.x_centroid
            for source in sources
        ]

        y_coordinates = [
            source.y_centroid
            for source in sources
        ]

        axis.scatter(
            x_coordinates,
            y_coordinates,
            s=60,
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            label=(
                f"Detected Sources "
                f"({len(sources)})"
            ),
        )

        self._logger.info(
            "Rendered %d detected sources.",
            len(sources),
        )

    def _overlay_quality(
        self,
        axis: Axes,
        observation: Observation,
        *,
        evaluation: PhotometryEvaluation | None,
        max_overlays: int | None,
    ) -> None:
        """
        Overlay source-level photometric validity and crowding status.

        Validity and crowding are deliberately represented as independent
        properties because the production evaluator calculates them
        independently.

        Therefore a source may simultaneously be:

            * valid and crowded;
            * invalid and crowded;
            * valid and not crowded;
            * invalid and not crowded.

        This method does not perform scientific classification. It only
        presents the authoritative SourceEvaluation records.
        """
        if evaluation is None:
            self._logger.debug(
                "Source-quality overlay requested without evaluation."
            )
            return

        detection_result = observation.detections

        if detection_result is None:
            self._logger.debug(
                "Source-quality overlay requested without detections."
            )
            return

        evaluations = {
            item.source_identifier: item
            for item in evaluation.source_evaluations
        }

        sources = detection_result.sources

        if max_overlays is not None:
            sources = sources[:max_overlays]

        #
        # Keep validity and crowding separate.
        #
        valid_coordinates: list[tuple[float, float]] = []
        invalid_coordinates: list[tuple[float, float]] = []
        missing_coordinates: list[tuple[float, float]] = []
        crowded_coordinates: list[tuple[float, float]] = []

        for source in sources:
            source_evaluation = evaluations.get(
                source.identifier,
            )

            if source_evaluation is None:
                continue

            coordinate = (
                float(source.x_centroid),
                float(source.y_centroid),
            )

            #
            # Photometric validity.
            #
            if source_evaluation.quality_flag == "MISSING_MEASUREMENT":
                missing_coordinates.append(coordinate)

            elif source_evaluation.valid_measurement:
                valid_coordinates.append(coordinate)

            else:
                invalid_coordinates.append(coordinate)

            #
            # Crowding is an independent property.
            #
            if source_evaluation.crowded:
                crowded_coordinates.append(coordinate)

        #
        # --------------------------------------------------------------
        # Photometric validity
        # --------------------------------------------------------------
        #

        if valid_coordinates:
            axis.scatter(
                [x for x, _ in valid_coordinates],
                [y for _, y in valid_coordinates],
                s=85,
                marker="o",
                facecolors="none",
                edgecolors="lime",
                linewidths=1.6,
                label=f"Valid ({len(valid_coordinates)})",
                zorder=8,
            )

        if invalid_coordinates:
            axis.scatter(
                [x for x, _ in invalid_coordinates],
                [y for _, y in invalid_coordinates],
                s=105,
                marker="x",
                color="red",
                linewidths=2.0,
                label=f"Invalid ({len(invalid_coordinates)})",
                zorder=9,
            )

        if missing_coordinates:
            axis.scatter(
                [x for x, _ in missing_coordinates],
                [y for _, y in missing_coordinates],
                s=90,
                marker="s",
                facecolors="none",
                edgecolors="orange",
                linewidths=1.6,
                label=(
                    f"Missing Measurement "
                    f"({len(missing_coordinates)})"
                ),
                zorder=8,
            )

        #
        # --------------------------------------------------------------
        # Independent crowding overlay
        # --------------------------------------------------------------
        #
        # Crowded sources are drawn as triangles on top of the validity
        # markers. This means a crowded source can visibly retain its
        # valid/invalid state at the same time.
        #

        if crowded_coordinates:
            axis.scatter(
                [x for x, _ in crowded_coordinates],
                [y for _, y in crowded_coordinates],
                s=115,
                marker="^",
                facecolors="none",
                edgecolors="magenta",
                linewidths=1.7,
                label=(
                    f"Crowded "
                    f"({len(crowded_coordinates)})"
                ),
                zorder=10,
            )

        self._logger.info(
            (
                "Rendered source-quality classifications for %d "
                "sources: valid=%d, invalid=%d, missing=%d, "
                "crowded=%d."
            ),
            len(sources),
            len(valid_coordinates),
            len(invalid_coordinates),
            len(missing_coordinates),
            len(crowded_coordinates),
        )

    @staticmethod
    def _quality_category(
        evaluation: SourceEvaluation,
    ) -> str:
        """
        Determine the visual quality category from an existing
        SourceEvaluation.

        This method does not perform scientific classification. It only
        maps the already-computed evaluation state to a presentation
        category.
        """

        if evaluation.quality_flag == "MISSING_MEASUREMENT":
            return "missing"

        if evaluation.crowded:
            return "crowded"

        if evaluation.valid_measurement:
            return "valid"

        return "invalid"

    def _overlay_photometry(
        self,
        axis: Axes,
        observation: Observation,
        *,
        max_overlays: int | None,
        target_match: TargetSourceMatch | None,
    ) -> None:
        """
        Overlay aperture-photometry geometry.

        Geometry is taken directly from the PhotometryResult stored in
        the Observation.
        """

        photometry_result = observation.photometry

        if photometry_result is None:
            self._logger.debug(
                "Observation contains no photometry result.",
            )
            return

        measurements = photometry_result.measurements

        if not measurements:
            self._logger.debug(
                "No photometric measurements available for rendering.",
            )
            return

        aperture_radius = float(
            photometry_result.aperture_radius,
        )

        annulus_inner_radius = float(
            photometry_result.annulus_inner_radius,
        )

        annulus_outer_radius = float(
            photometry_result.annulus_outer_radius,
        )

        measurements_to_render = measurements

        if max_overlays is not None:
            measurements_to_render = measurements[
                :max_overlays
            ]

        target_identifier = (
            target_match.source_identifier
            if target_match is not None
            else None
        )

        rendered_count = len(
            measurements_to_render,
        )

        omitted_count = (
            len(measurements)
            - rendered_count
        )

        for measurement in measurements_to_render:
            x = float(
                measurement.x_centroid,
            )

            y = float(
                measurement.y_centroid,
            )

            is_target = (
                measurement.source_identifier
                == target_identifier
            )

            aperture = Circle(
                (x, y),
                radius=aperture_radius,
                fill=False,
                edgecolor=(
                    "lime"
                    if is_target
                    else "cyan"
                ),
                linewidth=(
                    2.2
                    if is_target
                    else 1.2
                ),
            )

            annulus_inner = Circle(
                (x, y),
                radius=annulus_inner_radius,
                fill=False,
                edgecolor=(
                    "lime"
                    if is_target
                    else "yellow"
                ),
                linewidth=(
                    1.8
                    if is_target
                    else 1.0
                ),
                linestyle="--",
            )

            annulus_outer = Circle(
                (x, y),
                radius=annulus_outer_radius,
                fill=False,
                edgecolor=(
                    "lime"
                    if is_target
                    else "yellow"
                ),
                linewidth=(
                    1.8
                    if is_target
                    else 1.0
                ),
                linestyle="--",
            )

            axis.add_patch(
                aperture,
            )

            axis.add_patch(
                annulus_inner,
            )

            axis.add_patch(
                annulus_outer,
            )

        axis.plot(
            [],
            [],
            color="cyan",
            linewidth=1.2,
            label=(
                f"Source aperture "
                f"(r={aperture_radius:.1f}px)"
            ),
        )

        axis.plot(
            [],
            [],
            color="yellow",
            linestyle="--",
            linewidth=1.0,
            label=(
                "Background annulus "
                f"({annulus_inner_radius:.1f}–"
                f"{annulus_outer_radius:.1f}px)"
            ),
        )

        if target_identifier is not None:
            axis.plot(
                [],
                [],
                color="lime",
                linewidth=2.2,
                label="Target aperture",
            )

        self._logger.info(
            "Rendered photometry geometry for %d of %d "
            "measurements.",
            rendered_count,
            len(measurements),
        )

        if omitted_count > 0:
            self._logger.info(
                "Omitted %d photometry overlays from visualisation "
                "because the configured display limit is %d.",
                omitted_count,
                max_overlays,
            )

    def _overlay_target(
        self,
        axis: Axes,
        *,
        target_match: TargetSourceMatch | None,
        include_separation: bool = True,
    ) -> None:
        """
        Overlay the resolved target-source association.

        The expected target position is rendered as a filled gold star
        with a dark outline so that it remains visible against both
        bright and dark astronomical backgrounds.
        """

        if target_match is None:
            self._logger.debug(
                "Target overlay requested without a target match."
            )
            return

        target_x = float(
            target_match.target_x,
        )

        target_y = float(
            target_match.target_y,
        )

        source_x = float(
            target_match.source_x,
        )

        source_y = float(
            target_match.source_y,
        )

        #
        # Expected target position.
        #
        # A filled gold star with a black outline provides substantially
        # better contrast than the previous hollow white star, especially
        # when the expected position falls on a bright source or pixel.
        #
        axis.scatter(
            [target_x],
            [target_y],
            marker="*",
            s=220,
            facecolors="gold",
            edgecolors="black",
            linewidths=1.5,
            label="Expected Target Position",
            zorder=12,
        )

        #
        # Matched detected source.
        #
        axis.scatter(
            [source_x],
            [source_y],
            marker="o",
            s=110,
            facecolors="none",
            edgecolors="lime",
            linewidths=2.0,
            label=(
                f"Target Source "
                f"({target_match.source_identifier})"
            ),
            zorder=11,
        )

        #
        # Expected-to-detected separation.
        #
        # Gold provides a strong visual association with the expected
        # target marker while remaining visible over bright image pixels.
        #
        if include_separation:
            axis.plot(
                [target_x, source_x],
                [target_y, source_y],
                color="gold",
                linestyle=":",
                linewidth=1.4,
                label=(
                    f"Separation "
                    f"({target_match.separation_pixels:.2f}px)"
                ),
                zorder=10,
            )

        #
        # Target annotation.
        #
        axis.annotate(
            "Target",
            (
                target_x,
                target_y,
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
            },
            zorder=13,
        )

        self._logger.info(
            "Rendered target '%s' matched to source '%s' "
            "with separation %.3f pixels.",
            target_match.target_name,
            target_match.source_identifier,
            target_match.separation_pixels,
        )

    def _overlay_target_photometry(
        self,
        axis: Axes,
        observation: Observation,
        *,
        target_match: TargetSourceMatch,
    ) -> None:
        """
        Overlay production aperture geometry for the resolved target.
        """

        photometry_result = observation.photometry

        if photometry_result is None:
            return

        target_measurement = next(
            (
                measurement
                for measurement
                in photometry_result.measurements
                if measurement.source_identifier
                == target_match.source_identifier
            ),
            None,
        )

        if target_measurement is None:
            self._logger.warning(
                "No photometric measurement found for target source '%s'.",
                target_match.source_identifier,
            )
            return

        x = float(
            target_measurement.x_centroid,
        )

        y = float(
            target_measurement.y_centroid,
        )

        aperture_radius = float(
            photometry_result.aperture_radius,
        )

        annulus_inner_radius = float(
            photometry_result.annulus_inner_radius,
        )

        annulus_outer_radius = float(
            photometry_result.annulus_outer_radius,
        )

        axis.add_patch(
            Circle(
                (x, y),
                radius=aperture_radius,
                fill=False,
                edgecolor="lime",
                linewidth=2.5,
                label=(
                    f"Target aperture "
                    f"(r={aperture_radius:.1f}px)"
                ),
                zorder=8,
            ),
        )

        axis.add_patch(
            Circle(
                (x, y),
                radius=annulus_inner_radius,
                fill=False,
                edgecolor="lime",
                linestyle="--",
                linewidth=1.5,
                zorder=8,
            ),
        )

        axis.add_patch(
            Circle(
                (x, y),
                radius=annulus_outer_radius,
                fill=False,
                edgecolor="lime",
                linestyle="--",
                linewidth=1.5,
                label=(
                    "Target background annulus"
                ),
                zorder=8,
            ),
        )

    def _overlay_target_quality(
        self,
        axis: Axes,
        *,
        evaluation: PhotometryEvaluation,
        target_match: TargetSourceMatch,
    ) -> None:
        """
        Display the already-computed target source quality state.
        """

        source_evaluation = next(
            (
                item
                for item in evaluation.source_evaluations
                if item.source_identifier
                == target_match.source_identifier
            ),
            None,
        )

        if source_evaluation is None:
            self._logger.warning(
                "No source evaluation found for target source '%s'.",
                target_match.source_identifier,
            )
            return

        quality = source_evaluation.quality_flag

        status = (
            "VALID"
            if source_evaluation.valid_measurement
            else quality
        )

        crowding = (
            "YES"
            if source_evaluation.crowded
            else "NO"
        )

        text = (
            "TARGET PHOTOMETRY\n"
            f"Source: {source_evaluation.source_identifier}\n"
            f"Quality: {status}\n"
            f"Crowded: {crowding}\n"
            f"Neighbours: "
            f"{source_evaluation.neighbour_count}\n"
            f"Net flux: "
            f"{self._format_value(source_evaluation.net_flux)}\n"
            f"Instrumental mag: "
            f"{self._format_value(source_evaluation.instrumental_magnitude)}\n"
            f"Background: "
            f"{self._format_value(source_evaluation.background_level)}\n"
            f"Background fraction: "
            f"{self._format_value(source_evaluation.background_fraction)}\n"
            f"Aperture coverage: "
            f"{self._format_value(source_evaluation.aperture_coverage)}\n"
            f"Annulus coverage: "
            f"{self._format_value(source_evaluation.annulus_coverage)}"
        )

        axis.text(
            0.02,
            0.98,
            text,
            transform=axis.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=9,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "black",
                "alpha": 0.75,
            },
            color="white",
            zorder=20,
        )

    @staticmethod
    def _format_value(
        value: float | None,
    ) -> str:
        """
        Format an optional numerical evaluation value for display.

        This performs presentation formatting only.
        """

        if value is None:
            return "N/A"

        if not isfinite(
            float(value),
        ):
            return "N/A"

        return f"{float(value):.5g}"

    def _overlay_active_local_field(
        self,
        axis: Axes,
        observation: Observation,
        image_shape: tuple[int, ...],
    ) -> None:
        """Draw the authoritative LocalField boundary on downstream views.

        The LocalField is a scientific input to the photometric experiment, so
        downstream diagnostic plots should make its spatial scope explicit.
        This overlay is presentation-only: it does not alter the image or the
        LocalField selection.
        """
        local_field = self._active_local_field
        if local_field is None:
            return
        if self._active_local_field_observation_id != id(observation):
            return
        if len(image_shape) != 2:
            return

        target = local_field.target_source
        centre_x = float(target.x_centroid)
        centre_y = float(target.y_centroid)
        radius = float(local_field.selection_radius_pixels)
        if not all(isfinite(value) for value in (centre_x, centre_y, radius)):
            return
        if radius <= 0.0:
            return

        axis.add_patch(
            Circle(
                (centre_x, centre_y),
                radius=radius,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                edgecolor="yellow",
                alpha=0.95,
                label=f"LocalField boundary ({radius:.2f} px)",
                zorder=30,
            )
        )

    @staticmethod
    def _create_positive_scarlet_normalisation(
        image: NDArray[np.number],
    ) -> Normalize:
        """Create a robust positive-intensity display normalisation.

        This helper is presentation-only. A percentile-based interval is used
        so faint SCARLET components remain visible in crowded LocalField
        views while isolated saturated pixels do not flatten the image.
        """
        finite = np.asarray(image, dtype=float)
        finite = finite[np.isfinite(finite)]
        finite = finite[finite > 0.0]
        if finite.size < 10:
            finite = np.asarray(image, dtype=float)
            finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return Normalize(vmin=0.0, vmax=1.0)

        vmin = float(np.nanpercentile(finite, 1.0))
        vmax = float(np.nanpercentile(finite, 99.7))
        if not isfinite(vmin):
            vmin = 0.0
        if not isfinite(vmax) or vmax <= vmin:
            vmax = max(float(np.nanmax(finite)), vmin + 1.0)
        return Normalize(vmin=max(0.0, vmin), vmax=vmax)

    def _create_scarlet_normalisation(
        self,
        image: NDArray[np.number],
        observation: Observation,
        image_name: str,
    ) -> Normalize:
        """Create a presentation normalisation for SCARLET field products.

        SCARLET field products can contain a very large number of near-zero
        pixels outside the selected source region. A full-frame ZScale can
        therefore compress the useful LocalField structure into a nearly
        binary-looking display. When an active LocalField exists, derive the
        display interval from that target-centred region instead. The stored
        SCARLET image is never modified.
        """
        local_field = self._active_local_field
        radius = self._active_local_field_view_radius(observation)
        if local_field is None or radius is None or image.ndim != 2:
            return self._create_scarlet_asinh_normalisation(image)

        target = local_field.target_source
        centre_x = float(target.x_centroid)
        centre_y = float(target.y_centroid)
        height, width = image.shape
        x0 = max(0, int(np.floor(centre_x - radius)))
        x1 = min(width, int(np.ceil(centre_x + radius)) + 1)
        y0 = max(0, int(np.floor(centre_y - radius)))
        y1 = min(height, int(np.ceil(centre_y + radius)) + 1)
        if x1 <= x0 or y1 <= y0:
            return self._create_scarlet_asinh_normalisation(image)

        local_values = self._prepare_scarlet_display_array(
            np.asarray(image[y0:y1, x0:x1], dtype=float)
        )
        finite = local_values[np.isfinite(local_values)]
        if finite.size < 10:
            return self._create_scarlet_asinh_normalisation(image)

        return self._create_scarlet_asinh_normalisation(local_values)

    def _active_local_field_view_radius(
        self,
        observation: Observation,
    ) -> float | None:
        """Return the presentation radius for the active LocalField.

        The context is valid only for the Observation that established it.
        This prevents a LocalField from one experiment leaking into the next
        observation when the same visualizer instance is reused.
        """
        local_field = self._active_local_field
        if local_field is None:
            return None

        if self._active_local_field_observation_id != id(observation):
            return None

        selection_radius = float(local_field.selection_radius_pixels)
        if not isfinite(selection_radius) or selection_radius <= 0.0:
            return None

        return max(
            self._LOCAL_FIELD_MIN_VIEW_RADIUS,
            selection_radius * self._LOCAL_FIELD_VIEW_SCALE,
        )

    def _apply_active_local_field_view(
        self,
        axis: Axes,
        observation: Observation,
        image_shape: tuple[int, ...],
    ) -> None:
        """Apply the active target-centred LocalField view to an image axis.

        This is presentation-only. It does not crop or modify the underlying
        image array; it changes only the Matplotlib display limits.
        """
        local_field = self._active_local_field
        radius = self._active_local_field_view_radius(observation)
        if local_field is None or radius is None:
            return

        if len(image_shape) != 2:
            return

        height, width = image_shape
        target = local_field.target_source
        centre_x = float(target.x_centroid)
        centre_y = float(target.y_centroid)

        if not isfinite(centre_x) or not isfinite(centre_y):
            return

        x_min = max(0.0, centre_x - radius)
        x_max = min(float(width - 1), centre_x + radius)
        y_min = max(0.0, centre_y - radius)
        y_max = min(float(height - 1), centre_y + radius)

        self._set_view_limits(
            axis,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        self._logger.debug(
            "Applied active LocalField view to '%s': centre=(%.3f, %.3f), "
            "radius=%.3f px.",
            observation.metadata.target_name,
            centre_x,
            centre_y,
            radius,
        )

    def _calculate_target_zoom_radius(
        self,
        observation: Observation,
    ) -> float:
        """
        Determine a suitable target detail radius from production
        photometry geometry.
        """

        photometry_result = observation.photometry

        if photometry_result is None:
            return 15.0

        outer_radius = float(
            photometry_result.annulus_outer_radius,
        )

        return max(
            10.0,
            outer_radius * 1.5,
        )

    @staticmethod
    def _set_view_limits(
        axis: Axes,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        """
        Apply image limits for a target detail view.
        """

        axis.set_xlim(
            x_min,
            x_max,
        )

        axis.set_ylim(
            y_min,
            y_max,
        )

    def _refresh_legend(
        self,
        axis: Axes,
    ) -> None:
        """
        Create a single combined legend from active overlays.

        Duplicate labels are removed while preserving the order in which
        Matplotlib produced the legend handles.
        """

        handles, labels = axis.get_legend_handles_labels()

        if not handles:
            return

        unique_handles: list[Artist] = []
        unique_labels: list[str] = []
        seen_labels: set[str] = set()

        for handle, label in zip(
            handles,
            labels,
        ):
            if label in seen_labels:
                continue

            seen_labels.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)

        legend = axis.legend(
            handles=unique_handles,
            labels=unique_labels,
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.95,
        )

        # Keep the legend visually dominant over dense astronomical
        # imagery and source overlays, particularly for crowded-field
        # experiments such as SDSS M67.
        legend.set_zorder(100)

    def _create_normalisation(
        self,
        image: NDArray[np.number],
        image_name: str,
    ) -> Normalize:
        """
        Create an astronomical display normalisation.

        Estimated background images use percentile clipping to better
        reveal smooth large-scale structure. Other image products use
        an astronomical ZScale interval.
        """

        if image_name == "background":
            return cast(
                Normalize,
                ImageNormalize(
                    image,
                    interval=PercentileInterval(
                        99.5,
                    ),
                    stretch=AsinhStretch(),
                ),
            )

        return cast(
            Normalize,
            ImageNormalize(
                image,
                interval=ZScaleInterval(),
                stretch=AsinhStretch(),
            ),
        )