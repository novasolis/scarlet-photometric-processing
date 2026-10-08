"""
TESS time-series measurement writer.

Transforms completed time-series WorkflowResult objects into immutable
TimeSeriesMeasurementRecord objects and persists them as a CSV dataset.

Responsibilities
----------------
This module is responsible for:

    ExperimentConfig + WorkflowResult sequence
                |
                v
    TimeSeriesMeasurementRecord sequence
                |
                v
    results/exports/timeseries_measurements.csv

This module does NOT perform scientific calculations.

Scientific quantities such as net flux, instrumental magnitude,
background fraction, aperture coverage and annulus coverage are copied
directly from the completed photometry measurements.

Persistence is implemented as a deterministic upsert:

    experiment_id
    + cadence_index
    + source_identifier

is the logical key for one time-series measurement. The exported
``source_identifier`` is a persistent time-series source ID, not the
cadence-local DAOStarFinder detection ID.

When the same key already exists in the output CSV, the existing row is
replaced. Different experiments/cadences remain untouched.

The writer is intentionally independent of TESS/Lightkurve-specific
types. TESS-specific cadence identity is read from the Observation raw
image metadata populated by the TESS ObservationBuilder.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from core.models import ExperimentConfig
from domain.models import (
    DetectedSource,
    Observation,
    PhotometryMeasurement,
)
from export.models import TimeSeriesMeasurementRecord
from utils.logger import get_logger
from workflows.models import WorkflowResult


class TimeSeriesWriter:
    """
    Persist cadence-level time-series photometric measurements.

    Parameters
    ----------
    output_path
        Destination CSV path.

    logger
        Optional logger. If omitted, the project logger is used.

    Notes
    -----
    The writer performs no scientific calculations.

    DAOStarFinder source identifiers are cadence-local. The writer therefore
    establishes persistent source identities from the first completed
    LocalField cadence and propagates them to subsequent cadences using
    validated centroid matching.

    Existing rows matching the logical time-series record key are
    replaced rather than duplicated.
    """

    _COLUMNS: tuple[str, ...] = (
        "experiment_id",
        "target_name",
        "survey",
        "experiment_type",
        "pipeline",
        "cadence_index",
        "observation_time",
        "observation_time_scale",
        "observation_time_format",
        "source_identifier",
        "is_target_source",
        "x_centroid",
        "y_centroid",
        "target_x",
        "target_y",
        "target_source_separation_pixels",
        "peak_value",
        "aperture_flux",
        "background_level",
        "background_flux",
        "net_flux",
        "instrumental_magnitude",
        "aperture_area",
        "background_area",
        "aperture_coverage",
        "annulus_coverage",
        "background_fraction",
        "crowded",
        "valid",
        "quality_flag",
    )

    _KEY_COLUMNS: tuple[str, ...] = (
        "experiment_id",
        "cadence_index",
        "source_identifier",
    )

    # TESS source identifiers produced by DAOStarFinder are cadence-local.
    # Persistent time-series identity is therefore established by matching
    # LocalField source centroids to the immediately preceding active
    # population. The tolerance remains deliberately conservative.
    _PERSISTENT_SOURCE_PREFIX = "TS-"
    _SOURCE_MATCH_TOLERANCE_PIXELS = 2.5

    def __init__(
        self,
        *,
        output_path: Path | str = (
            Path("results")
            / "exports"
            / "timeseries_measurements.csv"
        ),
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the time-series writer.
        """

        self._output_path = Path(
            output_path,
        )

        self._logger = logger or get_logger(
            __name__,
        )

    # ========================================================================
    # Public API
    # ========================================================================

    @property
    def output_path(
        self,
    ) -> Path:
        """
        Return the configured output path.
        """

        return self._output_path

    def build_records(
        self,
        *,
        config: ExperimentConfig,
        results: Sequence[WorkflowResult],
    ) -> tuple[TimeSeriesMeasurementRecord, ...]:
        """
        Transform completed workflow results into immutable time-series
        export records.

        Parameters
        ----------
        config
            Experiment configuration used to create the workflow.

        results
            Ordered sequence of completed WorkflowResult objects.

        Returns
        -------
        tuple[TimeSeriesMeasurementRecord, ...]
            One record for each source represented by the completed
            evaluation/LocalField population at each cadence.

        Raises
        ------
        ValueError
            If the configuration or result sequence is invalid.
        """

        self._validate_config(
            config,
        )

        if results is None:
            raise ValueError(
                "Time-series workflow results cannot be None.",
            )

        result_sequence = tuple(
            results,
        )

        if not result_sequence:
            raise ValueError(
                "Time-series workflow results cannot be empty.",
            )

        records: list[
            TimeSeriesMeasurementRecord
        ] = []

        seen_keys: set[
            tuple[str, int, str]
        ] = set()

        experiment_id = self._build_experiment_id(
            config=config,
            first_result=result_sequence[0],
        )

        persistent_source_mappings = (
            self._build_persistent_source_mappings(
                results=result_sequence,
            )
        )

        for sequence_position, result in enumerate(
            result_sequence,
            start=1,
        ):
            if not isinstance(
                result,
                WorkflowResult,
            ):
                raise ValueError(
                    (
                        "Time-series result at sequence position "
                        f"{sequence_position} is not a WorkflowResult."
                    ),
                )

            cadence_records = self._build_records_for_result(
                config=config,
                experiment_id=experiment_id,
                result=result,
                persistent_source_mapping=(
                    persistent_source_mappings[
                        sequence_position - 1
                    ]
                ),
            )

            for record in cadence_records:
                key = self._record_key(
                    record,
                )

                if key in seen_keys:
                    raise ValueError(
                        (
                            "Duplicate time-series record key detected "
                            "within the supplied workflow results: "
                            f"{key}."
                        ),
                    )

                seen_keys.add(
                    key,
                )
                records.append(
                    record,
                )

        if not records:
            raise ValueError(
                "No time-series measurement records were generated.",
            )

        self._validate_record_sequence(
            records,
        )

        self._logger.info(
            (
                "Built %d time-series measurement record(s) for "
                "experiment '%s'."
            ),
            len(records),
            experiment_id,
        )

        return tuple(
            records,
        )

    def write(
        self,
        *,
        config: ExperimentConfig,
        results: Sequence[WorkflowResult],
    ) -> tuple[TimeSeriesMeasurementRecord, ...]:
        """
        Transform and persist a completed time-series result sequence.

        Existing records with matching logical keys are replaced.

        Parameters
        ----------
        config
            Experiment configuration used to create the workflow.

        results
            Ordered sequence of completed WorkflowResult objects.

        Returns
        -------
        tuple[TimeSeriesMeasurementRecord, ...]
            Records written for the supplied time-series results.
        """

        records = self.build_records(
            config=config,
            results=results,
        )

        self.write_records(
            records=records,
        )

        return records

    def write_records(
        self,
        *,
        records: Sequence[TimeSeriesMeasurementRecord],
    ) -> None:
        """
        Persist time-series export records using deterministic upsert.

        Parameters
        ----------
        records
            Records to persist.

        Raises
        ------
        ValueError
            If records are empty or contain duplicate logical keys.
        OSError
            If the output cannot be written.
        """

        if records is None:
            raise ValueError(
                "Time-series records cannot be None.",
            )

        record_sequence = tuple(
            records,
        )

        if not record_sequence:
            raise ValueError(
                "Time-series records cannot be empty.",
            )

        self._validate_record_sequence(
            record_sequence,
        )

        new_dataframe = self._records_to_dataframe(
            record_sequence,
        )

        existing_dataframe = self._load_existing_dataframe()

        if existing_dataframe.empty:
            combined = new_dataframe
        else:
            # Normalize identity columns BEFORE computing replacement keys.
            # Existing CSV values are parsed by pandas, so source identifiers
            # such as "1" may otherwise be represented as numeric/object
            # values (for example, "1.0"). That can prevent an existing
            # row from matching the incoming logical key. After final
            # normalization both rows then become identical and the duplicate
            # validator correctly rejects the combined dataframe.
            #
            # Normalizing the identity columns first makes upsert semantics
            # deterministic across CSV round-trips.
            existing_dataframe = self._normalise_identity_columns(
                existing_dataframe,
            )

            new_dataframe = self._normalise_identity_columns(
                new_dataframe,
            )

            combined = self._upsert_dataframes(
                existing=existing_dataframe,
                incoming=new_dataframe,
            )

        combined = self._normalise_output_dataframe(
            combined,
        )

        self._persist_dataframe_atomically(
            dataframe=combined,
        )

        self._logger.info(
            (
                "Time-series export persisted successfully: "
                "%d record(s) written."
            ),
            len(record_sequence),
        )

        self._logger.info(
            "Time-series export: %s",
            self._output_path,
        )

    # ========================================================================
    # Result transformation
    # ========================================================================

    def _build_records_for_result(
        self,
        *,
        config: ExperimentConfig,
        experiment_id: str,
        result: WorkflowResult,
        persistent_source_mapping: dict[str, str],
    ) -> tuple[TimeSeriesMeasurementRecord, ...]:
        """
        Convert one cadence-level WorkflowResult into source records.

        The export source population is the completed evaluation/LocalField
        population, not the full DAOStarFinder detection population.
        """

        observation = result.observation

        if observation is None:
            raise ValueError(
                "Workflow result does not contain an Observation.",
            )

        if observation.detections is None:
            raise ValueError(
                (
                    "Cannot export time-series result without completed "
                    "detection results."
                ),
            )

        if observation.photometry is None:
            raise ValueError(
                (
                    "Cannot export time-series result without completed "
                    "photometry results."
                ),
            )

        if result.evaluation is None:
            raise ValueError(
                (
                    "Cannot export time-series result without completed "
                    "evaluation."
                ),
            )

        cadence_index = self._extract_cadence_index(
            observation,
        )

        target_source_identifier = (
            result.target_match.source_identifier
            if result.target_match is not None
            else None
        )

        target_x = (
            result.target_match.target_x
            if result.target_match is not None
            else None
        )

        target_y = (
            result.target_match.target_y
            if result.target_match is not None
            else None
        )

        target_source_separation = (
            result.target_match.separation_pixels
            if result.target_match is not None
            else None
        )

        measurements_by_source = self._index_measurements(
            observation.photometry.measurements,
        )

        quality_flags_by_source = (
            self._index_quality_flags(
                result,
            )
        )

        export_source_identifiers = (
            self._extract_export_source_identifiers(
                result,
            )
        )

        # The persistent-source mapping defines the complete study population.
        # Sources that were detected for this cadence but dropped out of the
        # complete time series are deliberately excluded from export.
        export_source_identifiers = tuple(
            identifier
            for identifier in export_source_identifiers
            if identifier in persistent_source_mapping
        )

        if not export_source_identifiers:
            raise ValueError(
                (
                    "Time-series cadence contains no sources in the final "
                    f"complete-study population at cadence {cadence_index}."
                ),
            )

        missing_quality_sources = [
            identifier
            for identifier in export_source_identifiers
            if identifier not in quality_flags_by_source
        ]

        if missing_quality_sources:
            raise ValueError(
                (
                    "The completed evaluation is missing quality "
                    "classification for authoritative export source(s): "
                    f"{sorted(missing_quality_sources)}."
                ),
            )

        crowded_by_source = self._extract_crowding_state(
            result,
            source_identifiers=export_source_identifiers,
        )

        detected_sources_by_identifier = {
            source.identifier: source
            for source in observation.detections.sources
        }

        missing_detected_sources = [
            identifier
            for identifier in export_source_identifiers
            if identifier not in detected_sources_by_identifier
        ]

        if missing_detected_sources:
            raise ValueError(
                (
                    "The completed evaluation references source(s) that "
                    "are absent from the detection results: "
                    f"{sorted(missing_detected_sources)}."
                ),
            )

        records: list[
            TimeSeriesMeasurementRecord
        ] = []

        for identifier in export_source_identifiers:
            source = detected_sources_by_identifier[
                identifier
            ]
            measurement = measurements_by_source.get(
                source.identifier,
            )

            quality_flag = quality_flags_by_source.get(
                source.identifier,
            )

            if quality_flag is None:
                raise ValueError(
                    (
                        "Time-series export is missing a quality "
                        "classification for detected source "
                        f"'{source.identifier}' at cadence "
                        f"{cadence_index}."
                    ),
                )

            records.append(
                self._create_record(
                    config=config,
                    experiment_id=experiment_id,
                    observation=observation,
                    source=source,
                    measurement=measurement,
                    cadence_index=cadence_index,
                    persistent_source_identifier=(
                        persistent_source_mapping[
                            source.identifier
                        ]
                    ),
                    target_source_identifier=(
                        target_source_identifier
                    ),
                    target_x=target_x,
                    target_y=target_y,
                    target_source_separation=(
                        target_source_separation
                    ),
                    crowded=crowded_by_source.get(
                        source.identifier,
                        False,
                    ),
                    quality_flag=quality_flag,
                ),
            )

        if not records:
            raise ValueError(
                (
                    "Time-series cadence produced no detected sources "
                    f"at cadence {cadence_index}."
                ),
            )

        self._validate_cadence_records(
            records,
        )

        return tuple(
            records,
        )

    def _create_record(
        self,
        *,
        config: ExperimentConfig,
        experiment_id: str,
        observation: Observation,
        source: DetectedSource,
        measurement: PhotometryMeasurement | None,
        cadence_index: int,
        persistent_source_identifier: str,
        target_source_identifier: str | None,
        target_x: float | None,
        target_y: float | None,
        target_source_separation: float | None,
        crowded: bool,
        quality_flag: str,
    ) -> TimeSeriesMeasurementRecord:
        """
        Create one immutable time-series export record.

        Missing measurements are intentionally preserved. They are
        represented by ``None`` photometric values and ``valid=False``,
        matching the semantics of the existing source-level exporter.
        """

        if measurement is not None:
            if (
                measurement.source_identifier
                != source.identifier
            ):
                raise ValueError(
                    (
                        "Photometric measurement source identifier does "
                        "not match detected source identifier "
                        f"'{source.identifier}'."
                    ),
                )

        return TimeSeriesMeasurementRecord(
            experiment_id=experiment_id,
            target_name=(
                config.target.target_name
            ),
            survey=config.survey.value,
            experiment_type=config.experiment_type.value,
            pipeline=config.pipeline.value,
            cadence_index=cadence_index,
            observation_time=(
                observation.metadata.observation_time
            ),
            observation_time_scale=(
                observation.metadata.observation_time_scale
            ),
            observation_time_format=(
                observation.metadata.observation_time_format
            ),
            source_identifier=persistent_source_identifier,
            is_target_source=(
                source.identifier
                == target_source_identifier
            ),
            x_centroid=source.x_centroid,
            y_centroid=source.y_centroid,
            target_x=target_x,
            target_y=target_y,
            target_source_separation_pixels=(
                target_source_separation
            ),
            peak_value=source.peak_value,
            aperture_flux=(
                measurement.aperture_flux
                if measurement is not None
                else None
            ),
            background_level=(
                measurement.background_level
                if measurement is not None
                else None
            ),
            background_flux=(
                measurement.background_flux
                if measurement is not None
                else None
            ),
            net_flux=(
                measurement.net_flux
                if measurement is not None
                else None
            ),
            instrumental_magnitude=(
                measurement.instrumental_magnitude
                if measurement is not None
                else None
            ),
            aperture_area=(
                measurement.aperture_area
                if measurement is not None
                else None
            ),
            background_area=(
                measurement.background_area
                if measurement is not None
                else None
            ),
            aperture_coverage=(
                measurement.aperture_coverage
                if measurement is not None
                else None
            ),
            annulus_coverage=(
                measurement.annulus_coverage
                if measurement is not None
                else None
            ),
            background_fraction=(
                measurement.background_fraction
                if measurement is not None
                else None
            ),
            crowded=crowded,
            valid=(
                measurement.valid
                if measurement is not None
                else False
            ),
            quality_flag=quality_flag,
        )


    # ========================================================================
    # Persistent source identity
    # ========================================================================

    def _build_persistent_source_mappings(
        self,
        *,
        results: Sequence[WorkflowResult],
    ) -> tuple[dict[str, str], ...]:
        """
        Build stable source identities across all time-series cadences.

        DAOStarFinder source identifiers are cadence-local. Persistent
        identity is established on the first cadence and propagated
        sequentially using centroid matching against the immediately
        preceding cadence.

        A non-target source that cannot be matched within the configured
        tolerance is treated as a source dropout: it is retired from the
        active tracking population and is excluded from the complete
        time-series study. Later reappearance of that source does not
        reactivate it. This ensures that the exported time series contains
        only sources that are reliably present across every selected cadence.

        The target source is different: if its identity cannot be maintained
        at any cadence, the experiment fails because a target light curve
        cannot be established reliably.
        """
        if not results:
            raise ValueError(
                "Cannot build persistent source identities from no results.",
            )

        reference_result = results[0]
        reference_sources = self._get_export_sources(reference_result)

        if not reference_sources:
            raise ValueError(
                "The reference cadence contains no LocalField export sources.",
            )

        reference_target = self._get_target_coordinates(reference_result)

        reference_source_ids = [
            source.identifier
            for source in reference_sources
        ]

        if len(reference_source_ids) != len(set(reference_source_ids)):
            raise ValueError(
                "Reference cadence contains duplicate source identifiers.",
            )

        target_source_id = self._nearest_source_identifier(
            sources=reference_sources,
            x=reference_target[0],
            y=reference_target[1],
            context="reference cadence target",
        )

        ordered_reference_ids = [target_source_id]
        remaining_sources = [
            source
            for source in reference_sources
            if source.identifier != target_source_id
        ]

        remaining_sources.sort(
            key=lambda source: (
                self._squared_distance(
                    source.x_centroid,
                    source.y_centroid,
                    reference_target[0],
                    reference_target[1],
                ),
                source.x_centroid,
                source.y_centroid,
                source.identifier,
            ),
        )

        ordered_reference_ids.extend(
            source.identifier
            for source in remaining_sources
        )

        persistent_by_reference_id = {
            identifier: f"{self._PERSISTENT_SOURCE_PREFIX}{index:03d}"
            for index, identifier in enumerate(
                ordered_reference_ids,
                start=1,
            )
        }

        target_persistent_id = persistent_by_reference_id[target_source_id]

        # First cadence is the identity anchor.
        mappings: list[dict[str, str]] = [
            {
                source.identifier: persistent_by_reference_id[source.identifier]
                for source in reference_sources
            }
        ]

        previous_positions = {
            source.identifier: (
                source.x_centroid,
                source.y_centroid,
            )
            for source in reference_sources
        }
        previous_persistent_ids = dict(persistent_by_reference_id)

        for sequence_position, result in enumerate(
            results[1:],
            start=2,
        ):
            current_sources = self._get_export_sources(result)

            mapping = self._match_sources_to_reference(
                current_sources=current_sources,
                reference_positions=previous_positions,
                persistent_by_reference_id=previous_persistent_ids,
                sequence_position=sequence_position,
            )

            # The target must survive every cadence. A non-target dropout is
            # tolerated, but loss of the target invalidates the experiment.
            current_target_source_id = (
                result.target_match.source_identifier
                if result.target_match is not None
                else None
            )

            if (
                current_target_source_id is None
                or mapping.get(current_target_source_id)
                != target_persistent_id
            ):
                raise ValueError(
                    (
                        "Target source identity could not be maintained at "
                        f"sequence position {sequence_position}; expected "
                        f"persistent target '{target_persistent_id}'."
                    ),
                )

            mappings.append(mapping)

            # Advance only the successfully matched active population.
            # Unmatched reference sources are retired permanently; unmatched
            # current detections are ignored as untracked/new detections.
            previous_positions = {
                source.identifier: (
                    source.x_centroid,
                    source.y_centroid,
                )
                for source in current_sources
                if source.identifier in mapping
            }

            previous_persistent_ids = {
                source.identifier: mapping[source.identifier]
                for source in current_sources
                if source.identifier in mapping
            }

        # The exported study population is the intersection of persistent
        # identities present at every cadence. This removes a source from the
        # entire time series if it drops out at any point, rather than keeping
        # a partial light curve before its dropout.
        common_persistent_ids = set(mappings[0].values())
        for mapping in mappings[1:]:
            common_persistent_ids.intersection_update(mapping.values())

        if target_persistent_id not in common_persistent_ids:
            raise ValueError(
                (
                    "Target persistent source is not present across all "
                    "selected cadences; cannot produce a complete target "
                    "time series."
                ),
            )

        excluded_persistent_ids = set(mappings[0].values()) - common_persistent_ids

        if excluded_persistent_ids:
            self._logger.warning(
                (
                    "Excluded %d non-persistent source(s) from the complete "
                    "time-series study because they were absent or "
                    "unmatchable at one or more cadences: %s."
                ),
                len(excluded_persistent_ids),
                sorted(excluded_persistent_ids),
            )

        self._logger.info(
            (
                "Final time-series source scope: %d source(s) retained, "
                "%d source(s) excluded; target=%s."
            ),
            len(common_persistent_ids),
            len(excluded_persistent_ids),
            target_persistent_id,
        )

        # Filter every cadence to the same complete-study population.
        return tuple(
            {
                current_id: persistent_id
                for current_id, persistent_id in mapping.items()
                if persistent_id in common_persistent_ids
            }
            for mapping in mappings
        )

    def _get_export_sources(
        self,
        result: WorkflowResult,
    ) -> tuple[DetectedSource, ...]:
        """
        Resolve the authoritative LocalField sources for one result.

        The evaluation source identifiers define which sources actually
        participated in photometry/evaluation. Their corresponding detected
        source objects provide the centroid coordinates used for persistent
        identity matching.
        """
        identifiers = self._extract_export_source_identifiers(
            result,
        )

        if result.observation.detections is None:
            raise ValueError(
                "Cannot resolve LocalField sources without detections.",
            )

        sources_by_identifier = {
            source.identifier: source
            for source in result.observation.detections.sources
        }

        missing = [
            identifier
            for identifier in identifiers
            if identifier not in sources_by_identifier
        ]

        if missing:
            raise ValueError(
                (
                    "Authoritative LocalField source(s) are absent from "
                    f"detections: {sorted(missing)}."
                ),
            )

        return tuple(
            sources_by_identifier[identifier]
            for identifier in identifiers
        )

    @staticmethod
    def _get_target_coordinates(
        result: WorkflowResult,
    ) -> tuple[float, float]:
        """
        Return the catalogue target coordinates for a workflow result.
        """
        if result.target_match is None:
            raise ValueError(
                "Cannot establish persistent source identity without a target match.",
            )

        target_x = result.target_match.target_x
        target_y = result.target_match.target_y

        if target_x is None or target_y is None:
            raise ValueError(
                "Target match does not contain valid target coordinates.",
            )

        return (
            float(target_x),
            float(target_y),
        )

    @classmethod
    def _nearest_source_identifier(
        cls,
        *,
        sources: Sequence[DetectedSource],
        x: float,
        y: float,
        context: str,
    ) -> str:
        """
        Return the nearest source identifier and reject ambiguous matches.
        """
        ranked = sorted(
            (
                (
                    cls._squared_distance(
                        source.x_centroid,
                        source.y_centroid,
                        x,
                        y,
                    ),
                    source.identifier,
                )
                for source in sources
            ),
            key=lambda item: item[0],
        )

        if not ranked:
            raise ValueError(
                f"No sources are available for {context}.",
            )

        distance_squared, identifier = ranked[0]

        if distance_squared > (
            cls._SOURCE_MATCH_TOLERANCE_PIXELS ** 2
        ):
            raise ValueError(
                (
                    f"No LocalField source is within "
                    f"{cls._SOURCE_MATCH_TOLERANCE_PIXELS:.2f} pixels "
                    f"of the {context} coordinates."
                ),
            )

        if len(ranked) > 1:
            second_distance_squared = ranked[1][0]

            # Require a meaningful separation between the nearest and second
            # nearest source. This prevents target identity from silently
            # switching in a crowded field.
            if second_distance_squared <= (
                4.0 * distance_squared
            ):
                raise ValueError(
                    (
                        f"Ambiguous source match for {context}: "
                        f"nearest='{identifier}' at "
                        f"{distance_squared ** 0.5:.4f} px, "
                        f"second={ranked[1][1]} at "
                        f"{second_distance_squared ** 0.5:.4f} px."
                    ),
                )

        return identifier

    @classmethod
    def _match_sources_to_reference(
        cls,
        *,
        current_sources: Sequence[DetectedSource],
        reference_positions: dict[str, tuple[float, float]],
        persistent_by_reference_id: dict[str, str],
        sequence_position: int,
    ) -> dict[str, str]:
        """
        Match current cadence sources to the currently active source population.

        Matching is deliberately partial. Sources that cannot be matched within
        the configured tolerance are treated as dropouts/new detections rather
        than causing the complete time-series experiment to fail. Only matched
        sources remain active for subsequent cadences.

        A deterministic greedy nearest-neighbour assignment is retained so that
        close sources cannot silently receive duplicate persistent identities.
        """
        if not reference_positions:
            raise ValueError(
                (
                    "No active persistent sources remain before sequence "
                    f"position {sequence_position}."
                ),
            )

        mapping: dict[str, str] = {}
        ranked_candidates: list[tuple[float, str, str]] = []

        for source in current_sources:
            for reference_id, (ref_x, ref_y) in reference_positions.items():
                distance_squared = cls._squared_distance(
                    source.x_centroid,
                    source.y_centroid,
                    ref_x,
                    ref_y,
                )
                ranked_candidates.append(
                    (distance_squared, source.identifier, reference_id),
                )

        ranked_candidates.sort(
            key=lambda item: (item[0], item[1], item[2]),
        )

        assigned_current: set[str] = set()
        assigned_reference: set[str] = set()
        tolerance_squared = cls._SOURCE_MATCH_TOLERANCE_PIXELS ** 2

        for distance_squared, current_id, reference_id in ranked_candidates:
            if current_id in assigned_current or reference_id in assigned_reference:
                continue

            if distance_squared > tolerance_squared:
                continue

            assigned_current.add(current_id)
            assigned_reference.add(reference_id)
            mapping[current_id] = persistent_by_reference_id[reference_id]

        unmatched_current = sorted(
            source.identifier
            for source in current_sources
            if source.identifier not in mapping
        )
        unmatched_reference = sorted(
            reference_id
            for reference_id in reference_positions
            if reference_id not in assigned_reference
        )

        if unmatched_reference or unmatched_current:
            self_logger = logging.getLogger(__name__)
            # self_logger.warning(
            #     (
            #         "Partial persistent source matching at sequence_position=%d: "
            #         "matched=%d/%d active references; retired=%s; "
            #         "ignored_current=%s; tolerance=%.2f px."
            #     ),
            #     sequence_position,
            #     len(assigned_reference),
            #     len(reference_positions),
            #     [
            #         persistent_by_reference_id[reference_id]
            #         for reference_id in unmatched_reference
            #     ],
            #     unmatched_current,
            #     cls._SOURCE_MATCH_TOLERANCE_PIXELS,
            # )

            # Keep the detailed geometry diagnostics from the previous writer,
            # but emit them at DEBUG level because a non-target dropout is now
            # an expected recoverable condition.
            for reference_id in unmatched_reference:
                ref_x, ref_y = reference_positions[reference_id]
                nearest_current = min(
                    (
                        (
                            cls._squared_distance(
                                source.x_centroid,
                                source.y_centroid,
                                ref_x,
                                ref_y,
                            ) ** 0.5,
                            source.identifier,
                        )
                        for source in current_sources
                    ),
                    key=lambda item: (item[0], item[1]),
                    default=(float("inf"), "<none>"),
                )
                self_logger.debug(
                    (
                        "Unmatched reference geometry: reference=%s "
                        "persistent_id=%s at (%.6f, %.6f); "
                        "nearest_current=%s distance=%.6f px."
                    ),
                    reference_id,
                    persistent_by_reference_id[reference_id],
                    float(ref_x),
                    float(ref_y),
                    nearest_current[1],
                    nearest_current[0],
                )

            for current_id in unmatched_current:
                source = next(
                    source
                    for source in current_sources
                    if source.identifier == current_id
                )
                nearest_reference = min(
                    (
                        (
                            cls._squared_distance(
                                source.x_centroid,
                                source.y_centroid,
                                reference_positions[reference_id][0],
                                reference_positions[reference_id][1],
                            ) ** 0.5,
                            reference_id,
                        )
                        for reference_id in reference_positions
                    ),
                    key=lambda item: (item[0], item[1]),
                    default=(float("inf"), "<none>"),
                )
                self_logger.debug(
                    (
                        "Ignored current geometry: current=%s at (%.6f, %.6f); "
                        "nearest_reference=%s distance=%.6f px."
                    ),
                    current_id,
                    float(source.x_centroid),
                    float(source.y_centroid),
                    nearest_reference[1],
                    nearest_reference[0],
                )

        return mapping

    @staticmethod
    def _squared_distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        """
        Return squared Euclidean distance in detector pixels.
        """
        dx = float(x1) - float(x2)
        dy = float(y1) - float(y2)
        return (dx * dx) + (dy * dy)

    # ========================================================================
    # Experiment identity
    # ========================================================================

    @staticmethod
    def _build_experiment_id(
        *,
        config: ExperimentConfig,
        first_result: WorkflowResult,
    ) -> str:
        """
        Build the same deterministic experiment ID convention used by
        ExperimentExporter.

        The ID intentionally uses stable experiment/dataset provenance
        rather than wall-clock time.
        """

        observation = first_result.observation

        components = (
            observation.provenance.survey.value,
            observation.provenance.dataset_identifier,
            config.target.target_name,
            config.experiment_type.name,
            config.pipeline.name,
        )

        canonical_value = "|".join(
            components,
        )

        digest = hashlib.sha256(
            canonical_value.encode(
                "utf-8",
            ),
        ).hexdigest()[:16]

        return (
            f"{config.survey.value.lower()}-"
            f"{config.pipeline.name.lower()}-"
            f"{digest}"
        )

    # ========================================================================
    # Runtime extraction
    # ========================================================================

    @staticmethod
    def _extract_cadence_index(
        observation: Observation,
    ) -> int:
        """
        Extract the zero-based cadence index from raw image metadata.

        The TESS ObservationBuilder stores cadence identity in the raw
        ImageData metadata. The writer treats this as provenance rather
        than calculating cadence identity itself.
        """

        try:
            raw_image = observation.images[
                "raw"
            ].image
        except KeyError as exc:
            raise ValueError(
                (
                    "Cannot determine TESS cadence index because the "
                    "'raw' image product is missing."
                ),
            ) from exc

        metadata = raw_image.meta

        if metadata is None:
            raise ValueError(
                (
                    "Cannot determine TESS cadence index because the "
                    "raw image contains no metadata."
                ),
            )

        if "cadence_index" not in metadata:
            raise ValueError(
                (
                    "Cannot determine TESS cadence index because "
                    "'cadence_index' is absent from raw image metadata."
                ),
            )

        value: Any = metadata[
            "cadence_index"
        ]

        if isinstance(
            value,
            bool,
        ) or not isinstance(
            value,
            int,
        ):
            raise ValueError(
                (
                    "TESS cadence_index in raw image metadata must be "
                    f"an integer, received {type(value).__name__}."
                ),
            )

        if value < 0:
            raise ValueError(
                f"TESS cadence_index cannot be negative: {value}.",
            )

        return value

    # ========================================================================
    # Quality / crowding
    # ========================================================================

    @staticmethod
    def _index_quality_flags(
        result: WorkflowResult,
    ) -> dict[str, str]:
        """
        Index source evaluation quality flags by source identifier.
        """

        source_evaluations = result.evaluation.source_evaluations

        indexed: dict[str, str] = {}

        for source_evaluation in source_evaluations:
            identifier = source_evaluation.source_identifier

            if not identifier:
                raise ValueError(
                    (
                        "Source evaluation contains an empty source "
                        "identifier."
                    ),
                )

            if identifier in indexed:
                raise ValueError(
                    (
                        "Duplicate source evaluation found for "
                        f"source '{identifier}'."
                    ),
                )

            quality_flag = source_evaluation.quality_flag

            if not quality_flag:
                raise ValueError(
                    (
                        "Source evaluation contains an empty quality "
                        f"flag for source '{identifier}'."
                    ),
                )

            indexed[
                identifier
            ] = quality_flag

        return indexed

    @staticmethod
    def _extract_export_source_identifiers(
        result: WorkflowResult,
    ) -> tuple[str, ...]:
        """
        Return the authoritative source population for time-series export.

        The completed evaluation is the boundary between raw source
        detection and the sources that actually participated in the
        experiment. In the current production workflow this is the
        LocalField-selected population.

        The writer deliberately does not export every DAOStarFinder
        detection. Raw detections may contain sources that were not
        selected into the LocalField and therefore have no corresponding
        photometry, quality classification or crowding classification.

        Returns
        -------
        tuple[str, ...]
            Stable source identifiers in evaluation order.

        Raises
        ------
        ValueError
            If the evaluation contains no source population, an empty
            identifier, or duplicate source identifiers.
        """

        source_evaluations = (
            result.evaluation.source_evaluations
        )

        if not source_evaluations:
            raise ValueError(
                (
                    "The completed evaluation contains no source "
                    "evaluations; cannot determine the authoritative "
                    "time-series export source population."
                ),
            )

        identifiers: list[str] = []
        seen: set[str] = set()

        for source_evaluation in source_evaluations:
            identifier = source_evaluation.source_identifier

            if not identifier:
                raise ValueError(
                    (
                        "Source evaluation contains an empty source "
                        "identifier."
                    ),
                )

            if identifier in seen:
                raise ValueError(
                    (
                        "Duplicate source evaluation found for "
                        f"source '{identifier}'."
                    ),
                )

            seen.add(identifier)
            identifiers.append(identifier)

        return tuple(identifiers)

    @staticmethod
    def _extract_crowding_state(
        result: WorkflowResult,
        *,
        source_identifiers: Sequence[str],
    ) -> dict[str, bool]:
        """
        Determine per-source crowding state without calculating crowding.

        If the evaluator exposes a per-source ``crowded`` or
        ``is_crowded`` field, those values are used.

        If the completed evaluation reports zero crowded sources, every
        source is unambiguously non-crowded.

        If crowded sources are reported but the evaluation does not
        expose their identities, the writer fails rather than inventing
        source-level classifications.
        """

        if not source_identifiers:
            raise ValueError(
                "Cannot determine crowding for an empty source population.",
            )

        evaluation = result.evaluation

        source_evaluations = (
            evaluation.source_evaluations
        )

        source_flags: dict[str, bool] = {}

        explicit_field_found = False

        for source_evaluation in source_evaluations:
            identifier = source_evaluation.source_identifier

            for field_name in (
                "crowded",
                "is_crowded",
            ):
                if not hasattr(
                    source_evaluation,
                    field_name,
                ):
                    continue

                value = getattr(
                    source_evaluation,
                    field_name,
                )

                if not isinstance(
                    value,
                    bool,
                ):
                    raise ValueError(
                        (
                            "Per-source crowding value for source "
                            f"'{identifier}' must be bool."
                        ),
                    )

                source_flags[
                    identifier
                ] = value

                explicit_field_found = True
                break

        requested_source_ids = set(
            source_identifiers,
        )

        if explicit_field_found:
            missing_identifiers = (
                requested_source_ids
                - set(source_flags)
            )

            if missing_identifiers:
                raise ValueError(
                    (
                        "Per-source crowding classification is incomplete "
                        "for the authoritative export source population. "
                        "Missing source(s): "
                        f"{sorted(missing_identifiers)}."
                    ),
                )

            return {
                identifier: source_flags[identifier]
                for identifier in source_identifiers
            }

        if evaluation.crowded_source_count == 0:
            return {
                identifier: False
                for identifier in source_identifiers
            }

        raise ValueError(
            (
                "The completed evaluation reports "
                f"{evaluation.crowded_source_count} crowded source(s), "
                "but does not expose per-source crowding identity. "
                "The time-series writer will not invent source-level "
                "crowding labels."
            ),
        )

    # ========================================================================
    # Measurement indexing
    # ========================================================================

    @staticmethod
    def _index_measurements(
        measurements: Sequence[PhotometryMeasurement],
    ) -> dict[str, PhotometryMeasurement]:
        """
        Index photometric measurements by source identifier.

        Raises
        ------
        ValueError
            If duplicate or empty source identifiers are present.
        """

        indexed: dict[
            str,
            PhotometryMeasurement,
        ] = {}

        for measurement in measurements:
            identifier = measurement.source_identifier

            if not identifier:
                raise ValueError(
                    (
                        "Photometric measurement contains an empty "
                        "source identifier."
                    ),
                )

            if identifier in indexed:
                raise ValueError(
                    (
                        "Duplicate photometric measurement found for "
                        f"source '{identifier}'."
                    ),
                )

            indexed[
                identifier
            ] = measurement

        return indexed

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_config(
        config: ExperimentConfig,
    ) -> None:
        """
        Validate the minimum configuration required for export.
        """

        if config is None:
            raise ValueError(
                "Experiment configuration cannot be None.",
            )

        if not config.target.target_name:
            raise ValueError(
                "Experiment configuration must contain a target name.",
            )

    def _validate_record_sequence(
        self,
        records: Sequence[TimeSeriesMeasurementRecord],
    ) -> None:
        """
        Validate record identity and sequence integrity.
        """

        seen_keys: set[
            tuple[str, int, str]
        ] = set()

        for record in records:
            key = self._record_key(
                record,
            )

            if key in seen_keys:
                raise ValueError(
                    (
                        "Duplicate time-series record key found: "
                        f"{key}."
                    ),
                )

            seen_keys.add(
                key,
            )

            if record.cadence_index < 0:
                raise ValueError(
                    (
                        "Time-series cadence index cannot be negative: "
                        f"{record.cadence_index}."
                    ),
                )

            if not record.experiment_id:
                raise ValueError(
                    "Time-series record has an empty experiment ID.",
                )

            if not record.source_identifier:
                raise ValueError(
                    "Time-series record has an empty source identifier.",
                )

    @staticmethod
    def _validate_cadence_records(
        records: Sequence[TimeSeriesMeasurementRecord],
    ) -> None:
        """
        Validate records produced for one cadence.
        """

        if not records:
            raise ValueError(
                "No records were produced for the cadence.",
            )

        cadence_indices = {
            record.cadence_index
            for record in records
        }

        if len(
            cadence_indices
        ) != 1:
            raise ValueError(
                (
                    "Records generated for one cadence contain "
                    f"multiple cadence indices: {sorted(cadence_indices)}."
                ),
            )

        source_identifiers = [
            record.source_identifier
            for record in records
        ]

        if len(
            source_identifiers
        ) != len(
            set(
                source_identifiers,
            ),
        ):
            raise ValueError(
                "Duplicate source identifiers exist within one cadence.",
            )

    @staticmethod
    def _record_key(
        record: TimeSeriesMeasurementRecord,
    ) -> tuple[str, int, str]:
        """
        Return the logical identity of one time-series record.
        """

        return (
            record.experiment_id,
            record.cadence_index,
            record.source_identifier,
        )

    # ========================================================================
    # DataFrame conversion
    # ========================================================================

    def _records_to_dataframe(
        self,
        records: Sequence[TimeSeriesMeasurementRecord],
    ) -> pd.DataFrame:
        """
        Convert immutable export records to a DataFrame.
        """

        rows: list[dict[str, object]] = []

        for record in records:
            rows.append(
                {
                    column: getattr(
                        record,
                        column,
                    )
                    for column in self._COLUMNS
                }
            )

        dataframe = pd.DataFrame(
            rows,
            columns=self._COLUMNS,
        )

        if dataframe.empty:
            raise ValueError(
                "Cannot create time-series export from zero records.",
            )

        return dataframe

    def _load_existing_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Load the existing output CSV, if present.

        A missing file is treated as an empty dataset.
        """

        if not self._output_path.is_file():
            return pd.DataFrame(
                columns=self._COLUMNS,
            )

        try:
            dataframe = pd.read_csv(
                self._output_path,
            )

        except (
            OSError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as exc:
            raise OSError(
                (
                    "Unable to read existing time-series export: "
                    f"{self._output_path}"
                ),
            ) from exc

        missing_columns = [
            column
            for column in self._COLUMNS
            if column
            not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                (
                    "Existing time-series export is missing required "
                    f"columns: {missing_columns}"
                ),
            )

        return dataframe[
            list(self._COLUMNS)
        ].copy()

    def _normalise_identity_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize the logical-key columns used for deterministic upsert.

        CSV parsing can change the in-memory representation of values even
        when the logical identity has not changed. In particular,
        ``source_identifier`` values that look numeric can be read as values
        such as ``1.0`` instead of the original string ``"1"``.

        The logical identity is defined by ``_KEY_COLUMNS`` and therefore must
        be normalized before key comparison, not only after concatenation.

        This method deliberately normalizes only identity columns. Full type
        and boolean normalization remains the responsibility of
        ``_normalise_output_dataframe``.
        """

        result = dataframe.loc[
            :,
            list(self._COLUMNS),
        ].copy()

        result["cadence_index"] = pd.to_numeric(
            result["cadence_index"],
            errors="raise",
        ).astype(
            "int64",
        )

        result["experiment_id"] = (
            result["experiment_id"]
            .astype(str)
            .str.strip()
        )

        result["source_identifier"] = (
            result["source_identifier"]
            .astype(str)
            .str.strip()
        )

        return result

    def _upsert_dataframes(
        self,
        *,
        existing: pd.DataFrame,
        incoming: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Replace existing logical keys with incoming records.

        Records are identified by:

            experiment_id
            cadence_index
            source_identifier

        Existing rows matching an incoming key are removed, then the
        incoming rows are appended.

        A plain Python boolean list is used for row selection rather than
        the NumPy boolean array returned by ``MultiIndex.isin()``. This
        keeps the operation compatible with strict pandas/Pylance typing.
        """

        existing_keys = pd.MultiIndex.from_frame(
            existing.loc[
                :,
                list(self._KEY_COLUMNS),
            ],
        )

        incoming_keys = pd.MultiIndex.from_frame(
            incoming.loc[
                :,
                list(self._KEY_COLUMNS),
            ],
        )

        incoming_key_set = set(
            incoming_keys.tolist(),
        )

        keep_mask: list[bool] = [
            key not in incoming_key_set
            for key in existing_keys.tolist()
        ]

        retained_indices: list[int] = [
            index
            for index, keep in enumerate(
                keep_mask,
            )
            if keep
        ]

        retained = existing.iloc[
            retained_indices
        ].copy()

        replacement_count = len(existing) - len(retained)

        if replacement_count:
            self._logger.info(
                (
                    "Time-series upsert replacing %d existing record(s)"
                    " for %d incoming logical key(s)."
                ),
                replacement_count,
                len(incoming_key_set),
            )

        return pd.concat(
            [
                retained,
                incoming,
            ],
            ignore_index=True,
        )

    def _normalise_output_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply deterministic column ordering, type normalization and
        record ordering.
        """

        result = dataframe.loc[
            :,
            list(self._COLUMNS),
        ].copy()

        result["cadence_index"] = pd.to_numeric(
            result["cadence_index"],
            errors="raise",
        ).astype(
            "int64",
        )

        result["experiment_id"] = (
            result["experiment_id"]
            .astype(str)
            .str.strip()
        )

        result["source_identifier"] = (
            result["source_identifier"]
            .astype(str)
            .str.strip()
        )

        result["valid"] = self._parse_boolean_column(
            result["valid"],
            column_name="valid",
        )

        result["is_target_source"] = (
            self._parse_boolean_column(
                result["is_target_source"],
                column_name="is_target_source",
            )
        )

        result["crowded"] = self._parse_boolean_column(
            result["crowded"],
            column_name="crowded",
        )

        duplicate_mask = result.duplicated(
            subset=list(
                self._KEY_COLUMNS,
            ),
            keep=False,
        )

        if bool(
            duplicate_mask.any()
        ):
            duplicate_rows = result.loc[
                duplicate_mask,
                list(
                    self._KEY_COLUMNS,
                ),
            ].drop_duplicates()

            duplicates: list[dict[str, object]] = []

            for row_values in duplicate_rows.itertuples(
                index=False,
                name=None,
            ):
                duplicates.append(
                    {
                        column: row_values[index]
                        for index, column in enumerate(
                            self._KEY_COLUMNS,
                        )
                    }
                )

            raise ValueError(
                (
                    "Time-series export contains duplicate logical "
                    f"keys: {duplicates}"
                ),
            )

        return (
            result
            .sort_values(
                by=[
                    "experiment_id",
                    "cadence_index",
                    "source_identifier",
                ],
                kind="stable",
            )
            .reset_index(
                drop=True,
            )
        )

    @staticmethod
    def _parse_boolean_column(
        series: pd.Series,
        *,
        column_name: str,
    ) -> pd.Series:
        """
        Parse common CSV boolean representations.
        """

        true_values = {
            "true",
            "1",
            "yes",
            "y",
        }

        false_values = {
            "false",
            "0",
            "no",
            "n",
        }

        normalised = (
            series
            .astype("string")
            .str.strip()
            .str.lower()
        )

        recognised = normalised.isin(
            true_values
            | false_values,
        )

        invalid_mask = (
            normalised.notna()
            & ~recognised
        )

        if bool(
            invalid_mask.any()
        ):
            invalid_values = sorted(
                normalised.loc[
                    invalid_mask,
                ]
                .dropna()
                .tolist(),
            )

            raise ValueError(
                (
                    f"Column '{column_name}' contains unrecognised "
                    f"boolean values: {invalid_values}"
                ),
            )

        return normalised.isin(
            true_values,
        )

    # ========================================================================
    # Persistence
    # ========================================================================

    def _persist_dataframe_atomically(
        self,
        *,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Persist a DataFrame atomically.

        The temporary file is created in the destination directory and
        replaced into place only after the complete CSV has been written.
        """

        parent = self._output_path.parent

        parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path: Path | None = None

        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=(
                    f".{self._output_path.stem}."
                ),
                suffix=".tmp",
                dir=parent,
                text=True,
            )

            os.close(
                file_descriptor,
            )

            temporary_path = Path(
                temporary_name,
            )

            dataframe.to_csv(
                temporary_path,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )

            os.replace(
                temporary_path,
                self._output_path,
            )

            temporary_path = None

        except (
            OSError,
            ValueError,
        ) as exc:
            raise OSError(
                (
                    "Failed to persist time-series export to "
                    f"'{self._output_path}'."
                ),
            ) from exc

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                try:
                    temporary_path.unlink()
                except OSError:
                    self._logger.warning(
                        (
                            "Unable to remove temporary time-series "
                            "export file '%s'."
                        ),
                        temporary_path,
                    )