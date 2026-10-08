"""
TESS differential-photometry time-series measurement writer.

Transforms completed differential-photometry time-series WorkflowResult
objects into immutable DifferentialTimeSeriesMeasurementRecord objects
and persists them as a CSV dataset.

Responsibilities
----------------
This module is responsible for:

    ExperimentConfig + WorkflowResult sequence
                |
                v
    DifferentialTimeSeriesMeasurementRecord sequence
                |
                v
    results/exports/differential_timeseries_measurements.csv

This module does NOT perform scientific calculations.

The differential-photometry calculation is completed by the workflow's
DifferentialPhotometer before the WorkflowResult reaches this export layer.

The writer therefore copies the already-computed values from
WorkflowResult.differential_photometry into the export model.

Persistence is implemented as a deterministic upsert.

The logical key for one differential time-series measurement is:

    experiment_id
    + cadence_index

When the same key already exists in the output CSV, the existing row is
replaced. Different experiments/cadences remain untouched.

The writer is intentionally independent of TESS/Lightkurve-specific types.
TESS cadence identity is read from the Observation raw-image metadata
populated by the TESS ObservationBuilder.

Reference-source identifiers and excluded-reference-source identifiers are
persisted as JSON arrays so that the CSV representation is deterministic
and unambiguous.

The differential result must remain within the workflow's authoritative
LocalField source population. Raw DAOStarFinder detections outside that
population are never treated as valid differential-photometry sources by
this export layer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from core.models import ExperimentConfig
from domain.models import Observation
from export.models import DifferentialTimeSeriesMeasurementRecord
from photometry.differential import DifferentialPhotometryResult
from utils.logger import get_logger
from workflows.models import WorkflowResult


class DifferentialTimeSeriesWriter:
    """
    Persist cadence-level differential-photometry measurements.

    Parameters
    ----------
    output_path
        Destination CSV path.

    logger
        Optional logger. If omitted, the project logger is used.

    Notes
    -----
    The writer performs no scientific calculations.

    Differential flux, differential magnitude, target net flux and reference
    ensemble flux are copied directly from the completed
    DifferentialPhotometryResult.

    Existing rows matching the logical differential time-series record key
    are replaced rather than duplicated.
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
        "target_source_identifier",
        "reference_source_identifiers",
        "excluded_reference_source_identifiers",
        "target_net_flux",
        "reference_net_fluxes",
        "reference_flux",
        "differential_flux",
        "differential_magnitude",
    )

    _KEY_COLUMNS: tuple[str, ...] = (
        "experiment_id",
        "cadence_index",
    )

    def __init__(
        self,
        *,
        output_path: Path | str = (
            Path("results")
            / "exports"
            / "differential_timeseries_measurements.csv"
        ),
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the differential time-series writer.
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
    ) -> tuple[DifferentialTimeSeriesMeasurementRecord, ...]:
        """
        Transform completed workflow results into immutable differential
        time-series export records.

        Parameters
        ----------
        config
            Experiment configuration used to create the workflow.

        results
            Ordered sequence of completed WorkflowResult objects.

        Returns
        -------
        tuple[DifferentialTimeSeriesMeasurementRecord, ...]
            One record for each completed differential-photometry cadence.

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
                "Differential time-series workflow results cannot be None.",
            )

        result_sequence = tuple(
            results,
        )

        if not result_sequence:
            raise ValueError(
                "Differential time-series workflow results cannot be empty.",
            )

        records: list[
            DifferentialTimeSeriesMeasurementRecord
        ] = []

        seen_keys: set[
            tuple[str, int]
        ] = set()

        experiment_id = self._build_experiment_id(
            config=config,
            first_result=result_sequence[0],
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
                        "Differential time-series result at sequence "
                        "position "
                        f"{sequence_position} is not a WorkflowResult."
                    ),
                )

            record = self._build_record_for_result(
                config=config,
                experiment_id=experiment_id,
                result=result,
            )

            key = self._record_key(
                record,
            )

            if key in seen_keys:
                raise ValueError(
                    (
                        "Duplicate differential time-series record key "
                        "detected within the supplied workflow results: "
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
                (
                    "No differential time-series measurement records "
                    "were generated."
                ),
            )

        self._validate_record_sequence(
            records,
        )

        self._logger.info(
            (
                "Built %d differential time-series measurement record(s) "
                "for experiment '%s'."
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
    ) -> tuple[DifferentialTimeSeriesMeasurementRecord, ...]:
        """
        Transform and persist completed differential time-series results.

        Existing records with matching logical keys are replaced.

        Parameters
        ----------
        config
            Experiment configuration used to create the workflow.

        results
            Ordered sequence of completed WorkflowResult objects.

        Returns
        -------
        tuple[DifferentialTimeSeriesMeasurementRecord, ...]
            Records written for the supplied results.
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
        records: Sequence[DifferentialTimeSeriesMeasurementRecord],
    ) -> None:
        """
        Persist differential time-series records using deterministic upsert.

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
                "Differential time-series records cannot be None.",
            )

        record_sequence = tuple(
            records,
        )

        if not record_sequence:
            raise ValueError(
                "Differential time-series records cannot be empty.",
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
                "Differential time-series export persisted successfully: "
                "%d record(s) written."
            ),
            len(record_sequence),
        )

        self._logger.info(
            "Differential time-series export: %s",
            self._output_path,
        )

    # ========================================================================
    # Result transformation
    # ========================================================================

    def _build_record_for_result(
        self,
        *,
        config: ExperimentConfig,
        experiment_id: str,
        result: WorkflowResult,
    ) -> DifferentialTimeSeriesMeasurementRecord:
        """
        Convert one cadence-level WorkflowResult into one differential
        time-series export record.
        """

        observation = result.observation

        if observation is None:
            raise ValueError(
                (
                    "Differential time-series workflow result does not "
                    "contain an Observation."
                ),
            )

        differential_result = result.differential_photometry

        if differential_result is None:
            raise ValueError(
                (
                    "Differential time-series workflow result does not "
                    "contain a completed differential-photometry result."
                ),
            )

        if not isinstance(
            differential_result,
            DifferentialPhotometryResult,
        ):
            raise ValueError(
                (
                    "WorkflowResult.differential_photometry must be a "
                    "DifferentialPhotometryResult."
                ),
            )

        cadence_index = self._extract_cadence_index(
            observation,
        )

        self._validate_result_consistency(
            config=config,
            result=result,
            differential_result=differential_result,
            cadence_index=cadence_index,
        )

        self._validate_local_field_source_population(
            result=result,
            differential_result=differential_result,
        )

        return self._create_record(
            config=config,
            experiment_id=experiment_id,
            observation=observation,
            result=differential_result,
            cadence_index=cadence_index,
        )

    def _create_record(
        self,
        *,
        config: ExperimentConfig,
        experiment_id: str,
        observation: Observation,
        result: DifferentialPhotometryResult,
        cadence_index: int,
    ) -> DifferentialTimeSeriesMeasurementRecord:
        """
        Create one immutable differential time-series export record.

        No scientific quantities are calculated here.
        """

        return DifferentialTimeSeriesMeasurementRecord(
            experiment_id=experiment_id,
            target_name=config.target.target_name,
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
            target_source_identifier=(
                result.target_source_identifier
            ),
            reference_source_identifiers=(
                result.reference_source_identifiers
            ),
            excluded_reference_source_identifiers=(
                result.excluded_reference_source_identifiers
            ),
            target_net_flux=float(
                result.target_net_flux,
            ),
            reference_net_fluxes=(
                result.reference_net_fluxes
            ),
            reference_flux=float(
                result.reference_flux,
            ),
            differential_flux=float(
                result.differential_flux,
            ),
            differential_magnitude=float(
                result.differential_magnitude,
            ),
        )

    # ========================================================================
    # Result validation
    # ========================================================================

    @staticmethod
    def _validate_local_field_source_population(
        *,
        result: WorkflowResult,
        differential_result: DifferentialPhotometryResult,
    ) -> None:
        """
        Validate that differential sources belong to the authoritative
        LocalField population.

        The workflow's resolved target and reference matches define the
        source population that participated in the differential-photometry
        workflow. This export layer must not silently accept identifiers
        belonging only to the broader DAOStarFinder detection population.

        The differential result may contain:
            * the target source,
            * included reference sources,
            * excluded reference sources.

        All of those identifiers must resolve to the authoritative
        LocalField population. Excluded references are retained in the
        export for provenance and therefore must be validated as well.

        No scientific calculation is performed here.
        """

        if result.target_match is None:
            raise ValueError(
                (
                    "Cannot validate differential source population "
                    "without a resolved target source."
                ),
            )

        local_field_source_identifiers = {
            result.target_match.source_identifier,
            *(
                match.source_identifier
                for match in result.reference_matches
            ),
        }

        if not local_field_source_identifiers:
            raise ValueError(
                (
                    "The authoritative LocalField source population is "
                    "empty; differential export cannot proceed."
                ),
            )

        exported_source_identifiers = {
            differential_result.target_source_identifier,
            *differential_result.reference_source_identifiers,
            *differential_result.excluded_reference_source_identifiers,
        }

        missing_from_local_field = (
            exported_source_identifiers
            - local_field_source_identifiers
        )

        if missing_from_local_field:
            raise ValueError(
                (
                    "Differential-photometry result references source(s) "
                    "outside the authoritative LocalField population: "
                    f"{sorted(missing_from_local_field)}."
                ),
            )

    @staticmethod
    def _validate_result_consistency(
        *,
        config: ExperimentConfig,
        result: WorkflowResult,
        differential_result: DifferentialPhotometryResult,
        cadence_index: int,
    ) -> None:
        """
        Validate structural consistency between the workflow result,
        configuration and completed differential result.

        This method performs no scientific calculation.
        """

        if config.differential_photometry is None:
            raise ValueError(
                (
                    "Differential-photometry export requires "
                    "ExperimentConfig.differential_photometry."
                ),
            )

        if result.target_match is None:
            raise ValueError(
                (
                    "Differential time-series result requires a resolved "
                    "target source."
                ),
            )

        if not result.reference_matches:
            raise ValueError(
                (
                    "Differential time-series result requires at least "
                    "one resolved reference source."
                ),
            )

        if not differential_result.target_source_identifier:
            raise ValueError(
                (
                    "Differential-photometry result contains an empty "
                    "target source identifier."
                ),
            )

        if (
            differential_result.target_source_identifier
            != result.target_match.source_identifier
        ):
            raise ValueError(
                (
                    "Differential-photometry target source identifier "
                    "does not match the WorkflowResult target match: "
                    f"'{differential_result.target_source_identifier}' "
                    "!="
                    f"'{result.target_match.source_identifier}'."
                ),
            )

        if cadence_index < 0:
            raise ValueError(
                (
                    "Differential time-series cadence index cannot be "
                    f"negative: {cadence_index}."
                ),
            )

        configured_reference_count = len(
            config.differential_photometry.reference_stars,
        )

        resolved_reference_count = len(
            result.reference_matches,
        )

        if resolved_reference_count > configured_reference_count:
            raise ValueError(
                (
                    "Workflow resolved more reference sources than were "
                    "configured: "
                    f"{resolved_reference_count} > "
                    f"{configured_reference_count}."
                ),
            )

        if (
            len(
                differential_result.reference_source_identifiers,
            )
            != len(
                differential_result.reference_net_fluxes,
            )
        ):
            raise ValueError(
                (
                    "Differential-photometry result contains "
                    "mismatched reference identifier and flux counts."
                ),
            )

        if not differential_result.reference_source_identifiers:
            raise ValueError(
                (
                    "Differential-photometry result contains no valid "
                    "reference sources."
                ),
            )

        reference_identifiers = (
            differential_result.reference_source_identifiers
        )

        resolved_reference_identifiers = {
            match.source_identifier
            for match in result.reference_matches
        }

        unknown_reference_identifiers = (
            set(reference_identifiers)
            - resolved_reference_identifiers
        )

        if unknown_reference_identifiers:
            raise ValueError(
                (
                    "Differential-photometry result contains reference "
                    "source(s) that were not resolved by the workflow: "
                    f"{sorted(unknown_reference_identifiers)}."
                ),
            )

        excluded_identifiers = (
            differential_result.excluded_reference_source_identifiers
        )

        unknown_excluded_identifiers = (
            set(excluded_identifiers)
            - resolved_reference_identifiers
        )

        if unknown_excluded_identifiers:
            raise ValueError(
                (
                    "Differential-photometry result contains excluded "
                    "reference source(s) that were not resolved by the "
                    "workflow: "
                    f"{sorted(unknown_excluded_identifiers)}."
                ),
            )

        if len(reference_identifiers) != len(
            set(reference_identifiers),
        ):
            raise ValueError(
                (
                    "Differential-photometry result contains duplicate "
                    "reference source identifiers."
                ),
            )

        excluded_identifiers = (
            differential_result.excluded_reference_source_identifiers
        )

        if len(excluded_identifiers) != len(
            set(excluded_identifiers),
        ):
            raise ValueError(
                (
                    "Differential-photometry result contains duplicate "
                    "excluded reference source identifiers."
                ),
            )

        if differential_result.target_source_identifier in (
            set(reference_identifiers)
            | set(excluded_identifiers)
        ):
            raise ValueError(
                (
                    "Differential-photometry result contains the target "
                    "source in its reference-source population: "
                    f"'{differential_result.target_source_identifier}'."
                ),
            )

        overlap = (
            set(reference_identifiers)
            & set(excluded_identifiers)
        )

        if overlap:
            raise ValueError(
                (
                    "Differential-photometry result contains reference "
                    "sources that are both included and excluded: "
                    f"{sorted(overlap)}."
                ),
            )

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
        the existing TimeSeriesWriter and ExperimentExporter.

        The ID intentionally uses stable experiment/dataset provenance
        rather than wall-clock time.
        """

        observation = first_result.observation

        if observation is None:
            raise ValueError(
                (
                    "Cannot build differential time-series experiment "
                    "ID without an Observation."
                ),
            )

        provenance = observation.provenance

        components = (
            provenance.survey.value,
            provenance.dataset_identifier,
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
    # Cadence extraction
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

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
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
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_config(
        config: ExperimentConfig,
    ) -> None:
        """
        Validate the configuration required for differential export.
        """

        if config is None:
            raise ValueError(
                "Experiment configuration cannot be None.",
            )

        if not config.target.target_name:
            raise ValueError(
                "Experiment configuration must contain a target name.",
            )

        if config.differential_photometry is None:
            raise ValueError(
                (
                    "Differential time-series export requires a configured "
                    "differential_photometry section."
                ),
            )

        if not config.differential_photometry.reference_stars:
            raise ValueError(
                (
                    "Differential time-series export requires at least "
                    "one configured reference star."
                ),
            )

        if not config.time_series.enabled:
            raise ValueError(
                (
                    "Differential time-series export requires "
                    "time_series.enabled=True."
                ),
            )

    def _validate_record_sequence(
        self,
        records: Sequence[DifferentialTimeSeriesMeasurementRecord],
    ) -> None:
        """
        Validate record identity and sequence integrity.
        """

        seen_keys: set[
            tuple[str, int]
        ] = set()

        for record in records:
            if not isinstance(
                record,
                DifferentialTimeSeriesMeasurementRecord,
            ):
                raise ValueError(
                    (
                        "Differential time-series record must be a "
                        "DifferentialTimeSeriesMeasurementRecord."
                    ),
                )

            key = self._record_key(
                record,
            )

            if key in seen_keys:
                raise ValueError(
                    (
                        "Duplicate differential time-series record key "
                        f"found: {key}."
                    ),
                )

            seen_keys.add(
                key,
            )

            if not record.experiment_id:
                raise ValueError(
                    (
                        "Differential time-series record has an empty "
                        "experiment ID."
                    ),
                )

            if record.cadence_index < 0:
                raise ValueError(
                    (
                        "Differential time-series cadence index cannot "
                        f"be negative: {record.cadence_index}."
                    ),
                )

            if not record.target_source_identifier:
                raise ValueError(
                    (
                        "Differential time-series record has an empty "
                        "target source identifier."
                    ),
                )

            if not record.reference_source_identifiers:
                raise ValueError(
                    (
                        "Differential time-series record has no valid "
                        "reference source identifiers."
                    ),
                )

            if len(
                record.reference_source_identifiers,
            ) != len(
                record.reference_net_fluxes,
            ):
                raise ValueError(
                    (
                        "Differential time-series record has mismatched "
                        "reference identifier and flux counts."
                    ),
                )

    @staticmethod
    def _record_key(
        record: DifferentialTimeSeriesMeasurementRecord,
    ) -> tuple[str, int]:
        """
        Return the logical identity of one differential time-series record.
        """

        return (
            record.experiment_id,
            record.cadence_index,
        )

    # ========================================================================
    # DataFrame conversion
    # ========================================================================

    def _records_to_dataframe(
        self,
        records: Sequence[DifferentialTimeSeriesMeasurementRecord],
    ) -> pd.DataFrame:
        """
        Convert immutable export records to a DataFrame.

        Tuple-valued reference fields are serialized as JSON arrays so that
        the CSV representation is deterministic and can be parsed without
        ambiguity.
        """

        rows: list[dict[str, object]] = []

        for record in records:
            rows.append(
                {
                    "experiment_id": record.experiment_id,
                    "target_name": record.target_name,
                    "survey": record.survey,
                    "experiment_type": record.experiment_type,
                    "pipeline": record.pipeline,
                    "cadence_index": record.cadence_index,
                    "observation_time": record.observation_time,
                    "observation_time_scale": (
                        record.observation_time_scale
                    ),
                    "observation_time_format": (
                        record.observation_time_format
                    ),
                    "target_source_identifier": (
                        record.target_source_identifier
                    ),
                    "reference_source_identifiers": (
                        self._serialise_string_tuple(
                            record.reference_source_identifiers,
                        )
                    ),
                    "excluded_reference_source_identifiers": (
                        self._serialise_string_tuple(
                            record.excluded_reference_source_identifiers,
                        )
                    ),
                    "target_net_flux": record.target_net_flux,
                    "reference_net_fluxes": (
                        self._serialise_float_tuple(
                            record.reference_net_fluxes,
                        )
                    ),
                    "reference_flux": record.reference_flux,
                    "differential_flux": record.differential_flux,
                    "differential_magnitude": (
                        record.differential_magnitude
                    ),
                }
            )

        dataframe = pd.DataFrame(
            rows,
            columns=self._COLUMNS,
        )

        if dataframe.empty:
            raise ValueError(
                (
                    "Cannot create differential time-series export "
                    "from zero records."
                ),
            )

        return dataframe

    @staticmethod
    def _serialise_string_tuple(
        values: tuple[str, ...],
    ) -> str:
        """
        Serialize a tuple of strings as a deterministic JSON array.
        """

        return json.dumps(
            list(values),
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    @staticmethod
    def _serialise_float_tuple(
        values: tuple[float, ...],
    ) -> str:
        """
        Serialize a tuple of floating-point values as a deterministic
        JSON array.
        """

        return json.dumps(
            [
                float(value)
                for value in values
            ],
            separators=(
                ",",
                ":",
            ),
        )

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
                    "Unable to read existing differential time-series "
                    "export: "
                    f"{self._output_path}"
                ),
            ) from exc

        missing_columns = [
            column
            for column in self._COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                (
                    "Existing differential time-series export is missing "
                    f"required columns: {missing_columns}"
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
        Normalize logical-key columns used for deterministic upsert.
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

        return result

    def _upsert_dataframes(
        self,
        *,
        existing: pd.DataFrame,
        incoming: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Replace existing logical keys with incoming records.

        Existing rows matching an incoming key are removed, then incoming
        rows are appended.
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

        replacement_count = (
            len(existing)
            - len(retained)
        )

        if replacement_count:
            self._logger.info(
                (
                    "Differential time-series upsert replacing "
                    "%d existing record(s) for %d incoming "
                    "logical key(s)."
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

        result["target_name"] = (
            result["target_name"]
            .astype(str)
            .str.strip()
        )

        result["survey"] = (
            result["survey"]
            .astype(str)
            .str.strip()
        )

        result["experiment_type"] = (
            result["experiment_type"]
            .astype(str)
            .str.strip()
        )

        result["pipeline"] = (
            result["pipeline"]
            .astype(str)
            .str.strip()
        )

        result["target_source_identifier"] = (
            result["target_source_identifier"]
            .astype(str)
            .str.strip()
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
                    "Differential time-series export contains duplicate "
                    f"logical keys: {duplicates}"
                ),
            )

        return (
            result
            .sort_values(
                by=[
                    "experiment_id",
                    "cadence_index",
                ],
                kind="stable",
            )
            .reset_index(
                drop=True,
            )
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
                    "Failed to persist differential time-series export "
                    f"to '{self._output_path}'."
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
                            "Unable to remove temporary differential "
                            "time-series export file '%s'."
                        ),
                        temporary_path,
                    )