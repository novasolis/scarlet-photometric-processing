"""
Persistent storage for photometric experiment export records.

This module is responsible only for persisting already-transformed
export records produced by :class:`ExperimentExporter`.

The persistence boundary is:

    ExperimentSummary
    SourceMeasurementRecord
            |
            v
        ResultWriter
            |
            v
    results/exports/*.csv

No scientific calculations are performed here.

No workflow, acquisition, preprocessing, detection, photometry or
evaluation logic belongs in this module.

The writer uses deterministic upsert semantics:

    experiment_summary.csv
        experiment_id is the unique experiment key.

    source_measurements.csv
        (experiment_id, source_identifier) is the unique source key.

Re-running an experiment therefore replaces its previous exported
records rather than creating duplicate rows.

The implementation uses only the Python standard library.
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeVar

from export.models import (
    ExperimentSummary,
    SourceMeasurementRecord,
)
from utils.logger import get_logger


ExportRecord = TypeVar(
    "ExportRecord",
    ExperimentSummary,
    SourceMeasurementRecord,
)


class ExportWriter:
    """
    Persist photometric experiment export records as CSV files.

    Parameters
    ----------
    output_directory
        Directory in which the export files are stored.

        Defaults to::

            results/exports

    Raises
    ------
    ValueError
        If the output directory is invalid.
    """

    _SUMMARY_FILENAME = "experiment_summary.csv"
    _SOURCE_FILENAME = "source_measurements.csv"

    def __init__(
        self,
        *,
        output_directory: Path | str = Path("results") / "exports",
    ) -> None:
        if output_directory is None:
            raise ValueError(
                "Export output directory cannot be None."
            )

        self._output_directory = Path(
            output_directory,
        )

        self._validate_output_directory()

        self._logger = get_logger(
            __name__,
        )

    # ==================================================================
    # Public properties
    # ==================================================================

    @property
    def output_directory(self) -> Path:
        """
        Return the configured export directory.
        """

        return self._output_directory

    @property
    def summary_path(self) -> Path:
        """
        Return the experiment-summary CSV path.
        """

        return (
            self._output_directory
            / self._SUMMARY_FILENAME
        )

    @property
    def source_measurements_path(self) -> Path:
        """
        Return the source-measurement CSV path.
        """

        return (
            self._output_directory
            / self._SOURCE_FILENAME
        )

    # ==================================================================
    # Public write API
    # ==================================================================

    def write(
        self,
        *,
        summary: ExperimentSummary,
        source_records: tuple[
            SourceMeasurementRecord,
            ...,
        ],
    ) -> None:
        """
        Persist one completed experiment.

        Existing records for the same experiment are replaced.

        Parameters
        ----------
        summary
            Experiment-level export record.

        source_records
            Source-level records belonging to the same experiment.

        Raises
        ------
        ValueError
            If the supplied records are invalid or inconsistent.
        OSError
            If the result files cannot be written.
        """

        self._validate_records(
            summary=summary,
            source_records=source_records,
        )

        self._output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._logger.info(
            "Persisting experiment export '%s'.",
            summary.experiment_id,
        )

        self._upsert_summary(
            summary,
        )

        self._upsert_source_measurements(
            source_records,
        )

        self._logger.info(
            "Experiment export persisted successfully.",
        )

        self._logger.info(
            "Experiment summary: %s",
            self.summary_path,
        )

        self._logger.info(
            "Source measurements: %s",
            self.source_measurements_path,
        )

    # ==================================================================
    # Experiment summary persistence
    # ==================================================================

    def _upsert_summary(
        self,
        summary: ExperimentSummary,
    ) -> None:
        """
        Insert or replace one experiment summary.

        The experiment ID is the unique key.
        """

        rows = self._read_csv(
            self.summary_path,
        )

        new_row = self._record_to_row(
            summary,
        )

        experiment_id = summary.experiment_id

        rows = [
            row
            for row in rows
            if row.get("experiment_id") != experiment_id
        ]

        rows.append(
            new_row,
        )

        field_names = self._dataclass_field_names(
            ExperimentSummary,
        )

        self._write_csv_atomic(
            path=self.summary_path,
            field_names=field_names,
            rows=rows,
        )

    # ==================================================================
    # Source measurement persistence
    # ==================================================================

    def _upsert_source_measurements(
        self,
        source_records: tuple[
            SourceMeasurementRecord,
            ...,
        ],
    ) -> None:
        """
        Insert or replace source records belonging to one experiment.

        The source key is:

            (experiment_id, source_identifier)

        All existing records for the affected experiment are removed
        before the new source records are inserted. This is intentional:
        a rerun may change detection results, so stale source rows must
        never survive.
        """

        if not source_records:
            return

        experiment_ids = {
            record.experiment_id
            for record in source_records
        }

        if len(experiment_ids) != 1:
            raise ValueError(
                "Source measurement records supplied to one write "
                "operation must belong to exactly one experiment."
            )

        experiment_id = next(
            iter(experiment_ids),
        )

        rows = self._read_csv(
            self.source_measurements_path,
        )

        rows = [
            row
            for row in rows
            if row.get("experiment_id") != experiment_id
        ]

        existing_source_keys = {
            (
                row.get("experiment_id"),
                row.get("source_identifier"),
            )
            for row in rows
        }

        new_rows: list[dict[str, str]] = []

        for record in source_records:

            source_key = (
                record.experiment_id,
                record.source_identifier,
            )

            if source_key in existing_source_keys:
                raise ValueError(
                    "Duplicate source measurement key detected "
                    f"during export: experiment="
                    f"'{record.experiment_id}', source="
                    f"'{record.source_identifier}'."
                )

            existing_source_keys.add(
                source_key,
            )

            new_rows.append(
                self._record_to_row(
                    record,
                ),
            )

        rows.extend(
            new_rows,
        )

        field_names = self._dataclass_field_names(
            SourceMeasurementRecord,
        )

        self._write_csv_atomic(
            path=self.source_measurements_path,
            field_names=field_names,
            rows=rows,
        )

    # ==================================================================
    # CSV reading
    # ==================================================================

    @staticmethod
    def _read_csv(
        path: Path,
    ) -> list[dict[str, str]]:
        """
        Read an existing CSV file.

        A missing file is treated as an empty dataset.

        Existing CSV headers are validated against the expected schema
        by the caller when a write is performed.
        """

        if not path.exists():
            return []

        if not path.is_file():
            raise OSError(
                f"Export path exists but is not a file: {path}"
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:

                reader = csv.DictReader(
                    handle,
                )

                if reader.fieldnames is None:
                    raise ValueError(
                        f"Export CSV contains no header: {path}"
                    )

                return [
                    {
                        key: value
                        for key, value in row.items()
                        if key is not None
                    }
                    for row in reader
                ]

        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Export CSV is not valid UTF-8: {path}"
            ) from exc

    # ==================================================================
    # CSV writing
    # ==================================================================

    @classmethod
    def _write_csv_atomic(
        cls,
        *,
        path: Path,
        field_names: tuple[str, ...],
        rows: list[dict[str, str]],
    ) -> None:
        """
        Atomically replace a CSV file.

        The temporary file is written in the destination directory so
        that os.replace() remains on the same filesystem.
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing_field_names = cls._existing_field_names(
            path,
        )

        if (
            existing_field_names is not None
            and existing_field_names != field_names
        ):
            raise ValueError(
                "Existing export schema does not match the current "
                f"model schema for '{path.name}'. "
                f"Existing columns: {existing_field_names}. "
                f"Expected columns: {field_names}."
            )

        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:

                temp_path = Path(
                    handle.name,
                )

                writer = csv.DictWriter(
                    handle,
                    fieldnames=field_names,
                    extrasaction="raise",
                    lineterminator="\n",
                )

                writer.writeheader()

                for row in rows:
                    writer.writerow(
                        {
                            field: row.get(
                                field,
                                "",
                            )
                            for field in field_names
                        }
                    )

                handle.flush()
                os.fsync(
                    handle.fileno(),
                )

            os.replace(
                temp_path,
                path,
            )

            temp_path = None

        finally:
            if (
                temp_path is not None
                and temp_path.exists()
            ):
                try:
                    temp_path.unlink()
                except OSError:
                    # The original write error remains the primary
                    # exception. Cleanup failure must not obscure it.
                    pass

    @staticmethod
    def _existing_field_names(
        path: Path,
    ) -> tuple[str, ...] | None:
        """
        Read the existing CSV header without loading the whole file.
        """

        if not path.exists():
            return None

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:

            reader = csv.reader(
                handle,
            )

            try:
                header = next(
                    reader,
                )
            except StopIteration:
                return None

        return tuple(
            header,
        )

    # ==================================================================
    # Dataclass conversion
    # ==================================================================

    @classmethod
    def _record_to_row(
        cls,
        record: ExportRecord,
    ) -> dict[str, str]:
        """
        Convert an export dataclass into CSV-safe string values.
        """

        if not is_dataclass(record):
            raise ValueError(
                "Export record must be a dataclass instance."
            )

        return {
            field.name: cls._serialize_value(
                getattr(
                    record,
                    field.name,
                )
            )
            for field in fields(
                record,
            )
        }

    @staticmethod
    def _serialize_value(
        value: Any,
    ) -> str:
        """
        Convert one dataclass value into a stable CSV representation.

        ``None`` is represented by an empty field.

        datetime/date values are represented using ISO-8601.

        Boolean values use explicit lowercase textual values so they are
        straightforward to parse in pandas and other analysis tools.
        """

        if value is None:
            return ""

        if isinstance(
            value,
            datetime,
        ):
            return value.isoformat()

        if isinstance(
            value,
            date,
        ):
            return value.isoformat()

        if isinstance(
            value,
            bool,
        ):
            return "true" if value else "false"

        return str(
            value,
        )

    @staticmethod
    def _dataclass_field_names(
        record_type: type[ExportRecord],
    ) -> tuple[str, ...]:
        """
        Return the declared dataclass field order.

        Using the model itself as the schema source prevents the writer
        from silently becoming inconsistent with export/models.py.
        """

        if not is_dataclass(
            record_type,
        ):
            raise ValueError(
                "Export record type must be a dataclass."
            )

        return tuple(
            field.name
            for field in fields(
                record_type,
            )
        )

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _validate_records(
        *,
        summary: ExperimentSummary,
        source_records: tuple[
            SourceMeasurementRecord,
            ...,
        ],
    ) -> None:
        """
        Validate the export records before persistence.
        """

        if summary is None:
            raise ValueError(
                "Experiment summary cannot be None."
            )

        if not summary.experiment_id:
            raise ValueError(
                "Experiment summary must contain an experiment ID."
            )

        if not summary.target_name:
            raise ValueError(
                "Experiment summary must contain a target name."
            )

        if not summary.survey:
            raise ValueError(
                "Experiment summary must contain a survey."
            )

        if not summary.pipeline:
            raise ValueError(
                "Experiment summary must contain a pipeline."
            )

        if source_records is None:
            raise ValueError(
                "Source measurement records cannot be None."
            )

        seen_source_identifiers: set[str] = set()

        for record in source_records:

            if not record.experiment_id:
                raise ValueError(
                    "Source measurement record must contain an "
                    "experiment ID."
                )

            if record.experiment_id != summary.experiment_id:
                raise ValueError(
                    "Source measurement record belongs to a different "
                    "experiment than the supplied summary: "
                    f"'{record.experiment_id}' != "
                    f"'{summary.experiment_id}'."
                )

            if not record.source_identifier:
                raise ValueError(
                    "Source measurement record must contain a source "
                    "identifier."
                )

            if record.source_identifier in seen_source_identifiers:
                raise ValueError(
                    "Duplicate source measurement record detected "
                    f"for source '{record.source_identifier}'."
                )

            seen_source_identifiers.add(
                record.source_identifier,
            )

        if len(source_records) != summary.detected_source_count:
            raise ValueError(
                "Source measurement record count does not match the "
                "experiment summary: "
                f"{len(source_records)} record(s) for "
                f"{summary.detected_source_count} detected source(s)."
            )

    def _validate_output_directory(self) -> None:
        """
        Validate the configured output path.
        """

        if (
            self._output_directory.exists()
            and not self._output_directory.is_dir()
        ):
            raise ValueError(
                "Export output path exists but is not a directory: "
                f"{self._output_directory}"
            )