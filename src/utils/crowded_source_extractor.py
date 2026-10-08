"""
Crowded-source extraction utility.

Provides a production-safe way to extract detected sources that were
classified as crowded by the completed photometric evaluation.

This module performs NO scientific processing and does NOT recalculate
crowding.

The PhotometryEvaluator is the authoritative component for:

    * neighbour counting;
    * crowding-radius application;
    * crowded-source classification;
    * source-level photometric quality evaluation.

The extractor only joins:

    PhotometryEvaluation.source_evaluations
        +
    Observation.detections.sources

to produce immutable records containing:

    * source identifier;
    * X pixel coordinate;
    * Y pixel coordinate;
    * neighbour count;
    * crowded classification;
    * photometric validity;
    * quality flag;
    * peak value.

This utility is intentionally survey-independent.

It is useful for:

    * identifying crowded sources for follow-up experiments;
    * selecting targets for detailed photometric investigation;
    * producing thesis tables;
    * exporting crowded-source coordinates;
    * validating visualisations.

Scientific processing remains entirely within the production workflow.

Expected usage
--------------

    result = workflow.run()

    crowded_sources = CrowdedSourceExtractor(
        logger,
    ).extract(
        observation=result.observation,
        evaluation=result.evaluation,
    )

    for source in crowded_sources:
        print(
            source.source_identifier,
            source.x_pixel,
            source.y_pixel,
        )

For an M67 crowded-field experiment, this can be used to identify
specific crowded sources for subsequent target-specific experiments.

The actual identifiers and coordinates depend entirely on the
completed production workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from math import isfinite

from domain.models import (
    DetectedSource,
    Observation,
)
from evaluation.models import (
    PhotometryEvaluation,
    SourceEvaluation,
)
from utils.logger import get_logger


@dataclass(frozen=True, slots=True)
class CrowdedSourceRecord:
    """
    Immutable record describing one source classified as crowded.

    Parameters
    ----------
    source_identifier
        Identifier assigned to the source by the source-detection stage.

    x_pixel
        X-coordinate of the detected source centroid in pixel
        coordinates.

    y_pixel
        Y-coordinate of the detected source centroid in pixel
        coordinates.

    neighbour_count
        Number of neighbouring detected sources within the configured
        crowding radius, as determined by PhotometryEvaluator.

    crowded
        Crowded classification produced by PhotometryEvaluator.

    valid_measurement
        Whether the source's photometric measurement passed the
        configured photometric quality criteria.

    quality_flag
        Machine-readable photometric quality classification.

    peak_value
        Peak pixel value associated with the detected source.
    """

    source_identifier: str

    x_pixel: float

    y_pixel: float

    neighbour_count: int

    crowded: bool

    valid_measurement: bool

    quality_flag: str

    peak_value: float


class CrowdedSourceExtractor:
    """
    Extract crowded sources from a completed production workflow result.

    The extractor is deliberately read-only and survey-independent.

    It does not:

        * perform source detection;
        * perform photometry;
        * calculate neighbour distances;
        * determine the crowding radius;
        * classify sources;
        * modify the Observation;
        * modify the PhotometryEvaluation.

    Crowding classification is taken directly from the completed
    PhotometryEvaluation produced by the production evaluator.
    """

    def __init__(
        self,
        logger: Logger | None = None,
    ) -> None:
        """
        Initialise the crowded-source extractor.

        Parameters
        ----------
        logger
            Optional logger used for diagnostic messages.
        """

        self._logger = logger or get_logger(
            __name__,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        *,
        observation: Observation,
        evaluation: PhotometryEvaluation,
    ) -> tuple[CrowdedSourceRecord, ...]:
        """
        Extract all sources classified as crowded.

        Parameters
        ----------
        observation
            Completed Observation containing the authoritative detected
            source records.

        evaluation
            Completed PhotometryEvaluation containing the authoritative
            source-level crowding classifications.

        Returns
        -------
        tuple[CrowdedSourceRecord, ...]
            Immutable records for all sources classified as crowded.

            Records retain the original detected-source ordering.

        Raises
        ------
        ValueError
            If required workflow products are missing or internally
            inconsistent.

        Notes
        -----
        This method does not recalculate crowding.

        The ``crowded`` and ``neighbour_count`` values are taken directly
        from PhotometryEvaluation.

        Source coordinates are taken from Observation.detections because
        DetectedSource is the authoritative source-location model.
        """

        self._validate_inputs(
            observation=observation,
            evaluation=evaluation,
        )

        #
        # _validate_inputs guarantees that detections exists.
        #
        assert observation.detections is not None

        detected_sources: tuple[DetectedSource, ...] = (
            observation.detections.sources
        )

        evaluation_by_source = (
            self._build_evaluation_lookup(
                evaluation,
            )
        )

        self._validate_evaluation_completeness(
            detected_sources=detected_sources,
            evaluation_by_source=evaluation_by_source,
        )

        crowded_sources: list[CrowdedSourceRecord] = []

        #
        # Preserve the original detection ordering.
        #
        for source in detected_sources:

            source_evaluation: SourceEvaluation = (
                evaluation_by_source[
                    source.identifier
                ]
            )

            #
            # Crowding classification is owned by
            # PhotometryEvaluator.
            #
            if not source_evaluation.crowded:
                continue

            record = CrowdedSourceRecord(
                source_identifier=source.identifier,

                x_pixel=float(
                    source.x_centroid,
                ),

                y_pixel=float(
                    source.y_centroid,
                ),

                neighbour_count=int(
                    source_evaluation.neighbour_count,
                ),

                crowded=bool(
                    source_evaluation.crowded,
                ),

                valid_measurement=bool(
                    source_evaluation.valid_measurement,
                ),

                quality_flag=str(
                    source_evaluation.quality_flag,
                ),

                peak_value=float(
                    source.peak_value,
                ),
            )

            self._validate_record(
                record,
            )

            crowded_sources.append(
                record,
            )

        self._logger.info(
            "Crowded-source extraction completed for '%s': "
            "%d crowded source(s) identified.",
            observation.metadata.target_name,
            len(crowded_sources),
        )

        return tuple(
            crowded_sources,
        )

    def report(
        self,
        *,
        observation: Observation,
        evaluation: PhotometryEvaluation,
    ) -> tuple[CrowdedSourceRecord, ...]:
        """
        Extract and log a human-readable crowded-source report.

        Parameters
        ----------
        observation
            Completed Observation.

        evaluation
            Completed PhotometryEvaluation.

        Returns
        -------
        tuple[CrowdedSourceRecord, ...]
            Same records returned by :meth:`extract`.

        Notes
        -----
        This method is intended for demonstration scripts, development
        diagnostics and thesis experiment setup.

        It does not alter the underlying scientific results.
        """

        crowded_sources = self.extract(
            observation=observation,
            evaluation=evaluation,
        )

        self._log_report(
            observation=observation,
            evaluation=evaluation,
            records=crowded_sources,
        )

        return crowded_sources

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        *,
        observation: Observation,
        evaluation: PhotometryEvaluation,
    ) -> None:
        """
        Validate the completed workflow products.

        Raises
        ------
        ValueError
            If required workflow products are unavailable.
        """

        if observation is None:
            raise ValueError(
                "Observation cannot be None."
            )

        if evaluation is None:
            raise ValueError(
                "PhotometryEvaluation cannot be None."
            )

        if observation.detections is None:
            raise ValueError(
                "Observation does not contain source-detection results."
            )

    @staticmethod
    def _build_evaluation_lookup(
        evaluation: PhotometryEvaluation,
    ) -> dict[str, SourceEvaluation]:
        """
        Build a source-identifier lookup for source evaluations.

        Parameters
        ----------
        evaluation
            Completed photometric evaluation.

        Returns
        -------
        dict[str, SourceEvaluation]
            Source evaluations indexed by source identifier.

        Raises
        ------
        ValueError
            If a source evaluation contains an empty identifier or
            duplicate identifiers are present.
        """

        lookup: dict[str, SourceEvaluation] = {}

        for source_evaluation in evaluation.source_evaluations:

            identifier = source_evaluation.source_identifier

            if not identifier:
                raise ValueError(
                    "PhotometryEvaluation contains a source evaluation "
                    "with an empty source identifier."
                )

            if identifier in lookup:
                raise ValueError(
                    "PhotometryEvaluation contains duplicate source "
                    f"identifier '{identifier}'."
                )

            lookup[identifier] = source_evaluation

        return lookup

    @staticmethod
    def _validate_evaluation_completeness(
        *,
        detected_sources: tuple[DetectedSource, ...],
        evaluation_by_source: dict[str, SourceEvaluation],
    ) -> None:
        """
        Verify that every detected source has an evaluation record.

        The production evaluator is expected to produce exactly one
        SourceEvaluation for every detected source.

        Treating mismatches as errors prevents the extractor from
        silently producing an incomplete crowded-source report.

        Raises
        ------
        ValueError
            If one or more detected sources have no evaluation record,
            or if the evaluation contains orphan source records.
        """

        detected_identifiers = {
            source.identifier
            for source in detected_sources
        }

        evaluation_identifiers = set(
            evaluation_by_source,
        )

        missing_identifiers = (
            detected_identifiers
            - evaluation_identifiers
        )

        if missing_identifiers:

            identifiers = ", ".join(
                sorted(
                    missing_identifiers,
                ),
            )

            raise ValueError(
                "PhotometryEvaluation is incomplete. "
                "No source evaluation exists for detected source(s): "
                f"{identifiers}."
            )

        #
        # An evaluation record that does not correspond to any detected
        # source is also an internal consistency problem.
        #
        orphan_identifiers = (
            evaluation_identifiers
            - detected_identifiers
        )

        if orphan_identifiers:

            identifiers = ", ".join(
                sorted(
                    orphan_identifiers,
                ),
            )

            raise ValueError(
                "PhotometryEvaluation contains source evaluation "
                "record(s) that do not correspond to detected source(s): "
                f"{identifiers}."
            )

    @staticmethod
    def _validate_record(
        record: CrowdedSourceRecord,
    ) -> None:
        """
        Validate one extracted crowded-source record.

        Raises
        ------
        ValueError
            If the record contains physically invalid or internally
            inconsistent values.
        """

        if not record.source_identifier:
            raise ValueError(
                "Crowded-source record has an empty source identifier."
            )

        if not isfinite(
            record.x_pixel,
        ):
            raise ValueError(
                f"Source '{record.source_identifier}' has a non-finite "
                f"X coordinate: {record.x_pixel!r}."
            )

        if not isfinite(
            record.y_pixel,
        ):
            raise ValueError(
                f"Source '{record.source_identifier}' has a non-finite "
                f"Y coordinate: {record.y_pixel!r}."
            )

        if record.neighbour_count <= 0:
            raise ValueError(
                f"Source '{record.source_identifier}' is marked crowded "
                "but has no neighbouring sources."
            )

        if not isfinite(
            record.peak_value,
        ):
            raise ValueError(
                f"Source '{record.source_identifier}' has a non-finite "
                f"peak value: {record.peak_value!r}."
            )

        if not record.crowded:
            raise ValueError(
                f"Source '{record.source_identifier}' was included in "
                "the crowded-source result without crowded=True."
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _log_report(
        self,
        *,
        observation: Observation,
        evaluation: PhotometryEvaluation,
        records: tuple[CrowdedSourceRecord, ...],
    ) -> None:
        """
        Log a human-readable crowded-source report.
        """

        target_name = observation.metadata.target_name

        self._logger.info(
            "------------------------------------------------------------"
        )

        self._logger.info(
            "CROWDED SOURCE REPORT"
        )

        self._logger.info(
            "Target: %s",
            target_name,
        )

        self._logger.info(
            "Detected sources: %d",
            evaluation.detected_source_count,
        )

        self._logger.info(
            "Crowding radius: %.3f px",
            evaluation.crowding_radius,
        )

        self._logger.info(
            "Crowded sources: %d",
            len(records),
        )

        self._logger.info(
            "------------------------------------------------------------"
        )

        if not records:

            self._logger.info(
                "No crowded sources were identified."
            )

            self._logger.info(
                "------------------------------------------------------------"
            )

            return

        self._logger.info(
            "%-12s %-12s %-12s %-12s %-10s %-18s",
            "Source ID",
            "Pixel X",
            "Pixel Y",
            "Neighbours",
            "Valid",
            "Quality",
        )

        self._logger.info(
            "%-12s %-12s %-12s %-12s %-10s %-18s",
            "-" * 10,
            "-" * 10,
            "-" * 10,
            "-" * 10,
            "-" * 8,
            "-" * 16,
        )

        for record in records:

            self._logger.info(
                "%-12s %-12.3f %-12.3f %-12d %-10s %-18s",
                record.source_identifier,
                record.x_pixel,
                record.y_pixel,
                record.neighbour_count,
                (
                    "YES"
                    if record.valid_measurement
                    else "NO"
                ),
                record.quality_flag,
            )

        self._logger.info(
            "------------------------------------------------------------"
        )

    # ------------------------------------------------------------------
    # Convenience conversion
    # ------------------------------------------------------------------

    @staticmethod
    def to_rows(
        records: tuple[CrowdedSourceRecord, ...],
    ) -> tuple[dict[str, object], ...]:
        """
        Convert extracted records into serialisable row dictionaries.

        Parameters
        ----------
        records
            Crowded-source records returned by :meth:`extract`.

        Returns
        -------
        tuple[dict[str, object], ...]
            Immutable tuple of dictionaries suitable for CSV, JSON or
            dataframe conversion.

        Notes
        -----
        This method performs presentation/serialisation conversion only.

        It does not modify or recalculate any scientific result.
        """

        return tuple(
            {
                "source_identifier": record.source_identifier,
                "x_pixel": record.x_pixel,
                "y_pixel": record.y_pixel,
                "neighbour_count": record.neighbour_count,
                "crowded": record.crowded,
                "valid_measurement": record.valid_measurement,
                "quality_flag": record.quality_flag,
                "peak_value": record.peak_value,
            }
            for record in records
        )