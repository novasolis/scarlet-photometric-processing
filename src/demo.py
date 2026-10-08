"""
Demonstration entry point.

Executes the configured experiment suite and visualises the intermediate
Observation products produced by the photometric analysis workflow.

Unlike ``main.py``, which is intended for production execution, this entry
point is designed for interactive exploration, development, and thesis
figures.

The demo layer is deliberately presentation-oriented:
- it does not mutate the production experiment configuration;
- it may limit time-series cadence processing for faster interactive runs;
- it delegates scientific processing to the configured workflow;
- it delegates rendering to ``demos.observation_demo``.

SCARLET field diagnostics are presentation-only. When enabled, they visualise
which detected field sources would be supplied to SCARLET under the configured
source cap; they do not execute SCARLET fitting/deblending themselves.

Set ``MAX_CADENCES = None`` to process the full configured time series.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from config.experiments import EXPERIMENTS
from domain.enums import ExperimentType
from demos.observation_demo import demonstrate
from utils.logger import configure_logging


# ---------------------------------------------------------------------------
# Demo execution settings
# ---------------------------------------------------------------------------

# Maximum number of cadence indices processed for enabled time-series
# experiments during an interactive demo run.
#
# 1500 provides substantial temporal coverage for light-curve inspection
# while keeping interactive demonstrations more manageable than a full
# configured time series.
#
# Set to None to process the complete configured time series.
MAX_CADENCES: int | None = 1500

# Catalogue crossmatching can add network/runtime overhead. It remains
# enabled by default because catalogue information is useful to the demo
# and is already an explicit option of ``demonstrate``.
RUN_CATALOGUE_CROSSMATCH: bool = True

# ---------------------------------------------------------------------------
# SCARLET field visualisation
# ---------------------------------------------------------------------------

# These diagnostics show the source-selection boundary before SCARLET
# construction/fitting. They are deliberately disabled by default so the
# normal demo remains unchanged unless SCARLET diagnostics are requested.
SHOW_SCARLET_FIELD_SELECTION: bool = True

# Match the source cap used by the current SCARLET field experiments.
#
# Example:
#   380 detected sources -> 50 selected sources
#
# Use None to visualise all detected field sources as SCARLET candidates.
SCARLET_FIELD_MAX_SOURCES: int | None = 50


def _validate_demo_settings() -> None:
    """Validate module-level demo settings before execution."""

    if MAX_CADENCES is not None:
        if isinstance(MAX_CADENCES, bool) or not isinstance(MAX_CADENCES, int):
            raise ValueError(
                "MAX_CADENCES must be a positive integer or None; "
                f"received {MAX_CADENCES!r}."
            )
        if MAX_CADENCES <= 0:
            raise ValueError(
                "MAX_CADENCES must be a positive integer or None; "
                f"received {MAX_CADENCES!r}."
            )

    if not isinstance(SHOW_SCARLET_FIELD_SELECTION, bool):
        raise ValueError(
            "SHOW_SCARLET_FIELD_SELECTION must be a boolean; "
            f"received {SHOW_SCARLET_FIELD_SELECTION!r}."
        )

    if SCARLET_FIELD_MAX_SOURCES is not None:
        if (
            isinstance(SCARLET_FIELD_MAX_SOURCES, bool)
            or not isinstance(SCARLET_FIELD_MAX_SOURCES, int)
        ):
            raise ValueError(
                "SCARLET_FIELD_MAX_SOURCES must be a positive integer or None; "
                f"received {SCARLET_FIELD_MAX_SOURCES!r}."
            )
        if SCARLET_FIELD_MAX_SOURCES <= 0:
            raise ValueError(
                "SCARLET_FIELD_MAX_SOURCES must be a positive integer or None; "
                f"received {SCARLET_FIELD_MAX_SOURCES!r}."
            )


def _limit_demo_cadences(config: Any) -> Any:
    """
    Return a demo-only copy of an experiment configuration with its
    time-series cadence indices limited to ``MAX_CADENCES``.

    The original production configuration is never mutated.

    Non-time-series experiments are returned unchanged.

    If an enabled time-series experiment has no explicit cadence indices,
    the demo creates a sequence beginning at cadence zero. The acquisition
    and workflow layers remain responsible for validating those indices
    against the actual dataset.

    Parameters
    ----------
    config:
        Experiment configuration containing a ``time_series`` attribute.

    Returns
    -------
    Any
        The original configuration when no limit is required, otherwise a
        dataclass copy containing a demo-limited time-series configuration.

    Raises
    ------
    ValueError
        If ``MAX_CADENCES`` is configured as zero or a negative value.
    """

    if MAX_CADENCES is None:
        return config

    time_series = config.time_series

    if not time_series.enabled:
        return config

    cadence_indices = time_series.cadence_indices

    if cadence_indices is None:
        limited_cadence_indices = tuple(range(MAX_CADENCES))
    else:
        limited_cadence_indices = tuple(cadence_indices[:MAX_CADENCES])

    limited_time_series = replace(
        time_series,
        cadence_indices=limited_cadence_indices,
    )

    return replace(
        config,
        time_series=limited_time_series,
    )


def main() -> None:
    """
    Execute the configured demonstration suite.

    Each experiment receives an independent demo-only configuration copy.

    Exceptions are deliberately allowed to propagate so a failed
    demonstration cannot be mistaken for a successful run.
    """

    _validate_demo_settings()
    configure_logging()

    for config in EXPERIMENTS:
        demo_config = _limit_demo_cadences(config)

        # FIELD_ANALYSIS is a source-discovery/crowding stage only.
        # Keep the demo presentation layer from requesting SCARLET-specific
        # field-selection visualisation for this experiment type.
        show_scarlet_field_selection = (
            SHOW_SCARLET_FIELD_SELECTION
            and config.experiment_type is not ExperimentType.FIELD_ANALYSIS
        )

        demonstrate(
            demo_config,
            run_catalogue_crossmatch=RUN_CATALOGUE_CROSSMATCH,
            show_scarlet_field_selection=show_scarlet_field_selection,
            scarlet_field_max_sources=SCARLET_FIELD_MAX_SOURCES,
        )


if __name__ == "__main__":
    main()
