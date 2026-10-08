from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# TARGET_PHOTOMETRY production outputs
# ---------------------------------------------------------------------------

DATA_DIR = Path("results/exports/target_photometry")

EXPERIMENT_SUMMARY_FILE = DATA_DIR / "experiment_summary.csv"
SOURCE_MEASUREMENTS_FILE = DATA_DIR / "source_measurements.csv"


# ---------------------------------------------------------------------------
# Experiment-level data
# ---------------------------------------------------------------------------

def load_experiment_summary() -> pd.DataFrame:
    """
    Load the TARGET_PHOTOMETRY experiment-level summary.

    One row represents one experiment/pipeline execution.
    """

    if not EXPERIMENT_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Experiment summary not found: "
            f"{EXPERIMENT_SUMMARY_FILE}"
        )

    return pd.read_csv(
        EXPERIMENT_SUMMARY_FILE
    )


# ---------------------------------------------------------------------------
# Source-level data
# ---------------------------------------------------------------------------

def load_source_measurements() -> pd.DataFrame:
    """
    Load the complete TARGET_PHOTOMETRY source-level measurements.

    The dataset contains measurements for the full local field processed
    by both the Baseline and SCARLET pipelines.

    Each experiment therefore contains:
        - one designated target source
        - neighbouring local-field sources
    """

    if not SOURCE_MEASUREMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Source measurements not found: "
            f"{SOURCE_MEASUREMENTS_FILE}"
        )

    return pd.read_csv(
        SOURCE_MEASUREMENTS_FILE
    )