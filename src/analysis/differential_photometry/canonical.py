"""Canonical experiment definitions for differential-photometry RQ4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DifferentialPhotometryExperiment:
    """One target/pipeline differential-photometry experiment."""

    experiment_id: str
    target_name: str
    pipeline: str
    survey: str
    experiment_type: str
    n_cadences: int


def load_experiments(summary_csv: str | Path) -> tuple[DifferentialPhotometryExperiment, ...]:
    """Load the canonical experiment set from experiment_summary.csv."""
    df = pd.read_csv(summary_csv)

    required = {
        "experiment_id",
        "target_name",
        "pipeline",
        "survey",
        "experiment_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"experiment_summary.csv is missing columns: {sorted(missing)}")

    experiments = []
    for row in df.itertuples(index=False):
        experiments.append(
            DifferentialPhotometryExperiment(
                experiment_id=str(row.experiment_id),
                target_name=str(row.target_name),
                pipeline=str(row.pipeline),
                survey=str(row.survey),
                experiment_type=str(row.experiment_type),
                n_cadences=0,
            )
        )
    return tuple(experiments)
