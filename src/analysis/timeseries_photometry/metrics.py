"""Scientific metrics for paired TESS target light curves."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd


EPSILON: Final[float] = 1e-15


def _safe_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return math.nan
    return float(values.std(ddof=1))


def _safe_mad(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return math.nan
    median = float(values.median())
    return float(np.median(np.abs(values.to_numpy() - median)))


def _safe_ptp(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return math.nan
    return float(values.max() - values.min())


def _normalise_flux(series: pd.Series) -> pd.Series:
    """Median-normalise one flux series independently."""
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()

    if pd.isna(median) or abs(float(median)) <= EPSILON:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    return values / float(median)


def add_normalised_flux_columns(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Add independently median-normalised flux for each target.

    Baseline and SCARLET are normalised independently *within each target*.
    This is required because the seven TESS targets have different absolute
    source fluxes. The resulting columns therefore describe temporal
    variability rather than differences in target brightness.

    For every valid target:

        median(baseline_flux_normalised) == 1
        median(scarlet_flux_normalised) == 1
    """
    required = {
        "target_name",
        "baseline_net_flux",
        "scarlet_net_flux",
    }
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(
            "Cannot normalise paired flux: missing required columns: "
            + ", ".join(missing)
        )

    result = paired.copy()

    result["baseline_flux_normalised"] = np.nan
    result["scarlet_flux_normalised"] = np.nan

    for _, index in result.groupby(
        "target_name",
        sort=False,
        dropna=False,
    ).groups.items():
        result.loc[index, "baseline_flux_normalised"] = (
            _normalise_flux(result.loc[index, "baseline_net_flux"]).to_numpy()
        )
        result.loc[index, "scarlet_flux_normalised"] = (
            _normalise_flux(result.loc[index, "scarlet_net_flux"]).to_numpy()
        )

    return result


def validate_normalised_flux(
    paired: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> None:
    """Validate that normalised light curves are centred on median 1.

    Raises ValueError if a target has finite normalised flux values but its
    median differs from 1 beyond the supplied tolerance.
    """
    required = {
        "target_name",
        "baseline_flux_normalised",
        "scarlet_flux_normalised",
    }
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(
            "Cannot validate normalised flux: missing required columns: "
            + ", ".join(missing)
        )

    for raw_target, group in paired.groupby(
        "target_name",
        sort=False,
        dropna=False,
    ):
        target = str(raw_target)

        for pipeline in ("baseline", "scarlet"):
            values = pd.to_numeric(
                group[f"{pipeline}_flux_normalised"],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna()

            if values.empty:
                continue

            median = float(values.median())
            if not math.isclose(median, 1.0, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(
                    f"{target}: {pipeline} normalised-flux median is "
                    f"{median:.15g}; expected 1.0."
                )


def _correlation(
    x: pd.Series,
    y: pd.Series,
) -> float:
    frame = pd.concat([x, y], axis=1).dropna()

    if len(frame) < 3:
        return math.nan

    if frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return math.nan

    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def _pipeline_metrics(
    flux: pd.Series,
    magnitude: pd.Series,
) -> dict[str, float]:
    """Calculate precision and temporal-stability metrics for one pipeline."""
    normalised = _normalise_flux(flux)

    finite_flux = (
        normalised
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    magnitude_values = (
        pd.to_numeric(magnitude, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(finite_flux) >= 2:
        point_to_point = finite_flux.diff().dropna()
    else:
        point_to_point = pd.Series(dtype="float64")

    return {
        "flux_median": float(
            pd.to_numeric(flux, errors="coerce").median()
        ),
        "flux_scatter_std": _safe_std(finite_flux),
        "flux_scatter_mad": _safe_mad(finite_flux),
        "flux_range": _safe_ptp(finite_flux),
        "magnitude_scatter_std": _safe_std(magnitude_values),
        "magnitude_scatter_mad": _safe_mad(magnitude_values),
        "temporal_stability_std_diff": _safe_std(point_to_point),
        "temporal_stability_mad_diff": _safe_mad(point_to_point),
        "valid_cadence_count": float(finite_flux.count()),
    }


def calculate_target_metrics(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate paired metrics for each target.

    Primary scatter quantities are calculated from independently
    median-normalised target fluxes. Temporal stability is calculated from
    cadence-ordered normalised flux, using the standard deviation and MAD of
    consecutive-cadence differences.

    Raw flux values are retained as descriptive context and are not
    interpreted as precision.
    """
    required = {
        "target_name",
        "pair_status",
        "baseline_net_flux",
        "scarlet_net_flux",
        "baseline_magnitude",
        "scarlet_magnitude",
    }
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(
            "Cannot calculate target metrics: missing required columns: "
            + ", ".join(missing)
        )

    paired_valid = paired.loc[
        paired["pair_status"].eq("paired")
    ].copy()

    rows: list[dict[str, object]] = []

    for raw_target, group in paired_valid.groupby(
        "target_name",
        sort=True,
    ):
        target = str(raw_target)

        # Temporal stability is defined between consecutive cadences.
        # Explicitly sort here so that input row order cannot alter the
        # temporal metric.
        if "cadence_index" in group.columns:
            group = group.sort_values(
                "cadence_index",
                kind="stable",
            ).reset_index(drop=True)
        else:
            group = group.reset_index(drop=True)

        baseline = _pipeline_metrics(
            group["baseline_net_flux"],
            group["baseline_magnitude"],
        )

        scarlet = _pipeline_metrics(
            group["scarlet_net_flux"],
            group["scarlet_magnitude"],
        )

        baseline_flux = _normalise_flux(group["baseline_net_flux"])
        scarlet_flux = _normalise_flux(group["scarlet_net_flux"])

        row: dict[str, object] = {
            "target_name": target,
            "paired_cadence_count": int(len(group)),
            "baseline_experiment_id": (
                str(group["baseline_experiment_id"].dropna().iloc[0])
                if "baseline_experiment_id" in group.columns
                and group["baseline_experiment_id"].notna().any()
                else ""
            ),
            "scarlet_experiment_id": (
                str(group["scarlet_experiment_id"].dropna().iloc[0])
                if "scarlet_experiment_id" in group.columns
                and group["scarlet_experiment_id"].notna().any()
                else ""
            ),
            "lightcurve_correlation": _correlation(
                baseline_flux,
                scarlet_flux,
            ),
        }

        for prefix, values in (
            ("baseline", baseline),
            ("scarlet", scarlet),
        ):
            row.update(
                {
                    f"{prefix}_{key}": value
                    for key, value in values.items()
                }
            )

        for metric in (
            "flux_scatter_std",
            "flux_scatter_mad",
            "magnitude_scatter_std",
            "magnitude_scatter_mad",
            "temporal_stability_std_diff",
            "temporal_stability_mad_diff",
        ):
            baseline_value = row[f"baseline_{metric}"]
            scarlet_value = row[f"scarlet_{metric}"]

            if (
                isinstance(baseline_value, (int, float))
                and isinstance(scarlet_value, (int, float))
                and math.isfinite(float(baseline_value))
                and abs(float(baseline_value)) > EPSILON
                and math.isfinite(float(scarlet_value))
            ):
                change = (
                    (float(scarlet_value) - float(baseline_value))
                    / float(baseline_value)
                    * 100.0
                )
            else:
                change = math.nan

            row[f"{metric}_change_percent"] = change

        rows.append(row)

    return pd.DataFrame(rows)


def classify_scatter_change(change_percent: float) -> str:
    """Classify direction descriptively, without assigning scientific value."""
    if not math.isfinite(change_percent):
        return "not_available"

    if change_percent < 0:
        return "lower_scatter"

    if change_percent > 0:
        return "higher_scatter"

    return "unchanged"
