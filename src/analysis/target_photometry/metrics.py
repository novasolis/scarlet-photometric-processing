import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# TARGET_PHOTOMETRY paired metrics
# ---------------------------------------------------------------------------

def calculate_paired_metrics(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate paired Baseline-vs-SCARLET photometric differences.

    The input contains one row per experiment/source pair with Baseline
    and SCARLET measurements side-by-side.

    Derived quantities:

        flux_difference
            SCARLET net flux - Baseline net flux.

        flux_relative_difference
            Relative change in net flux with respect to Baseline.

        flux_relative_difference_percent
            Relative change in net flux with respect to Baseline,
            expressed as a percentage.

        absolute_flux_relative_difference_percent
            Absolute value of the relative flux difference percentage.

        magnitude_difference
            SCARLET instrumental magnitude - Baseline instrumental
            magnitude.

        absolute_magnitude_difference
            Absolute value of the instrumental-magnitude difference.

    Non-finite source measurements are retained in the paired dataset.
    Derived metrics are calculated only when the required inputs are
    finite and mathematically valid. Otherwise, the derived value is NaN.
    """

    required_columns = [
        "net_flux_baseline",
        "net_flux_scarlet",
        "instrumental_magnitude_baseline",
        "instrumental_magnitude_scarlet",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in paired.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot calculate paired metrics. "
            f"Missing columns: {missing_columns}"
        )

    metrics = paired.copy()

    # ------------------------------------------------------------------
    # Validity masks
    # ------------------------------------------------------------------

    baseline_flux = pd.to_numeric(
        metrics["net_flux_baseline"],
        errors="coerce",
    )

    scarlet_flux = pd.to_numeric(
        metrics["net_flux_scarlet"],
        errors="coerce",
    )

    baseline_magnitude = pd.to_numeric(
        metrics["instrumental_magnitude_baseline"],
        errors="coerce",
    )

    scarlet_magnitude = pd.to_numeric(
        metrics["instrumental_magnitude_scarlet"],
        errors="coerce",
    )

    finite_flux_pair = (
        np.isfinite(baseline_flux)
        & np.isfinite(scarlet_flux)
    )

    finite_magnitude_pair = (
        np.isfinite(baseline_magnitude)
        & np.isfinite(scarlet_magnitude)
    )

    # ------------------------------------------------------------------
    # Net-flux differences
    # ------------------------------------------------------------------

    metrics["flux_difference"] = np.nan

    metrics.loc[finite_flux_pair, "flux_difference"] = (
        scarlet_flux[finite_flux_pair]
        - baseline_flux[finite_flux_pair]
    )

    # Relative flux difference requires a finite, non-zero baseline.
    valid_relative_flux = (
        finite_flux_pair
        & (baseline_flux != 0)
    )

    metrics["flux_relative_difference"] = np.nan

    metrics.loc[valid_relative_flux, "flux_relative_difference"] = (
        (
            scarlet_flux[valid_relative_flux]
            - baseline_flux[valid_relative_flux]
        )
        / baseline_flux[valid_relative_flux]
    )

    metrics["flux_relative_difference_percent"] = (
        metrics["flux_relative_difference"] * 100.0
    )

    metrics["absolute_flux_relative_difference_percent"] = (
        metrics["flux_relative_difference_percent"].abs()
    )

    # ------------------------------------------------------------------
    # Instrumental-magnitude differences
    # ------------------------------------------------------------------

    metrics["magnitude_difference"] = np.nan

    metrics.loc[finite_magnitude_pair, "magnitude_difference"] = (
        scarlet_magnitude[finite_magnitude_pair]
        - baseline_magnitude[finite_magnitude_pair]
    )

    metrics["absolute_magnitude_difference"] = (
        metrics["magnitude_difference"].abs()
    )

    return metrics


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_paired_metrics(
    metrics: pd.DataFrame,
) -> None:
    """
    Validate the TARGET_PHOTOMETRY paired metrics dataset.

    The complete local-field population is retained. Therefore, individual
    derived metrics may legitimately be NaN when one or both source
    measurements are non-finite or mathematically unsuitable for the
    corresponding calculation.
    """

    required_columns = [
        "flux_difference",
        "flux_relative_difference",
        "flux_relative_difference_percent",
        "absolute_flux_relative_difference_percent",
        "magnitude_difference",
        "absolute_magnitude_difference",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metrics.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing derived metric columns: "
            f"{missing_columns}"
        )

    if metrics.empty:
        raise ValueError(
            "Paired metrics dataset is empty."
        )

    # ------------------------------------------------------------------
    # Validate finite values where metrics are defined
    # ------------------------------------------------------------------

    for column in required_columns:
        values = metrics[column]

        non_finite_defined = (
            values.notna()
            & ~np.isfinite(values)
        )

        if non_finite_defined.any():
            raise ValueError(
                f"Column '{column}' contains non-finite values "
                "where a metric is defined."
            )

    # ------------------------------------------------------------------
    # Absolute flux-relative differences
    # ------------------------------------------------------------------

    defined_absolute_flux = metrics[
        "absolute_flux_relative_difference_percent"
    ].notna()

    if not (
        metrics.loc[
            defined_absolute_flux,
            "absolute_flux_relative_difference_percent",
        ] >= 0
    ).all():
        raise ValueError(
            "Absolute flux relative difference percentages "
            "must be non-negative."
        )

    # ------------------------------------------------------------------
    # Absolute magnitude differences
    # ------------------------------------------------------------------

    defined_absolute_magnitude = metrics[
        "absolute_magnitude_difference"
    ].notna()

    if not (
        metrics.loc[
            defined_absolute_magnitude,
            "absolute_magnitude_difference",
        ] >= 0
    ).all():
        raise ValueError(
            "Absolute magnitude differences must be non-negative."
        )

    # ------------------------------------------------------------------
    # Internal consistency
    # ------------------------------------------------------------------

    defined_flux = metrics[
        "flux_relative_difference"
    ].notna()

    if not np.allclose(
        metrics.loc[
            defined_flux,
            "flux_relative_difference_percent",
        ],
        metrics.loc[
            defined_flux,
            "flux_relative_difference",
        ] * 100.0,
        equal_nan=True,
    ):
        raise ValueError(
            "Flux relative difference percentage is inconsistent "
            "with flux relative difference."
        )

    if not np.allclose(
        metrics.loc[
            defined_flux,
            "absolute_flux_relative_difference_percent",
        ],
        metrics.loc[
            defined_flux,
            "flux_relative_difference_percent",
        ].abs(),
        equal_nan=True,
    ):
        raise ValueError(
            "Absolute flux relative difference percentage is "
            "inconsistent with flux relative difference percentage."
        )

    defined_magnitude = metrics[
        "magnitude_difference"
    ].notna()

    if not np.allclose(
        metrics.loc[
            defined_magnitude,
            "absolute_magnitude_difference",
        ],
        metrics.loc[
            defined_magnitude,
            "magnitude_difference",
        ].abs(),
        equal_nan=True,
    ):
        raise ValueError(
            "Absolute magnitude difference is inconsistent "
            "with magnitude difference."
        )