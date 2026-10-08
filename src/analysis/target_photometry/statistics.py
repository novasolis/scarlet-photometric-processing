import pandas as pd


# ---------------------------------------------------------------------------
# TARGET_PHOTOMETRY descriptive statistics
# ---------------------------------------------------------------------------

STATISTIC_COLUMNS = [
    "count",
    "mean",
    "median",
    "std",
    "min",
    "q1",
    "q3",
    "max",
]


ANALYSIS_METRICS = [
    "flux_difference",
    "flux_relative_difference",
    "flux_relative_difference_percent",
    "absolute_flux_relative_difference_percent",
    "magnitude_difference",
    "absolute_magnitude_difference",
]


# ---------------------------------------------------------------------------
# Single-metric summary
# ---------------------------------------------------------------------------


def summarize_metric(
    data: pd.DataFrame,
    metric: str,
) -> dict:
    """Calculate descriptive statistics for a single metric."""

    if metric not in data.columns:
        raise ValueError(
            f"Metric '{metric}' not found in dataset."
        )

    values = data[metric].dropna()

    if values.empty:
        raise ValueError(
            f"No valid observations available for metric '{metric}'."
        )

    return {
        "count": int(values.count()),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(ddof=1),
        "min": values.min(),
        "q1": values.quantile(0.25),
        "q3": values.quantile(0.75),
        "max": values.max(),
    }


# ---------------------------------------------------------------------------
# Grouped statistics
# ---------------------------------------------------------------------------


def build_descriptive_statistics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build descriptive statistics for the complete source-level analysis.

    Groups are deliberately consolidated into one table:

        ALL
        TARGET
        NEIGHBOUR
        SDSS
        TESS
        SDSS_TARGET
        SDSS_NEIGHBOUR
        TESS_TARGET
        TESS_NEIGHBOUR

    This supports target-level, local-field, and cross-survey analysis
    without maintaining separate statistics pipelines.
    """

    required_columns = [
        "survey",
        "is_target_source",
        *ANALYSIS_METRICS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metrics.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot calculate descriptive statistics. "
            f"Missing columns: {missing_columns}"
        )

    groups = {
        "ALL": metrics,
        "TARGET": metrics[
            metrics["is_target_source"]
        ],
        "NEIGHBOUR": metrics[
            ~metrics["is_target_source"]
        ],
        "SDSS": metrics[
            metrics["survey"] == "SDSS"
        ],
        "TESS": metrics[
            metrics["survey"] == "TESS"
        ],
        "SDSS_TARGET": metrics[
            (metrics["survey"] == "SDSS")
            & metrics["is_target_source"]
        ],
        "SDSS_NEIGHBOUR": metrics[
            (metrics["survey"] == "SDSS")
            & ~metrics["is_target_source"]
        ],
        "TESS_TARGET": metrics[
            (metrics["survey"] == "TESS")
            & metrics["is_target_source"]
        ],
        "TESS_NEIGHBOUR": metrics[
            (metrics["survey"] == "TESS")
            & ~metrics["is_target_source"]
        ],
    }

    rows = []

    for group_name, group_data in groups.items():
        if group_data.empty:
            continue

        for metric in ANALYSIS_METRICS:
            summary = summarize_metric(
                group_data,
                metric,
            )

            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    **summary,
                }
            )

    statistics = pd.DataFrame(
        rows,
        columns=[
            "group",
            "metric",
            *STATISTIC_COLUMNS,
        ],
    )

    return statistics


# ---------------------------------------------------------------------------
# Statistics validation
# ---------------------------------------------------------------------------


def validate_descriptive_statistics(
    statistics: pd.DataFrame,
) -> None:
    """Validate the consolidated descriptive-statistics dataset."""

    required_columns = [
        "group",
        "metric",
        *STATISTIC_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in statistics.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing descriptive-statistics columns: "
            f"{missing_columns}"
        )

    if statistics.empty:
        raise ValueError("Descriptive-statistics dataset is empty.")

    observed_metrics = set(statistics["metric"])
    expected_metrics = set(ANALYSIS_METRICS)

    if observed_metrics != expected_metrics:
        raise ValueError(
            "Unexpected metric set in descriptive statistics. "
            f"Expected: {expected_metrics}; "
            f"Found: {observed_metrics}"
        )

    duplicate_rows = statistics.duplicated(
        subset=["group", "metric"]
    )

    if duplicate_rows.any():
        raise ValueError(
            "Duplicate group/metric rows found in descriptive statistics."
        )

    if (statistics["count"] <= 0).any():
        raise ValueError(
            "Descriptive-statistics groups must contain observations."
        )

    non_std_columns = [
        column
        for column in STATISTIC_COLUMNS
        if column != "std"
    ]

    if statistics[non_std_columns].isna().any().any():
        raise ValueError(
            "Missing values found in descriptive statistics."
        )

    # Sample standard deviation is mathematically undefined for a single
    # observation, so NaN is permitted only in that case.
    invalid_std = statistics["std"].isna() & (statistics["count"] > 1)

    if invalid_std.any():
        raise ValueError(
            "Undefined standard deviations found for groups with more than "
            "one observation."
        )
