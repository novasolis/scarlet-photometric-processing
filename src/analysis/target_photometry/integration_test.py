from .io import (
    load_experiment_summary,
    load_source_measurements,
)

from .validation import (
    validate_input_data,
)

from .dataset import (
    extract_local_field_measurements,
    validate_local_field_dataset,
    attach_experiment_metadata,
    validate_enriched_local_field_dataset,
)

from .canonical import (
    build_canonical_dataset,
    validate_canonical_dataset,
)

from .pairing import (
    build_paired_dataset,
    validate_paired_dataset,
)

from .metrics import (
    calculate_paired_metrics,
    validate_paired_metrics,
)

from .statistics import (
    build_descriptive_statistics,
    validate_descriptive_statistics,
)


def main():

    print(
        "Running TARGET_PHOTOMETRY analytical integration test..."
    )
    print()

    # ------------------------------------------------------------------
    # Load frozen production outputs
    # ------------------------------------------------------------------

    summary = load_experiment_summary()
    measurements = load_source_measurements()

    print(
        f"[PASS] Loaded experiment summary: "
        f"{len(summary)} rows"
    )

    print(
        f"[PASS] Loaded source measurements: "
        f"{len(measurements)} rows"
    )

    # ------------------------------------------------------------------
    # Validate raw production inputs
    # ------------------------------------------------------------------

    validate_input_data(
        summary,
        measurements,
    )

    print(
        "[PASS] Frozen production inputs validated"
    )

    # ------------------------------------------------------------------
    # Extract complete local-field dataset
    # ------------------------------------------------------------------

    local_field = extract_local_field_measurements(
        measurements
    )

    validate_local_field_dataset(
        local_field
    )

    print(
        "[PASS] Local-field dataset extracted"
    )

    print(
        f"[PASS] Local-field dataset contains "
        f"{len(local_field)} rows"
    )

    # ------------------------------------------------------------------
    # Attach experiment-level metadata
    # ------------------------------------------------------------------

    local_field = attach_experiment_metadata(
        local_field,
        summary,
    )

    validate_enriched_local_field_dataset(
        local_field
    )

    print(
        "[PASS] Experiment metadata attached"
    )

    print(
        f"[PASS] Enriched local-field dataset contains "
        f"{len(local_field)} rows"
    )

    # ------------------------------------------------------------------
    # Build canonical analytical dataset
    # ------------------------------------------------------------------

    canonical = build_canonical_dataset(
        local_field
    )

    validate_canonical_dataset(
        canonical
    )

    print(
        "[PASS] Canonical analytical dataset created"
    )

    print(
        f"[PASS] Canonical dataset contains "
        f"{len(canonical)} rows × "
        f"{len(canonical.columns)} columns"
    )

    # ------------------------------------------------------------------
    # Build paired Baseline / SCARLET dataset
    # ------------------------------------------------------------------

    paired = build_paired_dataset(
        canonical
    )

    validate_paired_dataset(
        paired
    )

    print(
        "[PASS] Paired Baseline/SCARLET dataset created"
    )

    print(
        f"[PASS] Paired dataset contains "
        f"{len(paired)} rows × "
        f"{len(paired.columns)} columns"
    )

    # ------------------------------------------------------------------
    # Calculate paired photometric metrics
    # ------------------------------------------------------------------

    metrics = calculate_paired_metrics(
        paired
    )

    validate_paired_metrics(
        metrics
    )

    print(
        "[PASS] Paired photometric metrics calculated"
    )

    print(
        f"[PASS] Metrics dataset contains "
        f"{len(metrics)} rows × "
        f"{len(metrics.columns)} columns"
    )

    # ------------------------------------------------------------------
    # Build descriptive statistics
    # ------------------------------------------------------------------

    statistics = build_descriptive_statistics(
        metrics
    )

    validate_descriptive_statistics(
        statistics
    )

    print(
        "[PASS] Descriptive statistics calculated"
    )

    print(
        f"[PASS] Statistics dataset contains "
        f"{len(statistics)} rows × "
        f"{len(statistics.columns)} columns"
    )

    # ------------------------------------------------------------------
    # Analytical coverage summary
    # ------------------------------------------------------------------

    target_count = int(
        paired["is_target_source"].sum()
    )

    neighbour_count = int(
        (~paired["is_target_source"]).sum()
    )

    sdss_count = int(
        (paired["survey"] == "SDSS").sum()
    )

    tess_count = int(
        (paired["survey"] == "TESS").sum()
    )

    print()
    print("Analytical coverage:")
    print(
        f"  - Paired sources: {len(paired)}"
    )
    print(
        f"  - Target sources: {target_count}"
    )
    print(
        f"  - Neighbour sources: {neighbour_count}"
    )
    print(
        f"  - SDSS sources: {sdss_count}"
    )
    print(
        f"  - TESS sources: {tess_count}"
    )

    # ------------------------------------------------------------------
    # Preview final analytical metrics
    # ------------------------------------------------------------------

    print()
    print("Paired metrics preview:")
    print()

    preview_columns = [
        "target_name",
        "survey",
        "source_identifier",
        "source_role",
        "net_flux_baseline",
        "net_flux_scarlet",
        "flux_difference",
        "flux_relative_difference_percent",
        "absolute_flux_relative_difference_percent",
        "magnitude_difference",
        "absolute_magnitude_difference",
    ]

    print(
        metrics[preview_columns]
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Preview descriptive statistics
    # ------------------------------------------------------------------

    print()
    print("Descriptive statistics preview:")
    print()

    print(
        statistics.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    print()
    print(
        "TARGET_PHOTOMETRY ANALYTICAL INTEGRATION TEST PASSED"
    )


if __name__ == "__main__":
    main()