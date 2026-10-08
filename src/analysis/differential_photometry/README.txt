Differential Photometry RQ4 analysis

Run from the parent directory containing differential_photometry/:

python -m differential_photometry.section_5_6_results ^
  --differential "path/to/differential_timeseries_measurements.csv" ^
  --timeseries "path/to/timeseries_measurements.csv" ^
  --summary "path/to/experiment_summary.csv" ^
  --output differential_results

Inputs:
- differential_timeseries_measurements.csv
- timeseries_measurements.csv
- experiment_summary.csv

Outputs:
- differential_measurement_summary.csv
- differential_timeseries_metrics.csv
- differential_metric_changes.csv
- differential_light_curve_correlations.csv
- target_timeseries_comparison_metrics.csv
- figure_5_6_1_differential_measurements.png
- figure_5_6_2_differential_stability.png
- section_5_6_results.txt
