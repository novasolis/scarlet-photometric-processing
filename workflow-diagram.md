# Photometric Analysis Pipeline

## Purpose

The experimental workflow compares two photometric processing paths:

- **BASELINE** — aperture photometry applied to the original observation.
- **SCARLET** — SCARLET source-separation preprocessing followed by the same downstream photometric measurement workflow.

The comparison is designed to isolate how preprocessing changes the resulting photometric measurements.

## Overall Workflow

```text
                         ExperimentConfig
                                │
                                ▼
                       PhotometricWorkflow
                                │
                                ▼
                       BackgroundSubtractor
                                │
                                ▼
                 Background-subtracted observation
                                │
                       ┌────────┴────────┐
                       │                 │
                       ▼                 ▼
                   BASELINE          SCARLET
                       │                 │
                       │                 ▼
                       │        ScarletPreprocessor
                       │                 │
                       │                 ▼
                       │        SCARLET preparation
                       │                 │
                       │                 ▼
                       │          SCARLET fitting
                       │                 │
                       │                 ▼
                       │          Target component
                       │                 │
                       │                 ▼
                       │        Project ImageData
                       │                 │
                       └────────┬────────┘
                                ▼
                         SourceDetector
                                │
                                ▼
                     TargetSourceResolver
                                │
                                ▼
                       TargetSourceMatch
                                │
                                ▼
                       AperturePhotometer
                                │
                                ▼
                       PhotometryResult
                                │
                                ▼
                        WorkflowResult
                                │
                                ▼
                  Comparative evaluation
```

The two branches share the same downstream source detection, target
resolution and aperture-photometry stages. The principal difference is
whether SCARLET source separation is applied before photometric
measurement.

## Baseline Branch

```text
Observation
    │
    ▼
Background subtraction
    │
    ▼
Source detection
    │
    ▼
Target resolution
    │
    ▼
Aperture photometry
    │
    ▼
PhotometryResult
```

The baseline branch provides the reference measurement for comparison
with the SCARLET-preprocessed branch.

## SCARLET Branch

```text
Observation
    │
    ▼
Background subtraction
    │
    ▼
SCARLET preparation
    │
    ▼
Project → SCARLET observation representation
    │
    ▼
SCARLET fitting
    │
    ▼
Target component
    │
    ▼
SCARLET → Project ImageData
    │
    ▼
Source detection
    │
    ▼
Target resolution
    │
    ▼
Aperture photometry
    │
    ▼
PhotometryResult
```

The SCARLET branch introduces source separation before the common
photometric measurement stages. The recovered target component is
converted back to the project's `ImageData` representation so that the
downstream measurement path remains consistent with the baseline.

## Common Photometric Path

Following either processing branch:

```text
Processed observation
        │
        ▼
SourceDetector
        │
        ▼
TargetSourceResolver
        │
        ▼
TargetSourceMatch
        │
        ▼
AperturePhotometer
        │
        ▼
PhotometryResult
```

Using a common downstream path is intentional. It allows the
experimental comparison to focus on the effect of preprocessing rather
than introducing a different photometric measurement method.

## Experimental Configuration

`ExperimentConfig` defines the configuration used for an experimental
run, including the selected processing pipeline.

```text
                    ExperimentConfig
                           │
                           ▼
                   ProcessingPipeline
                      /          \
                     /            \
                    ▼              ▼
               BASELINE         SCARLET
```

The selected pipeline determines whether the observation proceeds
directly to the common photometric stages or first passes through
SCARLET preprocessing.

## Output

The workflow produces a `PhotometryResult`, which is incorporated into
the broader `WorkflowResult`.

For TESS observations, measurements can subsequently be organised as a
time-series for comparative analysis of the baseline and SCARLET
photometric behaviour.

```text
WorkflowResult
      │
      ▼
Photometric measurements
      │
      ├── Baseline measurements
      │
      └── SCARLET measurements
              │
              ▼
      Comparative evaluation
```

## Experimental Comparison

The workflow supports comparison of the resulting measurements in
terms of:

- measured photometric flux;
- measurement variability;
- correspondence between baseline and SCARLET measurements;
- preservation of temporal signal structure for applicable TESS
  observations.

A change in measured flux is therefore treated as an experimental
observation, not by itself as evidence of improved photometric accuracy.

## Scope

This diagram describes the workflow used by the experimental
implementation published with the thesis. It documents the processing
structure rather than defining a general-purpose photometry framework.

The repository is currently published as a research snapshot. Future
development may simplify the implementation and documentation without
changing the conceptual comparison described here.
